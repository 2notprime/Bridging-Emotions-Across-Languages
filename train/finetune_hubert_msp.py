import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
from tqdm import tqdm

# Configuration
TRAIN_CSV = "/home4/quanpn/LoRA/data/MSP_PODCAST/train.csv"   # CSV with columns: path,EmoClass for training
VALID_CSV = "/home4/quanpn/LoRA/data/MSP_PODCAST/test.csv"    # CSV with columns: path,EmoClass for validation
AUDIO_DIR = "/home4/quanpn/interspeech2025/Audios"  # Directory containing audio files
MODEL_NAME = "facebook/hubert-base-ls960"
BATCH_SIZE = 4
NUM_EPOCHS = 10
MAX_AUDIO_SEC = 10   # maximum audio length in seconds
OUTPUT_DIR = "/home4/quanpn/LoRA/save_models/hubert_msp"
BEST_DIR = os.path.join(OUTPUT_DIR, "best")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# Load feature extractor and initialize model
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
SAMPLE_RATE = extractor.sampling_rate
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_SEC  # number of samples
label2id = {'A': 0, 'H': 1, 'N': 2, 'S': 3}
NUM_LABELS = len(label2id)
model = HubertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Helper to load, trim/pad waveform, and extract features
def load_audio(path, extractor):
    waveform, sr = torchaudio.load(path)
    # to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    # resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    # trim or pad
    if waveform.size(0) > MAX_AUDIO_LEN:
        waveform = waveform[:MAX_AUDIO_LEN]
    else:
        pad_len = MAX_AUDIO_LEN - waveform.size(0)
        waveform = F.pad(waveform, (0, pad_len))
    # extract features (no padding needed, uniform length)
    inputs = extractor(waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return inputs['input_values'].squeeze(0)

# Dataset class without attention_mask
class MSPDataset(Dataset):
    def __init__(self, csv_file, audio_dir, extractor, label2id):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.extractor = extractor
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['path'])
        input_values = load_audio(audio_path, self.extractor)
        label = torch.tensor(self.label2id[row['EmoClass']], dtype=torch.long)
        return {'input_values': input_values, 'labels': label}

# Prepare datasets and dataloaders (default collate stacks uniform tensors)
dataset_train = MSPDataset(TRAIN_CSV, AUDIO_DIR, extractor, label2id)
dataset_val = MSPDataset(VALID_CSV, AUDIO_DIR, extractor, label2id)
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Track best validation loss
best_val_loss = float('inf')

# Training & validation loops with tqdm
for epoch in range(1, NUM_EPOCHS + 1):
    # Training
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
    for batch in train_loop:
        optimizer.zero_grad()
        inputs = batch['input_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_values=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item()
        acc = correct / total * 100
        train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")
    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]")
    with torch.no_grad():
        for batch in val_loop:
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_values=inputs, labels=labels)
            loss = outputs.loss

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_loss += loss.item()
            acc = correct / total * 100
            val_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(BEST_DIR)
        extractor.save_pretrained(BEST_DIR)
        print(f"Saved best model at epoch {epoch} with Val Loss: {best_val_loss:.4f}")

# Save final model and extractor
model.save_pretrained(OUTPUT_DIR)
extractor.save_pretrained(OUTPUT_DIR)
print("Training complete. Final model saved to", OUTPUT_DIR)
print("Best model saved to", BEST_DIR)
