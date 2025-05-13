import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm

# Configuration
TRAIN_CSV = "/home4/quanpn/LoRA/data/MSP_PODCAST/train.csv"
VALID_CSV = "/home4/quanpn/LoRA/data/MSP_PODCAST/test.csv"
AUDIO_DIR = "/home4/quanpn/interspeech2025/Audios"
MODEL_NAME = "facebook/wav2vec2-base-960h"
BATCH_SIZE = 4
NUM_EPOCHS = 10
MAX_AUDIO_SEC = 10
OUTPUT_DIR = "/home4/quanpn/LoRA/save_models/wav2vec_msp"
BEST_DIR = os.path.join(OUTPUT_DIR, "best")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# Load feature extractor & model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
SAMPLE_RATE = feature_extractor.sampling_rate
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_SEC

label2id = {'A': 0, 'H': 1, 'N': 2, 'S': 3}
NUM_LABELS = len(label2id)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    # to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # resample nếu cần
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    # độ dài gốc trước khi pad/truncate
    orig_len = waveform.size(0)

    # trim hoặc pad
    if orig_len > MAX_AUDIO_LEN:
        waveform = waveform[:MAX_AUDIO_LEN]
        orig_len = MAX_AUDIO_LEN
        pad_len = 0
    else:
        pad_len = MAX_AUDIO_LEN - orig_len
        waveform = F.pad(waveform, (0, pad_len))

    # build attention mask: 1 cho phần "thật", 0 cho phần pad
    attention_mask = torch.cat([
        torch.ones(orig_len, dtype=torch.long),
        torch.zeros(pad_len, dtype=torch.long)
    ])

    # feature extraction
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )
    input_values = inputs.input_values.squeeze(0)  # [seq_len]

    return input_values, attention_mask


class MSPDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(AUDIO_DIR, row['path'])
        input_values, attention_mask = load_audio(audio_path)
        label = torch.tensor(label2id[row['EmoClass']], dtype=torch.long)
        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'labels': label
        }

# Dataloaders
train_loader = DataLoader(
    MSPDataset(TRAIN_CSV),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    MSPDataset(VALID_CSV),
    batch_size=BATCH_SIZE
)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = correct = total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
    for batch in loop:
        optimizer.zero_grad()
        inputs = batch['input_values'].to(device)     # [B, L]
        masks  = batch['attention_mask'].to(device)    # [B, L]
        labels = batch['labels'].to(device)
        outputs = model(
            input_values=inputs,
            attention_mask=masks,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total*100:.2f}%")
    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = correct = total = 0
    loop = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]")
    with torch.no_grad():
        for batch in loop:
            inputs = batch['input_values'].to(device)
            masks  = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_values=inputs, attention_mask=masks, labels=labels)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=f"{outputs.loss.item():.4f}", acc=f"{correct/total*100:.2f}%")
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f"Epoch {epoch} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f} — Val Acc: {val_acc:.4f}")

    # Save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(BEST_DIR)
        feature_extractor.save_pretrained(BEST_DIR)
        print(f"→ Saved best model at epoch {epoch} (Val Loss: {best_val_loss:.4f})")

# Cuối cùng
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
print("Training complete.")
print("– Final model:", OUTPUT_DIR)
print("– Best model:", BEST_DIR)
