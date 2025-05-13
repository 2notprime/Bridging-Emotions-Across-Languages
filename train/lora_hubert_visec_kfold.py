import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ===== CONFIGURATION =====
ALL_CSV = "/home4/quanpn/LoRA/data/visec/data.csv"
AUDIO_DIR = "/home4/quanpn/LoRA/data"
PRETRAINED_DIR = "/home4/quanpn/LoRA/save_models/hubert_msp/best"
OUTPUT_DIR_LORA = "/home4/quanpn/LoRA/save_models/lora_kfold/hubert_visec"
os.makedirs(OUTPUT_DIR_LORA, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
MAX_AUDIO_SEC = 10
NUM_FOLDS = 5

# Label mapping
label2id = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}
NUM_LABELS = len(label2id)

# ===== LOAD FEATURE EXTRACTOR =====
extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_DIR)
SAMPLE_RATE = extractor.sampling_rate
MAX_AUDIO_LEN = SAMPLE_RATE * MAX_AUDIO_SEC

# ===== AUDIO LOADER =====
def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    if wav.size(0) > MAX_AUDIO_LEN:
        wav = wav[:MAX_AUDIO_LEN]
    else:
        pad = MAX_AUDIO_LEN - wav.size(0)
        wav = F.pad(wav, (0, pad))
    enc = extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return enc['input_values'].squeeze(0)

# ===== DATASET CLASS =====
class AudioDataset(Dataset):
    def __init__(self, df, audio_dir):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        iv = load_audio(os.path.join(self.audio_dir, row['path']))
        lbl = torch.tensor(label2id[row['emotion']], dtype=torch.long)
        return {'input_values': iv, 'labels': lbl}

# ===== LoRA CONFIG =====
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
)

# ===== DEVICE =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== CROSS-VALIDATION =====
df = pd.read_csv(ALL_CSV)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold = 1
all_val_accs = []

for train_idx, val_idx in skf.split(df, df['emotion']):
    print(f"\n=== Fold {fold}/{NUM_FOLDS} ===")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(AudioDataset(train_df, AUDIO_DIR), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_df, AUDIO_DIR), batch_size=BATCH_SIZE)

    model = HubertForSequenceClassification.from_pretrained(PRETRAINED_DIR, num_labels=NUM_LABELS)
    model = get_peft_model(model, lora_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    best_fold_path = os.path.join(OUTPUT_DIR_LORA, f"fold_{fold}")
    os.makedirs(best_fold_path, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss, train_corr, train_cnt = 0.0, 0, 0
        train_loop = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch} [Train]")
        for batch in train_loop:
            optimizer.zero_grad()
            iv = batch['input_values'].to(device)
            lbl = batch['labels'].to(device)
            outputs = model.base_model(input_values=iv, labels=lbl)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            preds = outputs.logits.argmax(dim=-1)
            train_corr += (preds == lbl).sum().item()
            train_cnt += lbl.size(0)
            train_loss += loss.item()
            acc = train_corr / train_cnt * 100
            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

        # Validation
        model.eval()
        val_loss, val_corr, val_cnt = 0.0, 0, 0
        val_loop = tqdm(val_loader, desc=f"Fold {fold} Epoch {epoch} [Val]")
        with torch.no_grad():
            for batch in val_loop:
                iv = batch['input_values'].to(device)
                lbl = batch['labels'].to(device)
                outputs = model.base_model(input_values=iv, labels=lbl)
                loss = outputs.loss
                preds = outputs.logits.argmax(dim=-1)
                val_corr += (preds == lbl).sum().item()
                val_cnt += lbl.size(0)
                val_loss += loss.item()
                acc_v = val_corr / val_cnt * 100
                val_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc_v:.2f}%")

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_corr / val_cnt
        print(f"Fold {fold} Epoch {epoch} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(best_fold_path)
            print(f"[Fold {fold}] Saved best model at epoch {epoch} with Val Loss: {best_val_loss:.4f}")

    all_val_accs.append(val_acc)
    fold += 1

# ===== SUMMARY =====
print("\n=== Cross-validation summary ===")
for i, acc in enumerate(all_val_accs):
    print(f"Fold {i+1}: Val Acc = {acc*100:.2f}%")
print(f"Average Val Accuracy: {sum(all_val_accs)/NUM_FOLDS*100:.2f}%")
