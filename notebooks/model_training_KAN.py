# === STEP 1: Import libraries ===
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
# !pip install git+https://github.com/KindXiaoming/pykan.git
# !pip install pykan
from kan import KAN
import matplotlib.pyplot as plt

# === STEP 2: Load and prepare dataset ===
df = pd.read_csv("/content/sample_data/genai_vs_traditional_classification_5k.csv")
df = df[df['job_type'].isin(['modern', 'traditional'])]

# Combine skill columns
def combine_skills(row):
    skills = set()
    if pd.notnull(row['matched_genai_skills']):
        skills.update([s.strip() for s in row['matched_genai_skills'].split(",") if s.strip()])
    if pd.notnull(row['existing_skill_labels']):
        skills.update([s.strip() for s in row['existing_skill_labels'].split(",") if s.strip()])
    return list(skills)

top_N = 100
all_skills_flat = df['matched_genai_skills'].fillna('').str.split(',').explode().str.strip().tolist() + \
                   df['existing_skill_labels'].fillna('').str.split(',').explode().str.strip().tolist()
most_common_skills = set([s for s, _ in Counter(all_skills_flat).most_common(top_N)])
df['filtered_skills'] = df.apply(lambda row: [s for s in combine_skills(row) if s in most_common_skills], axis=1)

# === STEP 3: Multi-hot encode filtered skills ===
mlb = MultiLabelBinarizer(sparse_output=False)
skill_matrix = mlb.fit_transform(df['filtered_skills'])

# === STEP 4: Encode labels (modern = 0, traditional = 1) ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['job_type'])

# === STEP 5: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    skill_matrix, y, test_size=0.2, stratify=y, random_state=42
)

print(f"✅ Data ready: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

# === STEP 6: Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# === STEP 7: Create sampler for class balance ===
class_counts = np.bincount(y_train)
total = sum(class_counts)
class_weights = [total / c for c in class_counts]
sample_weights = [class_weights[label] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# === STEP 8: KAN model setup ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KAN(width=[X_train.shape[1], 16, 1], grid=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

# === STEP 9: Training loop with accuracy tracking ===
epochs = 70
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Training accuracy
        predicted = (preds > 0.5).float()
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

    train_acc = correct / total
    train_accuracies.append(train_acc)

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test_tensor.to(device))
        val_pred_labels = (val_preds > 0.5).float().cpu()
        val_acc = (val_pred_labels == y_test_tensor).float().mean().item()
        val_accuracies.append(val_acc)

    print(f"📘 Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# === STEP 10: Plot accuracy curves ===
plt.figure(figsize=(8, 5))
plt.plot(train_accuracies, label='Training Accuracy', color='royalblue')
plt.plot(val_accuracies, label='Validation Accuracy', color='darkorange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("KAN - Accuracy over Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()