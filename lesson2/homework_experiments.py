import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from homework_datasets import CustomDataset
from homework_model_modification import LinearRegression, LogisticRegression, Metrics, EarlyStopping


# =============================================== Задание 3: Эксперименты и анализ ===============================================
# 3.1 - 3.2
def titanic_features(df):
    df = df.copy()

    # категориальные
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = (df["FamilySize"] == 0).astype(int)

    df["Age_Fare"] = df["Age"] * df["Fare"]
    df["Age_Class"] = df["Age"] * df["Pclass"]
    df["Fare_Class"] = df["Fare"] * df["Pclass"]

    # полиномиальные
    df["Age_sq"] = df["Age"] ** 2
    df["Fare_log"] = np.log1p(df["Fare"])

    # статистические
    df["mean_fare_by_sex"] = df.groupby("Sex")["Fare"].transform("mean")
    df["mean_age_by_class"] = df.groupby("Pclass")["Age"].transform("mean")

    return df


device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(1)
torch.manual_seed(1)
l1_lambda = 1e-4
l2_lambda = 1e-4
early_stopping = EarlyStopping(patience=5, min_delta=0.01)
epochs = 10

logistic_dataset = CustomDataset(
    "lesson2/data/logreg_dataset/titanic.csv",
    target_column="Survived",
    encode_categories=True,
    normalize=True,
    task_type="classification",
    feature_engineering_fn=titanic_features
)

train_size = int(0.8 * len(logistic_dataset))
val_size = len(logistic_dataset) - train_size
train_dataset, val_dataset = random_split(logistic_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(len(train_dataset), len(val_dataset))

in_features = train_dataset[0][0].shape[0]
out_features = len(torch.unique(torch.tensor([y for _, y in train_dataset])))

model = LogisticRegression(in_features=in_features, out_features=out_features).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=l2_lambda) # L2-регуляризация через weight_decay

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).long()
        optimizer.zero_grad()
        logits = model(xb)
        ce_loss = criterion(logits, yb)

        # L1 - регуляризация
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))

        loss = ce_loss + l1_lambda * l1_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).long()
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_targets.append(yb.cpu())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    acc = Metrics.accuracy(all_targets, all_probs)
    prec = Metrics.precision(all_targets, all_probs)
    rec = Metrics.recall(all_targets, all_probs)
    f1 = Metrics.f1_score(all_targets, all_probs)
    auc = Metrics.roc_auc(all_targets, all_probs)
    cm = Metrics.confusion_matrix(all_targets, all_probs)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"Prec: {prec:.4f} | "
        f"Recall: {rec:.4f} | "
        f"F1: {f1:.4f} | "
        f"ROC-AUC: {auc:.4f}"
    ) 

print("Confusion matrix:")
print(cm)

Metrics.plot_confusion_matrix(all_targets, all_probs)