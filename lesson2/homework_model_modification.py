import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================== Задание 1: Модификация существующих моделей ===============================================
# 1.1
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience 
        self.min_delta = min_delta
        self.best_loss = None
        self.epoch_counter = 0
        self.should_stop = False

    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.epoch_counter = 0
        else:
            self.epoch_counter += 1
            if self.epoch_counter >= self.patience:
                self.should_stop = True


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


# 1.2
class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features=2):
        """ out_features=2 для бинарной классификации, >2 для многоклассовой (потому что буду использовать CrossEntropyLoss) """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)


class Metrics:
    @staticmethod
    def to_binary_preds(y_pred, threshold=0.5):
        if y_pred.ndim > 1:
            y_pred = y_pred.squeeze()
        return (y_pred >= threshold).long()

    @staticmethod
    def confusion_matrix(y_true, y_pred, threshold=0.5):
        y_true = y_true.long().view(-1)
        y_pred = Metrics.to_binary_preds(y_pred, threshold)

        cm = torch.zeros((2, 2), dtype=torch.int64)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        return cm

    @staticmethod
    def accuracy(y_true, y_pred, threshold=0.5):
        y_true = y_true.long().view(-1)
        y_pred = Metrics.to_binary_preds(y_pred, threshold)
        return (y_true == y_pred).float().mean().item()

    @staticmethod
    def precision(y_true, y_pred, threshold=0.5):
        cm = Metrics.confusion_matrix(y_true, y_pred, threshold)
        tn, fp = cm[0, 0], cm[0, 1]
        tp, fn = cm[1, 1], cm[1, 0]

        return (tp / (tp + fp + 1e-12)).item()

    @staticmethod
    def recall(y_true, y_pred, threshold=0.5):
        cm = Metrics.confusion_matrix(y_true, y_pred, threshold)
        tn, fp = cm[0, 0], cm[0, 1]
        tp, fn = cm[1, 1], cm[1, 0]

        return (tp / (tp + fn + 1e-12)).item()

    @staticmethod
    def f1_score(y_true, y_pred, threshold=0.5):
        p = Metrics.precision(y_true, y_pred, threshold)
        r = Metrics.recall(y_true, y_pred, threshold)
        return 2 * p * r / (p + r + 1e-12)

    @staticmethod
    def roc_auc(y_true, y_score):
        y_true = y_true.view(-1)
        y_score = y_score.view(-1)

        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]

        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("ROC-AUC requires both positive and negative samples.")

        auc = 0.0

        for p in pos:
            auc += torch.sum(p > neg).item()
            auc += 0.5 * torch.sum(p == neg).item()

        auc /= (len(pos) * len(neg))

        return auc

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, threshold=0.5, normalize=False):
        cm = Metrics.confusion_matrix(y_true, y_pred, threshold).float()

        if normalize:
            cm = cm / (cm.sum(dim=1, keepdim=True) + 1e-12)

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")

        labels = ["0 (Negative)", "1 (Positive)"]
        plt.xticks([0, 1], labels)
        plt.yticks([0, 1], labels)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                plt.text(
                    j, i,
                    f"{val:.2f}" if normalize else int(val),
                    ha="center", va="center"
                )

        plt.colorbar()
        plt.tight_layout()
        plt.show()