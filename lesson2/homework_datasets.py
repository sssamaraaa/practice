import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


# =============================================== Задание 2: Работа с датасетами ===============================================
# 2.1
class CustomDataset(Dataset):
    def __init__(self, csv_file, target_column=None, normalize=False, encode_categories=False, task_type="regression", feature_engineering_fn=False):
        self.data = pd.read_csv(csv_file)
        self.target_column = target_column
        self.normalize = normalize
        self.encode_categories = encode_categories
        self.task_type = task_type
        
        self.data = self.data.dropna()

        if feature_engineering_fn:
            self.data = feature_engineering_fn(self.data)
        
        if self.target_column is not None:
            self.y = self.data[self.target_column]
            self.X = self.data.drop(columns=[self.target_column])
        else:
            raise ValueError("Need to specify target_column.")
        
        numerical_columns = self.X.select_dtypes(include=["number"]).columns
        categorical_columns = self.X.select_dtypes(exclude=["number"]).columns

        # практически всегда регрессии используют z-нормализацию (для линейной регрессии без регуляризации - нормализация не нужна)
        if normalize:
            self.X[numerical_columns] = (self.X[numerical_columns] - self.X[numerical_columns].mean()) / self.X[numerical_columns].std()
        
        if encode_categories:
            self.X = pd.get_dummies(self.X, columns=categorical_columns)

        self.X = self.X.astype(np.float32)
        self.X = torch.tensor(self.X.values, dtype=torch.float32)

        if self.y.dtype == object:
            self.y = pd.factorize(self.y)[0]  # преобразуем категориальные метки в числовые
        
        self.y = torch.tensor(self.y.values, dtype=torch.float32 if self.task_type == "regression" else torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]