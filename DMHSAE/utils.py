import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def evaluate_metrics(model, dataloader, device, save_dir, fold):
    model.eval()
    all_preds = [[] for _ in range(4)]
    all_trues = [[] for _ in range(4)]

    with torch.no_grad():
        for mri, fdg, av45, gene in dataloader:
            inputs = [t.to(device) for t in (mri, fdg, av45, gene)]
            recons = model(*inputs)

            for i in range(4):
                all_preds[i].append(recons[i].cpu().numpy())
                all_trues[i].append(inputs[i].cpu().numpy())

    metrics = {}
    modalities = ['MRI', 'FDG', 'AV45', 'Gene']

    for i, name in enumerate(modalities):
        y_pred = np.concatenate(all_preds[i])
        y_true = np.concatenate(all_trues[i])

        # 确保数据范围正确
        y_pred = np.clip(y_pred, 0, 1)
        y_true = np.clip(y_true, 0, 1)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        pearson_list = []
        for j in range(y_true.shape[1]):
            try:
                r, _ = pearsonr(y_true[:, j], y_pred[:, j])
                if not np.isnan(r):
                    pearson_list.append(r)
            except:
                continue

        pearson = np.mean(pearson_list) if pearson_list else 0.0

        metrics[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'Pearson': pearson
        }

    df = pd.DataFrame(metrics).T
    df.to_csv(os.path.join(save_dir, f'eval_metrics_fold{fold}.csv'))
    return df


class AlignedMultiModalDataset(Dataset):
    def __init__(self, mri, fdg, av45, gene, indices):
        self.mri = mri[indices]
        self.fdg = fdg[indices]
        self.av45 = av45[indices]
        self.gene = gene[indices]

    def __len__(self):
        return len(self.mri)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.mri[idx], dtype=torch.float32),
            torch.tensor(self.fdg[idx], dtype=torch.float32),
            torch.tensor(self.av45[idx], dtype=torch.float32),
            torch.tensor(self.gene[idx], dtype=torch.float32)
        )


class MultiModalDataset:
    def __init__(self, data_paths, batch_size=32, random_seed=100):
        self.batch_size = batch_size
        self.raw_data = {
            'mri': pd.read_csv(data_paths['mri'], header=None).values.astype(np.float32),
            'fdg': pd.read_csv(data_paths['fdg'], header=None).values.astype(np.float32),
            'av45': pd.read_csv(data_paths['av45'], header=None).values.astype(np.float32),
            'gene': pd.read_csv(data_paths['gene'], header=None).values.astype(np.float32)
        }
        self.indices = np.arange(len(self.raw_data['mri']))
        self.kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    def __call__(self, fold_idx):
        fold_indices = list(self.kf.split(self.indices))[fold_idx]
        train_idx, test_idx = fold_indices

        # 随机划分验证集
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.2,
            random_state=42
        )

        # 保存当前折叠的全部索引
        self.current_indices = np.concatenate([train_idx, val_idx, test_idx])

        # 归一化处理函数
        def scale_data(modality, train_indices):
            train_data = self.raw_data[modality][train_indices]
            data_min = train_data.min(axis=0)
            data_max = train_data.max(axis=0)

            # 处理当前折叠全部数据
            scaled = (self.raw_data[modality][self.current_indices] - data_min) / (data_max - data_min + 1e-8)
            return np.clip(scaled, 0.0, 1.0)  # 添加裁剪确保范围

        # 应用归一化
        self.scaled_data = {
            'mri': scale_data('mri', train_idx),
            'fdg': scale_data('fdg', train_idx),
            'av45': scale_data('av45', train_idx),
            'gene': scale_data('gene', train_idx)
        }

        # 构建数据集加载器
        def create_loader(indices):
            positions = np.where(np.isin(self.current_indices, indices))[0]
            dataset = AlignedMultiModalDataset(
                self.scaled_data['mri'],
                self.scaled_data['fdg'],
                self.scaled_data['av45'],
                self.scaled_data['gene'],
                positions
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return (
            create_loader(train_idx),
            create_loader(val_idx),
            create_loader(test_idx),
            test_idx
        )


class EarlyStopping:
    def __init__(self, patience=15, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def merge_all_folds(save_dir='./reconstructed', output_prefix='full'):
    modalities = ['MRI', 'FDG', 'AV45', 'Gene']
    dfs_all = {mod: [] for mod in modalities}

    for fold in range(5):
        for mod in modalities:
            path = os.path.join(save_dir, f'fold{fold}_{mod}_recon.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs_all[mod].append(df)

    for mod in modalities:
        if dfs_all[mod]:
            df = pd.concat(dfs_all[mod], axis=0)
            df_sorted = df.sort_values(by='index').drop(columns=['index'])
            df_sorted.to_csv(f'{save_dir}/{output_prefix}_{mod}_recon.csv', index=False, header=False)

# Call
if __name__ == '__main__':
    merge_all_folds()