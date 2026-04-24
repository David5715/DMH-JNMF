import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import merge_all_folds
from Autoencoder import Autoencoder_Reconstructor
from utils import MultiModalDataset, EarlyStopping, evaluate_metrics


# 数据加载
def train_reconstructor(data_paths, save_dir='./reconstructed'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'epochs': 500,
        'init_lr': 0.01,
        'batch_size': 32,
        'weight_decay': 1e-3,
        'patience': 100
    }

    os.makedirs(save_dir, exist_ok=True)
    dataset = MultiModalDataset(data_paths, batch_size=config['batch_size'])

    for fold in range(5):
        print(f"\n=== Fold {fold + 1}/5 ===")
        train_loader, val_loader, test_loader, test_idx = dataset(fold)

        # 获取输入维度
        sample_batch = next(iter(train_loader))
        input_dims = [x.shape[1] for x in sample_batch]

        model = Autoencoder_Reconstructor(input_dims).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=config['patience'])

        best_loss = float('inf')
        for epoch in range(config['epochs']):
            model.train()
            epoch_loss = 0

            for mri, fdg, av45, gene in train_loader:
                mri, fdg, av45, gene = mri.to(device), fdg.to(device), av45.to(device), gene.to(device)
                optimizer.zero_grad()
                recon_mri, recon_fdg, recon_av45, recon_gene = model(mri, fdg, av45, gene)

                # --- 改进 C: 使用可学习的损失权重 ---
                inputs = [mri, fdg, av45, gene]
                recons = [recon_mri, recon_fdg, recon_av45, recon_gene]

                losses = [criterion(recons[i], inputs[i]) for i in range(4)]
                log_vars = model.log_vars

                loss = 0
                for i in range(len(losses)):
                    precision = torch.exp(-log_vars[i])
                    loss += precision * losses[i] + log_vars[i]
                # --- 损失计算修改结束 ---

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # 验证
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_fold{fold}.pth'))

            print(f"Epoch {epoch + 1}/{config['epochs']} | Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 测试集保存重建结果
        # 修复 FutureWarning
        save_results(model, test_loader, save_dir, fold, device, weights_only=True)
        evaluate_metrics(model, test_loader, device, save_dir, fold)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for mri, fdg, av45, gene in dataloader:
            mri, fdg, av45, gene = mri.to(device), fdg.to(device), av45.to(device), gene.to(device)
            recons = model(mri, fdg, av45, gene)

            # --- 改进 C: 使用可学习的损失权重 (验证集) ---
            inputs = [mri, fdg, av45, gene]
            losses = [criterion(recons[i], inputs[i]) for i in range(4)]
            log_vars = model.log_vars

            val_loss = 0
            for i in range(len(losses)):
                precision = torch.exp(-log_vars[i])
                val_loss += precision * losses[i] + log_vars[i]
            # --- 损失计算修改结束 ---

            total_loss += val_loss.item()
    return total_loss / len(dataloader)


# 添加 weights_only 参数以修复 FutureWarning
def save_results(model, dataloader, save_dir, fold, device, weights_only=False):
    model_path = os.path.join(save_dir, f'best_fold{fold}.pth')
    # 应用修复
    model.load_state_dict(torch.load(model_path, weights_only=weights_only))
    model.eval()

    recons = [[] for _ in range(4)]
    indices = []

    with torch.no_grad():
        for batch_idx, (mri, fdg, av45, gene) in enumerate(dataloader):
            # 修复 test_idx 未定义的 bug，使用 dataloader 的 batch size
            batch_size = mri.shape[0]
            start_idx = batch_idx * dataloader.batch_size
            if dataloader.batch_size is None:  # handle batch_size=None
                start_idx = batch_idx * batch_size

            indices.extend(range(start_idx, start_idx + batch_size))

            mri, fdg, av45, gene = mri.to(device), fdg.to(device), av45.to(device), gene.to(device)
            batch_recons = model(mri, fdg, av45, gene)
            for i, rec in enumerate(batch_recons):
                recons[i].append(rec.cpu().numpy())

    modalities = ['MRI', 'FDG', 'AV45', 'Gene']
    for name, data in zip(modalities, recons):
        data = np.concatenate(data)
        df = pd.DataFrame(data)
        df['index'] = indices
        df.to_csv(f'{save_dir}/fold{fold}_{name}_recon.csv', index=False)


if __name__ == '__main__':
    data_paths = {
        'mri': r"C:\Users\David\Desktop\期刊论文1\代码\DSAE\MRi.csv",
        'fdg': r"C:\Users\David\Desktop\期刊论文1\代码\DSAE\FDG.csv",
        'av45': r"C:\Users\David\Desktop\期刊论文1\代码\DSAE\AV45.csv",
        'gene': r"C:\Users\David\Desktop\期刊论文1\代码\DSAE\Gene.csv"
    }
    train_reconstructor(data_paths)
    merge_all_folds(save_dir='./reconstructed')