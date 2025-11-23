"""
Script for training Pulse3D v2 (Hybrid CNN-Transformer)
"""
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import random
import pandas
from datetime import datetime
import os
import warnings

from experiment_config import config
from dataloader import get_data_loader
from models.pulse_3d_v2 import Pulse3D_v2 

torch.backends.cudnn.benchmark = True

# Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)

def make_weights_for_balanced_classes(labels):
    """Tạo trọng số để cân bằng dữ liệu"""
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))
    weights = [n_samples / float(cnt_dict[label]) for label in labels]
    return weights

def train_one_fold(
    train_csv_path,
    valid_csv_path,
    exp_save_root,
):
    # Seed Everything
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"📂 Training Data: {train_csv_path}")
    logging.info(f"📂 Validation Data: {valid_csv_path}")

    # Load CSV
    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)

    # DataLoader Setup
    weights = make_weights_for_balanced_classes(train_df.label.values)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), len(train_df))

    train_loader = get_data_loader(
        config.DATADIR, train_df, mode=config.MODE, sampler=sampler,
        workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION, translations=config.TRANSLATION,
        size_mm=config.SIZE_MM, size_px=config.SIZE_PX,
    )

    valid_loader = get_data_loader(
        config.DATADIR, valid_df, mode=config.MODE,
        workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
        rotations=None, translations=None,
        size_mm=config.SIZE_MM, size_px=config.SIZE_PX,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- KHỞI TẠO MODEL ---
    logging.info("🚀 Initializing Pulse3D v2 (Hybrid Model)...")
    model = Pulse3D_v2(
        num_classes=1,
        input_channels=1,     # Quan trọng: Đặt là 1 cho ảnh CT
        pool_size=(8, 4, 4),  # Kích thước chuẩn sau khi qua ResNet (với input 64x64x64)
        freeze_bn=False
    ).to(device)

    # Loss & Optimizer
    loss_function = torch.nn.BCEWithLogitsLoss() # Tốt nhất cho binary classification
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_auc = -1
    best_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):
        if counter > patience:
            logging.info(f"🛑 Early stopping at epoch {epoch}")
            break

        # --- TRAINING LOOP ---
        model.train()
        train_loss = 0
        steps = 0
        
        # Tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_data in pbar:
            steps += 1
            inputs, labels = batch_data["image"], batch_data["label"]
            
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(1)
            
            # Nếu input là [Batch, 3, D, H, W] nhưng model cần 1 -> Lấy mean hoặc slice
            if inputs.shape[1] == 3:
                inputs = inputs[:, 0:1, :, :, :] # Lấy kênh đầu tiên (hoặc dùng mean)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.squeeze())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= steps
        logging.info(f"Average Train Loss: {train_loss:.4f}")

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        val_steps = 0
        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for batch_data in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                val_steps += 1
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].float().to(device) # <--- QUAN TRỌNG: Thêm .float()
                
                if inputs.dim() == 4: inputs = inputs.unsqueeze(1)
                if inputs.shape[1] == 3: inputs = inputs[:, 0:1, :, :, :]

                outputs = model(inputs)
                loss = loss_function(outputs.squeeze(), labels.squeeze())
                val_loss += loss.item()

                # Lưu kết quả để tính AUC
                y_pred_list.append(outputs.sigmoid().cpu()) # Sigmoid để ra xác suất
                y_true_list.append(labels.cpu())

        val_loss /= val_steps
        
        # Tính Metrics
        y_pred = torch.cat(y_pred_list).numpy().flatten()
        y_true = torch.cat(y_true_list).numpy().flatten()
        
        try:
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            auc_score = metrics.auc(fpr, tpr)
        except ValueError:
            auc_score = 0.0 # Trường hợp chỉ có 1 class trong batch

        logging.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | AUC: {auc_score:.4f}")
        
        # Update Scheduler
        scheduler.step(auc_score)

        # Save Best Model
        if auc_score > best_auc:
            best_auc = auc_score
            best_epoch = epoch + 1
            counter = 0 # Reset patience
            
            save_path = exp_save_root / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            logging.info(f"🔥 New Best AUC! Model saved to {save_path}")
        else:
            counter += 1

    logging.info(f"🏁 Fold Complete. Best AUC: {best_auc:.4f} at Epoch {best_epoch}")

if __name__ == "__main__":
    # Setup thư mục lưu kết quả
    experiment_name = f"{config.EXPERIMENT_NAME}-Pulse3Dv2-{datetime.today().strftime('%Y%m%d')}"
    base_save_root = config.EXPERIMENT_DIR / experiment_name
    base_save_root.mkdir(parents=True, exist_ok=True)

    # Chạy K-Fold (Hoặc chỉ 1 Fold)
    NUM_FOLDS = 5
    for fold in range(NUM_FOLDS):
        logging.info(f"\n{'='*40}\n STARTING FOLD {fold}\n{'='*40}")
        
        fold_save_root = base_save_root / f"fold_{fold}"
        fold_save_root.mkdir(parents=True, exist_ok=True)
        
        train_csv = config.CSV_DIR / f"train_fold{fold}.csv"
        valid_csv = config.CSV_DIR / f"valid_fold{fold}.csv"

        if train_csv.exists() and valid_csv.exists():
            train_one_fold(train_csv, valid_csv, fold_save_root)
        else:
            logging.warning(f"Skipping Fold {fold} (CSVs not found)")