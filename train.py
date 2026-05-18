import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from github.datasets.dataset import CropDataset
from losses.TGV import TGV2Loss
from models.ufno import ufno
from configs.config_setting import setting_config


def main(config):
    config.create_save_dir()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load dataset
    full_train_dataset = CropDataset(data_path=config.h5_path, target_size=(config.target_size, config.target_size))
    full_eval_dataset = CropDataset(data_path=config.h5_path, target_size=(config.target_size, config.target_size))

    total_size = len(full_train_dataset)
    train_size = int(config.train_ratio * total_size)
    val_size = int(config.val_ratio * total_size)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size: train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_eval_dataset, val_indices)
    test_dataset = Subset(full_eval_dataset, test_indices)

    print(f"Data split complete. Total: {total_size} Train: {len(train_dataset)} Val: {len(val_dataset)} Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # Initialize model
    model = ufno(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        modes1=config.modes1,
        modes2=config.modes2,
        width=config.width,
        num_fno_layers=config.num_fno_layers,
        num_ufno_layers=config.num_ufno_layers
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=config.beta_smoothl1)
    tgv_loss_fn = TGV2Loss(
        alpha0=config.tgv_alpha0,
        alpha1=config.tgv_alpha1,
        inner_iters=config.tgv_inner_iters,
        lr=config.tgv_lr
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-8)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for img, gt in train_loader:
            img, gt = img.to(DEVICE), gt.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(img)

            data_loss = criterion(outputs, gt)
            regularization_loss = tgv_loss_fn(outputs)

            loss = data_loss + config.lamda_tgv * regularization_loss
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{config.epochs}] | Train Loss: {avg_epoch_loss:.6f}")

    final_save_path = os.path.join(config.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_save_path)


if __name__ == '__main__':
    config = setting_config()
    main(config)