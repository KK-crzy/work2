import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

def init_random_seed(seed=777):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )


def get_loss_fn(loss_type):
    if loss_type == 'L1':
        return torch.nn.L1Loss()
    elif loss_type == 'L2':
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def train_loop(cfg, dataset, model, optimizer, device, output_dir):
    dataloader = build_dataloader(dataset, cfg['training']['batch_size'])
    writer = SummaryWriter(log_dir=output_dir)

    # Loss functions
    loss_rec = get_loss_fn(cfg['training']['loss_rec']).to(device)
    loss_consist = get_loss_fn(cfg['training']['loss_consist']).to(device)

    batch_num = 0

    for epoch in range(cfg['training']['epoch']):
        print(f"[Epoch {epoch}]")

        for i_batch, sample in enumerate(dataloader):

            img_low = sample['image_low'].to(device)
            img_high = sample['image_high'].to(device)
            img_mid = sample['image_mid'].to(device)
            label = sample.get('image_clean', None) # simulator dataset has clean labels
            if label is not None:
                label = label.to(device)

            optimizer.zero_grad()
            pred_hr, pred_lr, pred_mid = model(img_high, img_low, img_mid)

            # Reconstruction Loss
            rec_loss = (
                cfg['training']['High_res_weight_rec'] * loss_rec(pred_hr, img_high) +
                cfg['training']['Low_res_weight_rec'] * loss_rec(pred_lr, img_low) +
                cfg['training']['Mid_res_weight_rec'] * loss_rec(pred_mid, img_mid)
            )

            # Consistency Loss
            consist_loss = (
                cfg['training']['Hl_res_weight_consist'] * loss_consist(pred_hr, pred_lr) +
                cfg['training']['HM_res_weight_consist'] * loss_consist(pred_hr, pred_mid) +
                cfg['training']['ML_res_weight_consist'] * loss_consist(pred_lr, pred_mid)
            )

            total_loss = rec_loss + consist_loss
            total_loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/total', total_loss.item(), batch_num)
            batch_num += 1

            if batch_num % cfg['training']['print_every'] == 0:
                print(f"Batch {batch_num}, Loss: {total_loss.item():.4f}, Rec: {rec_loss.item():.4f}, Consist: {consist_loss.item():.4f}")

            if batch_num % cfg['training']['visualize_every'] == 0:
                writer.add_image('img_lowRes', img_low[0], epoch)
                writer.add_image('img_highRes', img_high[0], epoch)
                writer.add_image('img_midRes', img_mid[0], epoch)
                writer.add_image('pred_hr', pred_hr[0], epoch)
                writer.add_image('pred_lr', pred_lr[0], epoch)
                writer.add_image('pred_mid', pred_mid[0], epoch)
                if label is not None:
                    writer.add_image('label', label[0], epoch)

            if batch_num % cfg['training']['checkpoint_every'] == 0 and epoch >= cfg['training']['save_after_epoch']:
                save_path = os.path.join(output_dir, "save_model", f"model_{epoch}.pth")
                torch.save(model.state_dict(), save_path)
                print(f" Model saved at: {save_path}")

