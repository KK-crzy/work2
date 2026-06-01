import os
import yaml
import torch
from networks.srn.net import SpeckleReductionNet
from utils.datasets import DenoisingDatasetCCA, DenoisingDatasetSimulator
from utils.training_utils import init_random_seed, train_loop

import argparse

def main(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    init_random_seed(cfg.get('seed', 777))

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg['training']['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['training']['output_dir'], "save_model"), exist_ok=True)

    # Dataset
    dataset_type = cfg['data'].get('type', 'cca').lower()
    if dataset_type == 'cca':
        dataset = DenoisingDatasetCCA(cfg['data']['image_dir'], cfg['data']['interp_method'])
    elif dataset_type == 'simulator':
        dataset = DenoisingDatasetSimulator(cfg['data']['image_dir'], cfg['data']['interp_method'])
    else:
        raise ValueError("Unsupported dataset type.")

    model = SpeckleReductionNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        betas=(cfg['training']['b1'], cfg['training']['b2'])
    )

    train_loop(cfg, dataset, model, optimizer, device, cfg['training']['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    # In vivo dataset
    # parser.add_argument('--config', type=str, default='configs/params_inVivo.yaml', help='Path to config file')
    
    # Simulator dataset
    parser.add_argument('--config', type=str, default='configs/params_Simulator.yaml', help='Path to config file')

    args = parser.parse_args()
    main(args.config)
