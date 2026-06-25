import os
import torch
import argparse
from torch.backends import cudnn
from model import build_net
from train import _train
from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DHFSN: Dynamic Hierarchical Feature Selection Network for Single Image Dehazing')

    # Basic configuration
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str, help='Train or test mode')
    parser.add_argument('--dataset', type=str, default='ITS',
                        choices=['ITS', 'Haze4K', 'DenseHaze', 'NH-HAZE', 'O-HAZE', 'FoggyCityscapes'],
                        help='Dataset type')
    parser.add_argument('--data_dir', type=str, default='../ITS_v2', help='Path to dataset root directory')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--num_epoch', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--print_freq', type=int, default=25, help='Print frequency')
    parser.add_argument('--num_worker', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=100, help='Model save frequency')
    parser.add_argument('--valid_freq', type=int, default=5, help='Validation frequency')
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint')

    # Test parameters
    parser.add_argument('--test_model', type=str, default='results/Best.pkl', help='Path to test model')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False], help='Save test images')

    args = parser.parse_args()

    # Set default configurations based on dataset
    dataset_defaults = {
        'ITS': {'data_dir': '../ITS_v2', 'learning_rate': 1e-4, 'batch_size': 5, 'num_epoch': 100, 'save_freq': 1000},
        'Haze4K': {'data_dir': '../Haze4K', 'learning_rate': 4e-4, 'batch_size': 8, 'num_epoch': 1500, 'save_freq': 100},
        'DenseHaze': {'data_dir': '../DenseHaze', 'learning_rate': 2e-4, 'batch_size': 2, 'num_epoch': 5000, 'save_freq': 100},
        'NH-HAZE': {'data_dir': '../NH-HAZE', 'learning_rate': 2e-4, 'batch_size': 2, 'num_epoch': 5000, 'save_freq': 1000},
        'O-HAZE': {'data_dir': '../O-HAZE', 'learning_rate': 2e-4, 'batch_size': 2, 'num_epoch': 5000, 'save_freq': 100},
        'FoggyCityscapes': {'data_dir': '../FoggyCityscapes', 'learning_rate': 4e-4, 'batch_size': 2, 'num_epoch': 200, 'save_freq': 10},
    }

    # Apply dataset defaults if not overridden
    if args.dataset in dataset_defaults:
        defaults = dataset_defaults[args.dataset]
        for key, value in defaults.items():
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    args.model_save_dir = os.path.join('results/')
    args.result_dir = os.path.join('results/', 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epoch}")

    main(args)
