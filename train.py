import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from data_loaders import TusimpleSet
from model import LaneNet
from train_lanenet import train_model
from transformer import Rescale

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    dataset = 'D:\\cs329\\archive\\TUSimple\\train_set\\training'
    height = 256
    width = 512
    epochs = 500
    lr = 0.0001
    batch_size = 4
    save_path = 'log'

    train_dataset_file = os.path.join(dataset, 'train.txt')
    val_dataset_file = os.path.join(dataset, 'val.txt')

    resize_height = height
    resize_width = width

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    train_dataset = TusimpleSet(train_dataset_file, transform=data_transforms['train'],
                                target_transform=target_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'], target_transform=target_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

    model = LaneNet()

    weights = torch.load('log/epoch_100_model.pth')
    model.load_state_dict(weights['state_dict'])

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    optimizer.load_state_dict(weights['optimizer'])

    print(f"{epochs} epochs {len(train_dataset)} training samples\n")

    model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                             device=DEVICE, num_epochs=epochs, save_path=save_path, training_log=weights['log'])
    df = pd.DataFrame({'epoch': [], 'training_loss': [], 'val_loss': []})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch', 'training_loss', 'val_loss'], header=True, index=False,
              encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))

    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))


if __name__ == '__main__':
    train()
