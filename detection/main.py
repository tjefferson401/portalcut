import argparse
# from detection.custom_engine import train_model
from data_utils import get_datamoduels, get_transform
from datasets import KittiTorch
from custom_engine import reset_environment, train_model
from configs import model_configs, epochs_list
from models import get_model



def main():
    for epochs in epochs_list:
        for config in model_configs:
            reset_environment()
            dataset = KittiTorch(root='../data', download=True, transform=get_transform())
            train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = get_datamoduels(dataset, batch_size=config['batch_size'])
            datasets = (train_dataset, val_dataset, test_dataset)
            dataloaders = (train_dataloader, val_dataloader, test_dataloader)
            model = get_model(config)
            model = train_model(model, dataloaders, datasets, epochs, config, learning_rate=0.001)
    
    
     
if __name__ == '__main__':
    main()
    print("Done!")