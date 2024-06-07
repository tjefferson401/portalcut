# from detection.custom_engine import train_model
from data_utils import  get_transform, get_augmented_reduced_datamodules
from datasets import KittiAugmentedV3, KittiTorch
from custom_engine import reset_environment, train_model


# reset_environment()

from configs import dataset_configs
from models import get_model

percentages_list = [0.5, 1.0]

model_configs = [
    {'name': 'fasterrcnn_mobilenet_v3_large_320_fpn', 
     'pretrained': True,
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },
    
    {'name': 'fasterrcnn_mobilenet_v3_large_320_fpn', 
     'pretrained': False,
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },

     {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },
    
    {'name': 'fasterrcnn_resnet50_fpn_v2', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn_v2', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },
    
    {'name': 'fasterrcnn_mobilenet_v3_large_fpn', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_mobilenet_v3_large_fpn', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kittiv3_reduced',
     'dataset_config': dataset_configs['kitti'],
     },
]


def main():
    for epochs in [5, 10, 20, 40, 80, 100]:
        for percentage in percentages_list:
            for config in model_configs:

                # Renaming the dataset
                curr_name = f'kittiv3_reduced_{int(100 * (1 - percentage))}'
                config['dataset'] = curr_name

                #Restting the env
                reset_environment()

                # Getting the datasets
                original_dataset = KittiTorch(root='../data', download=True, transform=get_transform())
                augmented_dataset = KittiAugmentedV3(root='../data', download=True, transform=get_transform())

                # training loop for both datasets
                for dataset in [original_dataset, augmented_dataset]:

                    # Renamig the augmented dtaset
                    if dataset == augmented_dataset:
                        config['dataset'] = f'{curr_name}_augmented_v3'

                    # Getting the dataloaders
                    train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = \
                        get_augmented_reduced_datamodules(dataset, batch_size=config['batch_size'], original_data_percentage=percentage)
                    
                    datasets = (train_dataset, val_dataset, test_dataset)
                    dataloaders = (train_dataloader, val_dataloader, test_dataloader)

                    # Training the model
                    model = get_model(config)
                    model = train_model(model, dataloaders, datasets, epochs, config, learning_rate=0.001)
    
    
     
if __name__ == '__main__':
    main()
    print("Done!")

