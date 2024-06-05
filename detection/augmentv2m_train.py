# from detection.custom_engine import train_model
from data_utils import  get_transform, get_augmented_reduced_datamodules
from datasets import KittiAugmentedV2M, KittiTorch
from custom_engine import reset_environment, train_model


# reset_environment()

from configs import dataset_configs
from models import get_model

epochs_list = [5, 10]

model_configs = [
    {'name': 'fasterrcnn_mobilenet_v3_large_320_fpn', 
     'pretrained': True,
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },
    
    {'name': 'fasterrcnn_mobilenet_v3_large_320_fpn', 
     'pretrained': False,
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },

     {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },
    
    {'name': 'fasterrcnn_resnet50_fpn_v2', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn_v2', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },
    
    {'name': 'fasterrcnn_mobilenet_v3_large_fpn', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_mobilenet_v3_large_fpn', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti_reduced_50',
     'dataset_config': dataset_configs['kitti'],
     },
]


def main():
    for epochs in epochs_list:
        for config in model_configs:
            reset_environment()
            original_dataset = KittiTorch(root='../data', download=True, transform=get_transform())
            augmented_dataset = KittiAugmentedV2M(root='../data', download=True, transform=get_transform())
            for dataset in [original_dataset, augmented_dataset]:
                if dataset == augmented_dataset:
                    config['dataset'] = 'kitti_reduced_50_augmented_v2m'
                train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = \
                    get_augmented_reduced_datamodules(dataset, batch_size=config['batch_size'], original_data_percentage=0.5)
                    
                datasets = (train_dataset, val_dataset, test_dataset)
                dataloaders = (train_dataloader, val_dataloader, test_dataloader)
                model = get_model(config)
                model = train_model(model, dataloaders, datasets, epochs, config, learning_rate=0.001)
    
    
     
if __name__ == '__main__':
    main()
    print("Done!")