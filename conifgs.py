epochs_list = [5, 10, 20]


kitti_class_list = ['Background', 
                    'Car', 
                    'Van', 
                    'Truck', 
                    'Pedestrian', 
                    'Person_sitting', 
                    'Cyclist',
                    'Tram',
                    'Misc',
                    'DontCare']



dataset_configs = {'kitti': {'class_list': kitti_class_list,
                             'num_classes': len(kitti_class_list),
                             }
                    }


model_configs = [
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 8,
     'dataset': 'kitti',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 8,
     'dataset': 'kitti',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': True, 
     'freeze_layers': True,
     'batch_size': 8,
     'dataset': 'kitti',
     'dataset_config': dataset_configs['kitti'],
     },
    
    {'name': 'fasterrcnn_resnet50_fpn_v2', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn_v2', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 4,
     'dataset': 'kitti',
     'dataset_config': dataset_configs['kitti'],
     },
    {'name': 'fasterrcnn_resnet50_fpn_v2', 
     'pretrained': True, 
     'freeze_layers': True,
     'batch_size': 4,
     'dataset': 'kitti',
     'dataset_config': dataset_configs['kitti'],
     },
    # Add more configurations as needed
]
