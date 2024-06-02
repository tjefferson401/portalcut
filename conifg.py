model_configs = [
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': True, 
     'freeze_layers': False,
     'batch_size': 8,
     },
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': False, 
     'freeze_layers': False,
     'batch_size': 8,
     },
    {'name': 'fasterrcnn_resnet50_fpn', 
     'pretrained': True, 
     'freeze_layers': True,
     'batch_size': 8,
     },
    # Add more configurations as needed
]

epochs_list = [5, 10, 50, 100]
