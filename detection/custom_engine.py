import math
import sys
import os
import torch
import utils
import wandb
import utils
import json
import random
import numpy as np
from evals import evaluate_model
import pprint as pp

def train_model(model, dataloaders, datasets, epochs, model_config, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print_freq = 10
    best_val_accuracy = 0
    batch_size = model_config['batch_size']

    train_dataset, val_dataset, test_dataset = datasets
    # Data loaders is a tuple of (train_loader, val_loader, test_loader)
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    pretrained_suffix = '_pretrained' if model_config['pretrained'] else ''
    frozen_layers_suffix = '_frozen_layers' if model_config['freeze_layers'] else ''

    model_full_name = f"{model_config['dataset']}_{model_config['name']}_{epochs}{pretrained_suffix}{frozen_layers_suffix}"


    model_saving_path = f"models/{model_full_name}"

    tags = []
    if model_config['pretrained']:
        tags.append('pretrained')
    if model_config['freeze_layers']:
        tags.append('frozen_layers')
        
    # Check if final model already exists
    final_model_path = os.path.join(model_saving_path, "final_model.pth")
    if os.path.exists(final_model_path):
        print()
        print(f"\n##############################################################################################################\n")
        print(f"Model {final_model_path} already exists. Loading and returning the model.")
        print(f"\n##############################################################################################################\n")
        print()
        model.load_state_dict(torch.load(final_model_path))
        model.to(device)
        return model

    # Continue with the rest of the function if the model does not exist


    result_saving_path = f"results/{model_full_name}"
    os.makedirs(model_saving_path , exist_ok=True)
    os.makedirs(result_saving_path , exist_ok=True)
    

    wandb.init(project="portalcut",
            entity='231n-augmentation',
            name=model_full_name,
            notes=model_config['name'],
            tags=tags, 
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": "SGD",
            })
    config = wandb.config
    

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    wandb.config.update({
    "model_architecture": str(model),  # Include model architecture details
    "training_data_size": len(train_dataset),
    "validation_data_size": len(val_dataset),
    "test_data_size": len(test_dataset),
    })


    # Training loop

    # Assume we have an existing setup
    for i in range(epochs):
        epoch = i + 1
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"


        for images, targets in metric_logger.log_every(train_dataloader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


            # Log metrics to WandB
            wandb.log({
                **loss_dict,
                "epoch": epoch,
                "loss": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"]
            })
            metric_logger.update(loss=losses, **loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        """
        Evaluation after each epoch
        """
        # Evaluate on validation set
        mean_iou, avg_precision, avg_recall, avg_f1, mAP, per_class_ap = evaluate_model(model, val_dataset, model_config['dataset_config']['class_list'])
        # print(f"Mean IoU: {mean_iou:.4f}, mAP: {mAP:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
        wandb.log({
            'val_iou': mean_iou,
            'val_precision': avg_precision,
            'val_recall': avg_recall,
            'val_f1': avg_f1,
            'val_mAP': mAP,
            'epoch': epoch,
            **per_class_ap
        })
        

        # Save the model if it has the best validation accuracy so far
        
        if mAP > best_val_accuracy:
            best_val_accuracy = mAP
            torch.save(model.state_dict(), f"{model_saving_path}/best_val_model.pth")

        # Save model after every epoch or as needed

        torch.save(model.state_dict(), f"{model_saving_path}/epoch_{epoch}.pth")

    """
    Evaluation on the test set
    """
    test_iou, test_precision, test_recall, test_f1, test_mAP, test_per_class_ap = evaluate_model(model, test_dataset, model_config['dataset_config']['class_list'])
    wandb.log({
        'test_iou': test_iou,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_map': test_mAP,
        **test_per_class_ap
    })

    # Save final model
    torch.save(model.state_dict(), f"{model_saving_path}/final_model.pth")

    
    results = {
        **model_config,
        'val_iou': float(mean_iou),
        'val_precision': avg_precision,
        'val_recall': avg_recall,
        'val_f1': avg_f1,
        'val_map': mAP,
        'test_iou': float(test_iou),
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_map': test_mAP,
        'Car_ap': per_class_ap['Car_ap'],
        'Van_ap': per_class_ap['Van_ap'],
        'Truck_ap': per_class_ap['Truck_ap'],
        'Pedestrian_ap': per_class_ap['Pedestrian_ap'],
        'Person_sitting_ap': per_class_ap['Person_sitting_ap'],
        'Cyclist_ap': per_class_ap['Cyclist_ap'],
        'Tram_ap': per_class_ap['Tram_ap'],
        'Misc_ap': per_class_ap['Misc_ap'],     
    }
    # # Assuming 'results' is already defined as shown above
    # for key, value in results.items():
    #     print(f"The type of '{key}' is {type(value)}")

    
    with open(f"{result_saving_path}/results.json", 'w') as f:
        json.dump(results, f) 


    wandb.finish()
    
    return model


def reset_environment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    else:
        torch.set_num_threads(1)