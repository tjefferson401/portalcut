import math
import sys
import time
import torch
import utils
import wandb
import utils
import json
import random
import numpy as np
from evals import evaluate_model


def train_model(model, dataloaders, epochs, config, batch_size, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print_freq = 10
    batch_size = config['batch_size']

    # Data loaders is a tuple of (train_loader, val_loader, test_loader)
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    wandb.init(project="portalcut",
            entity='231n-augmentation', 
            notes="2024-05-30-kitti-test1-fasterrcnn_resnet50_fpn_v2_scratch_50ep_v2",
            
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": "Adam",
            })
    config = wandb.config
    

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)


    # Training loop

    # Assume we have an existing setup
    for epoch in range(epochs):
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
        avg_iou, avg_precision, avg_recall, avg_f1, avg_map = evaluate_model(model, dataloaders['val'], device)
        wandb.log({
            'val_iou': avg_iou,
            'val_precision': avg_precision,
            'val_recall': avg_recall,
            'val_f1': avg_f1,
            'val_map': avg_map,
            'epoch': epoch
        })

        # Save the model if it has the best validation accuracy so far
        if avg_map > best_val_accuracy:
            best_val_accuracy = avg_map
            torch.save(model.state_dict(), f"{config['name']}_best_val_model.pth")

        # Save model after every epoch or as needed
        torch.save(model.state_dict(), f"{config['name']}_epoch_{epoch}.pth")

    """
    Evaluation on the test set
    """
    test_iou, test_precision, test_recall, test_f1, test_map = evaluate_model(model, dataloaders['test'], device)
    wandb.log({
        'test_iou': test_iou,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_map': test_map
    })

    # Save final model
    torch.save(model.state_dict(), f"{config['name']}_final_model.pth")

    # Save accuracy results locally
    results = {
        'config': config,
        'val_iou': avg_iou,
        'val_precision': avg_precision,
        'val_recall': avg_recall,
        'val_f1': avg_f1,
        'val_map': avg_map,
        'test_iou': test_iou,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_map': test_map
    }
    with open(f"{config['name']}_results.json", 'w') as f:
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