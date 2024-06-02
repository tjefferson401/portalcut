import torch
import json
import wandb
import argparse
import detection.models as models
import detection.datasets as datasets
import detection.evals as evaluate_model

def train_model(model, dataloaders, epochs, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # Adjust this based on your task

    best_val_accuracy = 0
    for epoch in range(epochs):
        model.train()
        for images, targets in dataloaders['train']:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            wandb.log({**loss_dict, 'epoch': epoch, 'config': config})

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

    # Evaluate on test set
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

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, required=True)
    args = parser.parse_args()

    

    config = models.model_configs[args.config]
    model = models.get_model(config)
    dataloaders = datasets.get_dataloaders(config)
    model = train_model(model, dataloaders, epochs_list[0], config)
    print(f"Training of {config['name']} completed.")
    # Path: detection/models.py