import torch
from collections import defaultdict
import numpy as np

def calculate_iou(box1, box2):
    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union != 0 else 0

def calculate_precision_recall_f1(pred_boxes, true_boxes, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    for pred_box in pred_boxes:
        if any(calculate_iou(pred_box, true_box) >= iou_threshold for true_box in true_boxes):
            tp += 1
        else:
            fp += 1

    fn = len(true_boxes) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def calculate_map(pred_boxes, true_boxes, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    aps = []
    for iou_threshold in iou_thresholds:
        precision, recall, _ = calculate_precision_recall_f1(pred_boxes, true_boxes, iou_threshold)
        aps.append(precision * recall)

    return np.mean(aps)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_map = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                true_boxes = targets[i]['boxes'].cpu().numpy()

                iou = np.mean([calculate_iou(pred_box, true_box) for pred_box, true_box in zip(pred_boxes, true_boxes)])
                precision, recall, f1 = calculate_precision_recall_f1(pred_boxes, true_boxes)
                map_score = calculate_map(pred_boxes, true_boxes)

                total_iou += iou
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_map += map_score
                total_samples += 1

    avg_iou = total_iou / total_samples
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_f1 = total_f1 / total_samples
    avg_map = total_map / total_samples

    return avg_iou, avg_precision, avg_recall, avg_f1, avg_map
