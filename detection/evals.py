import torch
from collections import defaultdict
import numpy as np
from torchvision.ops import box_iou
from collections import defaultdict

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

def compute_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.tensor([]).to(boxes1.device)
    boxes1 = boxes1.to('cuda')
    boxes2 = boxes2.to('cuda')
    ious = box_iou(boxes1, boxes2)
    return ious



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


def new_evaluate_model(model, dataloader, device):
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



# Alversion already working
def evaluate_model(model, dataset, label_names, iou_threshold=0.5):
    all_true_boxes = []
    all_pred_boxes = []
    all_true_labels = []
    all_pred_labels = []

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        input_tensor = image.unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            predictions = model(input_tensor)[0]
        
        true_boxes = target['boxes'].to('cuda')
        true_labels = target['labels'].to('cuda')
        pred_boxes = predictions['boxes'].to('cuda')
        pred_labels = predictions['labels'].to('cuda')

        all_true_boxes.append(true_boxes)
        all_pred_boxes.append(pred_boxes)
        all_true_labels.append(true_labels)
        all_pred_labels.append(pred_labels)

    iou_scores = []
    for true_boxes, pred_boxes in zip(all_true_boxes, all_pred_boxes):
        iou_scores.append(compute_iou(true_boxes, pred_boxes))

    mean_iou = torch.mean(torch.stack([torch.mean(iou) for iou in iou_scores if iou.numel() > 0]))
    print(f"Mean IoU: {mean_iou:.4f}")

    # Compute mAP (mean Average Precision)
    aps = []
    for i, label_name in enumerate(label_names):
        if label_name == "Background":
            continue
        
        true_positives = []
        false_positives = []
        num_gt = 0

        for true_boxes, true_labels, pred_boxes, pred_labels in zip(all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels):
            gt_boxes = true_boxes[true_labels == i]
            pred_boxes = pred_boxes[pred_labels == i]
            num_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue
            
            ious = compute_iou(gt_boxes, pred_boxes)
            if ious.numel() == 0:
                continue
            true_positive = ious.max(dim=0)[0] > iou_threshold
            false_positive = ~true_positive

            true_positives.extend(true_positive.cpu().numpy())
            false_positives.extend(false_positive.cpu().numpy())
        
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (num_gt + 1e-6)

        ap = np.trapz(precisions, recalls)
        aps.append(ap)

        print(f"AP for {label_name}: {ap:.4f}")

    mAP = np.mean(aps)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")


