import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import box_iou
from collections import defaultdict

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
        if any(compute_iou(pred_box.unsqueeze(0), true_box.unsqueeze(0)) >= iou_threshold for true_box in true_boxes):
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


def evaluate_model(model, dataset, label_names, iou_threshold=0.5):
    print("Evaluation started...")
    all_true_boxes = []
    all_pred_boxes = []
    all_true_labels = []
    all_pred_labels = []

    model.eval()
    for idx in range(len(dataset)):
        # get the image and the target from the dataset
        image, target = dataset[idx]
        # put the image into the right tensor format 
        input_tensor = image.unsqueeze(0).to('cuda')
        
        # Get the predictions given the input tensor
        with torch.no_grad():
            predictions = model(input_tensor)[0]
        
        # Break into separate tensors
        true_boxes = target['boxes'].to('cuda')
        true_labels = target['labels'].to('cuda')
        pred_boxes = predictions['boxes'].to('cuda')
        pred_labels = predictions['labels'].to('cuda')
        
        # Create masks for excluding 'Background' and 'DontCare' labels
        exclude_labels = torch.tensor([label_names.index('Background'), label_names.index('DontCare')], device='cuda')

        # get the compliment values of the labels
        relevant_true_mask = ~(true_labels.unsqueeze(1) == exclude_labels).any(1)
        relevant_pred_mask = ~(pred_labels.unsqueeze(1) == exclude_labels).any(1)


        # filter the boxes and labels based on the masks
        filtered_true_boxes = true_boxes[relevant_true_mask]
        filtered_true_labels = true_labels[relevant_true_mask]
        filtered_pred_boxes = pred_boxes[relevant_pred_mask]
        filtered_pred_labels = pred_labels[relevant_pred_mask]

        # At this point, all Background and DontCare labeled rows have been removed

        all_true_boxes.append(filtered_true_boxes)
        all_pred_boxes.append(filtered_pred_boxes)
        all_true_labels.append(filtered_true_labels)
        all_pred_labels.append(filtered_pred_labels)


    # Calculate IoU
    iou_scores = []
    for true_boxes, pred_boxes in zip(all_true_boxes, all_pred_boxes):
        iou_scores.append(compute_iou(true_boxes, pred_boxes))

    mean_iou = torch.mean(torch.stack([torch.mean(iou) for iou in iou_scores if iou.numel() > 0]))
    print(f"Mean IoU: {mean_iou:.4f}")

    # Calculate Precision, Recall, F1-Score, and mAP
    aps = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_samples = 0

    per_class_ap = {}
    for i, label_name in enumerate(label_names):
        if label_name == "Background" or label_name == "DontCare":
            continue
        
        true_positives = []
        false_positives = []
        false_negatives = []
        num_gt = 0

        for true_boxes, true_labels, pred_boxes, pred_labels in zip(all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels):
            gt_boxes = true_boxes[true_labels == i]
            pred_boxes = pred_boxes[pred_labels == i]
            num_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                false_negatives.append(len(gt_boxes))
                continue
            
            ious = compute_iou(gt_boxes, pred_boxes)
            if ious.numel() == 0:
                false_negatives.append(len(gt_boxes))
                continue

            max_ious, _ = ious.max(dim=1)
            detected = max_ious > iou_threshold
            true_positives.extend(detected.cpu().numpy())
            false_positives.extend(~detected.cpu().numpy())
            false_negatives.append(len(gt_boxes) - detected.sum().item())
        
        tp_cumsum = np.cumsum(true_positives) if true_positives else np.array([0])
        fp_cumsum = np.cumsum(false_positives) if false_positives else np.array([0])
        fn_cumsum = np.cumsum(false_negatives) if false_negatives else np.array([0])
        
        if len(tp_cumsum) > 0 and len(fp_cumsum) > 0:
            precision = tp_cumsum[-1] / (tp_cumsum[-1] + fp_cumsum[-1] + 1e-6)
            recall = tp_cumsum[-1] / (fn_cumsum[-1] + tp_cumsum[-1] + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            ap = np.trapz(np.clip(tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6), 0, 1), np.clip(tp_cumsum / (num_gt + 1e-6), 0, 1))
        else:
            precision = recall = f1 = ap = 0

        aps.append(ap)
        print(f"AP for {label_name}: {ap:.4f}")

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_samples += 1

        per_class_ap[f'{label_name}_ap'] = ap

    mAP = np.mean(aps)
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_f1 = total_f1 / total_samples

    # Ensure recall does not exceed 1
    avg_recall = min(avg_recall, 1.0)

    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    return mean_iou, avg_precision, avg_recall, avg_f1, mAP, per_class_ap
