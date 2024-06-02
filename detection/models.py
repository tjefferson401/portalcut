import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(model_name):
    if model_name == 'resnet18':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'mobilenet':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    else:
        raise ValueError("Invalid model name")
    return model
  

def get_fasterrcnn_resnet50_fpn_v2(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#   for param in model.backbone.parameters():
#         param.requires_grad = False
        
  return model