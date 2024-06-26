import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def get_model(config):
    
    model_name = config['name']
    pretrained = config['pretrained']
    freeze_layers = config['freeze_layers']
    num_classes = config['dataset_config']['num_classes']
    
    if model_name == 'fasterrcnn_resnet50_fpn':
        model = get_fasterrcnn_resnet50_fpn(num_classes, pretrained, freeze_layers)
    elif model_name == 'fasterrcnn_resnet50_fpn_v2':
        model = get_fasterrcnn_resnet50_fpn_v2(num_classes, pretrained, freeze_layers)
    elif model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
        model = get_fasterrcnn_mobilenet_v3_large_fpn(num_classes, pretrained, freeze_layers)
    elif model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
        model = get_fasterrcnn_mobilenet_v3_large_320_fpn(num_classes, pretrained, freeze_layers)
        
    # elif model_name == 'mobilenet':
    #     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # else:
    #     raise ValueError("Invalid model name")
    return model
  
  
  
def get_fasterrcnn_resnet50_fpn(num_classes, pretrained=True, freeze_layers=False):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_layers:
      for param in model.backbone.parameters():
            param.requires_grad = False
            
    return model
  
  

def get_fasterrcnn_resnet50_fpn_v2(num_classes, pretrained=True, freeze_layers=False):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if freeze_layers:
        for param in model.backbone.parameters():
                param.requires_grad = False
            
    return model

def get_fasterrcnn_mobilenet_v3_large_fpn(num_classes, pretrained=True, freeze_layers=False):
    
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if freeze_layers:
        for param in model.backbone.parameters():
                param.requires_grad = False
            
    return model


def get_fasterrcnn_mobilenet_v3_large_320_fpn(num_classes, pretrained=True, freeze_layers=False):
    
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if freeze_layers:
        for param in model.backbone.parameters():
                param.requires_grad = False
            
    return model