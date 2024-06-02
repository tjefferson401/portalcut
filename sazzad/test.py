# Define the custom dataset class using torchvision.datasets.Kitti
import torchvision
import torchvision.models.detection as detection
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
import torch



class Kitti(torchvision.datasets.Kitti):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # Convert target format from list of dicts to the correct dict format
        labels = [1 + ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'].index(t['type']) for t in target]
        boxes = [t['bbox'] for t in target]
        
        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), 'labels': torch.as_tensor(labels)}
        return image, target
