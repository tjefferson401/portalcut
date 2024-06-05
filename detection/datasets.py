import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor



class KittiTorch(datasets.Kitti):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # Adding +1 to all indices to reserve 0 for background
        labels = [1 + ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'].index(t['type']) for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        boxes = [t['bbox'] for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        
        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)}
        return image, target
    
    def get_filename(self, index):
        # Assuming the dataset stores file paths or names in a structure you can access
        return self.images[index]  # Adjust based on your actual data structure where file names are stored
    

class KittiAugmentedV1(datasets.Kitti):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # Adding +1 to all indices to reserve 0 for background
        labels = [1 + ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'].index(t['type']) for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        boxes = [t['bbox'] for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        
        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)}
        return image, target
    
    def get_filename(self, index):
        # Assuming the dataset stores file paths or names in a structure you can access
        return self.images[index]  # Adjust based on your actual data structure where file names are stored
    

class KittiAugmentedV2B(datasets.Kitti):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # Adding +1 to all indices to reserve 0 for background
        labels = [1 + ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'].index(t['type']) for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        boxes = [t['bbox'] for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        
        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)}
        return image, target
    
    def get_filename(self, index):
        # Assuming the dataset stores file paths or names in a structure you can access
        return self.images[index]  # Adjust based on your actual data structure where file names are stored
    
    
class KittiAugmentedV2M(datasets.Kitti):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # Adding +1 to all indices to reserve 0 for background
        labels = [1 + ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'].index(t['type']) for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        boxes = [t['bbox'] for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        
        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)}
        return image, target
    
    def get_filename(self, index):
        # Assuming the dataset stores file paths or names in a structure you can access
        return self.images[index]  # Adjust based on your actual data structure where file names are stored
    
    