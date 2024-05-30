import os
import pandas as pd
from torchvision.io import read_image
import waymo_utils as v2
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from fastparquet import ParquetFile
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from general_utils import timeit
import torch


class WaymoDataset(Dataset):
    def __init__(self, image_dir, box_dir, transform=None):
        with timeit("Dataset initialization"):
            self.image_dir = image_dir
            self.box_dir = box_dir
            self.transform = transform
            self.image_files = os.listdir(image_dir)
            self.box_files = os.listdir(box_dir)
            merged_dfs = []
            box_dfs = []
            # Iterate over the parquet files in the "cam" folder
            for filename in self.image_files:
                # Read the cam and box dataframes
                df_cam = pd.read_parquet(os.path.join(image_dir, filename))
                df_box = pd.read_parquet(os.path.join(box_dir, filename))
                df_cam['filename'] = filename
                df_box['filename'] = filename
                
                # Merge the cam and box dataframes using the v2.merge function
                
                # Append the merged dataframe to the list
                merged_dfs.append(df_cam)
                box_dfs.append(df_box)


            main_df = pd.concat(merged_dfs)
            box_df = pd.concat(box_dfs)
            merged_df = v2.merge(main_df, box_df, right_group=True)


            self.index_to_info = {}
            prev_filename = None
            curr_idx = 0
            total_idx = 0
            for idx, row in merged_df.iterrows():
                if row['filename_x'] != prev_filename:
                    curr_idx = 0
                    prev_filename = row['filename_x']
                self.index_to_info[total_idx] = {
                    'filename': row['filename_x'],
                    'inner_idx': curr_idx,
                }
                curr_idx += 1
                total_idx += 1

            self.length = len(merged_df)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with timeit("Dataset getitem"):
            row_info = self.index_to_info[idx]
            filename = row_info['filename']
            parquet_idx = row_info['inner_idx']
            images_parquet_path = os.path.join(self.image_dir, filename)
            boxes_parquet_path = os.path.join(self.box_dir, filename)
            images_df = pd.read_parquet(images_parquet_path)
            boxes_df = pd.read_parquet(boxes_parquet_path)
            merged_df = v2.merge(images_df, boxes_df, right_group=True)
            image_row = merged_df.iloc[parquet_idx]
            encoded_data = image_row['[CameraImageComponent].image']
            image = Image.open(io.BytesIO(encoded_data))
            bboxes = v2.convert_bounding_boxes(
                    image_row['[CameraBoxComponent].box.center.x'],
                                                    image_row['[CameraBoxComponent].box.center.y'],
                                                    image_row['[CameraBoxComponent].box.size.x'],
                                                    image_row['[CameraBoxComponent].box.size.y'])
            
            bboxes = np.array(bboxes)
            labels = np.array(image_row['[CameraBoxComponent].type'])
            target = {
                'boxes': bboxes,
                'labels': labels
            }

            if self.transform:
                image = self.transform(image)
                # labels = self.transform(labels)
                # bboxes = self.transform(bboxes)     
                       
                target = {
                    'boxes': torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)
                }

            return image, target


        


# Usage example
# waymo = WaymoDataset(image_dir='path_to_image_dir', box_dir='path_to_box_dir')
# image_row, bbox_rows = waymo[200]
