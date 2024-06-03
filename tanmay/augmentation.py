# take images from kitti dataset
# augment them by adding stuff to it
# ask stability ai to take input image
# and add smth to it (like pedestrian/cyclist)
# then ask SAM to extract the pedestrian/cyclist
# and add the pedestrian to original image with
# box label for it as a new image in dataset

# Challenges:
# 1. stable diffusion is shit
# 2. image segmentation
# 3. where to add back in original img

import io, time
import requests
import base64
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

API_KEY = "sk-KoPm8KUzA0NlDKI8U8AjbTbtuWFvKJEla3QgTmIg2jjwEGpi"

def get_transform():
    transform = [ToTensor()]
    return Compose(transform)

def visualize_image_with_boxes(image, boxes, labels, label_names):
    # Convert tensor image to numpy array
    image = image.numpy().transpose((1, 2, 0))
    # Scale the image's pixel values to [0, 255]
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    # Define colors for different classes
    colors = {
        'Car': (255, 0, 0), 'Van': (0, 255, 0), 'Truck': (0, 0, 255),
        'Pedestrian': (255, 255, 0), 'Person_sitting': (255, 0, 255), 'Cyclist': (0, 255, 255),
        'Tram': (127, 127, 255), 'Misc': (255, 127, 127), "Don'tCare": (127, 127, 127)
    }

    # Draw boxes and labels
    for box, label in zip(boxes, labels):
        box = box.numpy()
        label = int(label)
        box = box.astype(int)
        print(label)
        label_text = label_names[label]
        color = colors.get(label_text, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Put label
        cv2.putText(image, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def augmentWithSD(img, prompt="Add a male cyclist on the road without changing anything else in the scene"):
    # Create a BytesIO object
    image_byte_arr = io.BytesIO()

    # Save the image to the BytesIO object in PNG format
    img.save(image_byte_arr, format='PNG')

    # Get the byte data from the BytesIO object
    image_bytes = image_byte_arr.getvalue()

    response = requests.post(
        f"https://api.stability.ai/v1/generation/stable-diffusion-v1-6/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        files={
            "init_image": image_bytes
        },
        data={
            "image_strength": 0.4,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": prompt,
            "cfg_scale": 7,
            "samples": 1,
            "steps": 30,
        }
    )

    if response.status_code == 200:
        out = response.json()
        aug_img = out["artifacts"][0]
        
        # writing to file is unnecessary and will be removed eventually
        # with open("./sample_test_cyclist2.png", 'wb') as file:
        #     file.write(base64.b64decode(aug_img["base64"]))
        return Image.open(io.BytesIO(base64.b64decode(aug_img["base64"])))
    else:
        raise Exception(str(response.json()))

class KittiTorch(datasets.Kitti):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # # Adding +1 to all indices to reserve 0 for background
        # labels = [1 + ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'].index(t['type']) for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        # boxes = [t['bbox'] for t in target if t['type'] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']]
        # target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        # 'labels': torch.as_tensor(labels, dtype=torch.int64)}
        return image, target


def segmentGetLabel(img):
    # use inference (instance segmentation): https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mask2Former/Inference_with_Mask2Former.ipynb

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to processor for postprocessing
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[img.size[::-1]], threshold=0.9)[0]
    segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}

    for key, val in segment_to_label.items():
        print("Label:", model.config.id2label[val])
        if model.config.id2label[val] == "bicycle":
            print("Visualizing mask for", key, ":", model.config.id2label[val])
            mask = (results['segmentation'].numpy() == key)
            return mask

def createData(sample):
    image, target = sample
    plt.subplot(221).imshow(image)
    plt.title("Original image from KITTI")

    image_np = np.array(image)

    # generate image and display it (default cyclist)
    generated = augmentWithSD(image)
    plt.subplot(222).imshow(generated)
    plt.title("Diffusion output image")
    generated.show()
    
    # image2 = Image.open("sample_test_cyclist.png")

    # Get mask for the cyclist
    mask_np = segmentGetLabel(generated)
    mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)

    # Extract bounding box for object from mask
    non_mask_coords = np.where(mask_np == 1)
    lowest_x = np.min(non_mask_coords[0])
    highest_x = np.max(non_mask_coords[0])
    lowest_y = np.min(non_mask_coords[1])
    highest_y = np.max(non_mask_coords[1])
    box = [lowest_x, lowest_y, highest_x, highest_y]

    # Apply the mask to the original image
    aug_image_np = np.array(generated)
    masked_image_np = aug_image_np * mask_3d

    # Convert the result back to a PIL image
    masked_image = Image.fromarray(masked_image_np.astype(np.uint8))

    # Display masked image
    plt.subplot(223).imshow(masked_image)
    plt.title("Mask outputted by segmenter")

    # Use masked image to augment original image to obtain new image with new label
    combined_image_np = image_np * (1 - mask_3d) + masked_image_np

    # Convert the resulting NumPy array back to a PIL image
    combined_image = Image.fromarray(combined_image_np.astype(np.uint8))

    # Save or display the combined image
    plt.subplot(224).imshow(combined_image)
    plt.title("Final Output!")
    
    # Update labels and boxes
    target.append({
        'type': 'Cyclist', 
        'truncated': 0.0, 
        'occluded': 0, 
        'alpha': -10, 
        'bbox': box, 
        'dimensions': [-1, -1, -1], 
        'location': [-1, -1, -1], 
        'rotation_y': -10
    })
    
    plt.axis('off')
    plt.show()

    return combined_image, target

dataset = KittiTorch(root='../data', download=False)
labels = ['background', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', "Don'tCare"]

# Execute pipeline
out_image, out_target = createData(dataset[2])

timestamp = int(time.time() * 1000)
out_image.save(f"images/{timestamp}.png")

with open(f"labels/{timestamp}.txt", "w") as f:
    out_target['bbox'] = " ".join(out_target['bbox'])
    out_target['dimensions'] = " ".join(out_target['dimensions'])
    out_target['location'] = " ".join(out_target['location'])

    f.write(" ".join(out_target.values()))

# get_tensor = get_transform()
# tensored = get_tensor(image2)
# visualize_image_with_boxes(tensored, target['boxes'], target['labels'], labels)