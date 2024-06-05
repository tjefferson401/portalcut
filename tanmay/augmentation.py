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

# API_KEY = "sk-KoPm8KUzA0NlDKI8U8AjbTbtuWFvKJEla3QgTmIg2jjwEGpi"
# API_KEY = "sk-kz7F0Vipz4gGTBZ0aROhGkwd0jcRuty9ZkHgM98fLLzo9Uaz"
API_KEY = "sk-8w4kBxXndXPyHtjw5GKEvWqC4oQTPWSaUPTnlrUsCkfrXZMg"

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
        box = np.array(box)
        # label = int(label)
        box = box.astype(int)
        print(label)
        label_text = label
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

def augmentWithSD_old(img, prompt="Add a man walking on the road without changing anything else in the scene"):
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
            "image_strength": 0.35,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": prompt,
            "cfg_scale": 7,
            "samples": 1,
            "steps": 40,
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

def plotPipeline(filename, type="default"):
    plt.figure(figsize=(20, 6))

    orig_image, _ = dataset[filename.split("_")[0]]
    plt.subplot(221).imshow(orig_image)
    plt.title("Original image from KITTI")
    plt.axis('off')

    generated = Image.open(f"augmented_data/diffused/{filename}.png")
    plt.subplot(222).imshow(generated)
    plt.title("Diffusion output image")
    plt.axis('off')

    # Display masked image
    mask_type = "box_masks" if type == "box" else "masks"
    masked_img = Image.open(f"augented_data/{mask_type}/{filename}.png")
    plt.subplot(223).imshow(masked_img)
    plt.title("Mask outputted by segmenter")
    plt.axis('off')

    # Save or display the combined image
    output_type = "box_outputs" if type == "box" else "outputs"
    combined_img = Image.open(f"augented_data/{output_type}/{filename}.png")
    plt.subplot(224).imshow(combined_img)
    plt.title("Final Output!")
    plt.axis('off')

    plt.show()
    # plt.savefig(f"augmented_data/pipelines/{filename}.png", bbox_inches='tight')

def augmentWithSD(img, prompt):
    # Create a BytesIO object
    image_byte_arr = io.BytesIO()

    # Save the image to the BytesIO object in PNG format
    img.save(image_byte_arr, format='PNG')

    # Get the byte data from the BytesIO object
    image_bytes = image_byte_arr.getvalue()

    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "authorization": f"Bearer {API_KEY}",
            "accept": "application/json"
        },
        files={"image": image_bytes},
        data={
            "prompt": prompt,
            "mode": "image-to-image",
            "model": "sd3-turbo",
            "strength": 0.65,
            "output_format": "png",
        },
    )

    if response.status_code == 200:
        out = response.json()
        aug_img = out["image"]

        return Image.open(io.BytesIO(base64.b64decode(aug_img)))
    else:
        raise Exception(str(response.json()))

def segmentGetLabel(img, person):
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

    req_type = "person" if person else "bicycle"

    # you can pass them to processor for postprocessing
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[img.size[::-1]], threshold=0.9)[0]
    segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}

    mask = None

    for key, val in segment_to_label.items():
        print("Label:", model.config.id2label[val])
        
        if model.config.id2label[val] == req_type or (not person and model.config.id2label[val] == "rider"):
            print("Visualizing mask for", key, ":", model.config.id2label[val])
            if mask is None:
                mask = (results['segmentation'].numpy() == key)
            else:
                mask = np.logical_or(mask, results['segmentation'].numpy() == key)
            
    return mask

def getCombinedFromMask(original, augmented, mask):
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Apply the mask to the generated image
    masked_image_np = augmented * mask_3d
    masked_image = Image.fromarray(masked_image_np.astype(np.uint8))

    # Use masked image to augment original image to obtain new image with new label
    combined_image_np = original * (1 - mask_3d) + masked_image_np
    combined_image = Image.fromarray(combined_image_np.astype(np.uint8))

    return masked_image, combined_image

def createData(sample, label, prompt, i):
    image, target = sample
    image_np = np.array(image)

    # generate image and display it
    generated = augmentWithSD(image, prompt)
    # generated.show()
    generated.save(f"augmented_data/misc/{i}.png")
    
    aug_image_np = np.array(generated)
    aug_image_np = np.array(Image.fromarray(aug_image_np).resize(image_np.shape[:2][::-1]))
    
    # Get mask for the object
    segment_label = False if label == "Cyclist" else True
    mask_np = segmentGetLabel(Image.fromarray(aug_image_np), segment_label)

    # TODO: smarter box extraction (only look at dense part)
    # Extract bounding box for object from mask
    non_mask_coords = np.where(mask_np == 1)
    lowest_x = np.min(non_mask_coords[0])
    highest_x = np.max(non_mask_coords[0])
    lowest_y = np.min(non_mask_coords[1])
    highest_y = np.max(non_mask_coords[1])
    box = [lowest_y, lowest_x, highest_y, highest_x]

    box_mask_np = mask_np.copy()
    box_mask_np[lowest_x:highest_x+1, lowest_y:highest_y+1] = 1

    masked_image, combined_image = getCombinedFromMask(image_np, aug_image_np, mask_np)
    box_masked_image, box_combined_image = getCombinedFromMask(image_np, aug_image_np, box_mask_np)
    
    # box_mask_3d = np.repeat(box_mask_np[:, :, np.newaxis], 3, axis=2)
    # box_masked_image_np = aug_image_np * box_mask_3d
    # box_combined_image_np = image_np * (1 - box_mask_3d) + box_masked_image_np
    # box_masked_image = Image.fromarray(box_masked_image_np.astype(np.uint8))
    # box_combined_image = Image.fromarray(box_combined_image_np.astype(np.uint8))
    # combined_image_box.show()

    # Update labels and boxes
    target.append({
        'type': label, 
        'truncated': 0.0, 
        'occluded': 0, 
        'alpha': -10, 
        'bbox': box, 
        'dimensions': [-1, -1, -1], 
        'location': [-1, -1, -1], 
        'rotation_y': -10
    })
    
    return (generated, masked_image, box_masked_image, combined_image, box_combined_image), target

dataset = datasets.Kitti(root='../data', download=False)
labels = ['background', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', "Don'tCare"]

# Execute pipeline
# image = Image.open("../data/Kitti/raw/training/image_2/000000.png")
num_worked = 0
constant = "Don't change anything else in the scene!"

prompts = [
    ("Pedestrian", "Add a clearly visible and realistic sized man crossing the road in the foreground of the picture. The man should not be taller than any houses, trees or cars. He should be wearing bright colors that match the vibe of the scene. "+constant),
    ("Pedestrian", "Add a clearly visible and realistic sized woman crossing the road in the foreground of the picture. The woman should not be taller than any houses, trees or cars. She should be wearing bright colors that match the vibe of the scene. "+constant),
    ("Pedestrian", "Add a clearly visible and realistic sized man walking on the sidewalk in the foreground of the picture. He should be looking at the viewer. He should be wearing bright colors that match the vibe of the scene. "+constant),
    ("Pedestrian", "Add a clearly visible and realistic sized woman walking on the sidewalk in the foreground of the picture. She should be looking at the viewer. She should be wearing bright colors that match the vibe of the scene. "+constant),
    ("Cyclist", "Add a realistic sized cyclist on the road. Make sure he is biking across the road. "+constant),
    ("Cyclist", "Add a realistic sized cyclist on the road. Make sure he is biking parallel to the road in the same direction as cars."+constant),
    ("Person_sitting", "Add a realistic sized person sitting on a bench on the side of the road. Make sure the bench is brown and the person is wearing contrasting colors to it. "+constant)
]

# ("Pedestrian", "Add a clearly visible and realistic person walking in the background of the picture. They should be wearing bright colors that match the vibe of the scene. "+constant),

curr_index = 653
folders = ["diffused", "masks", "box_masks", "outputs", "box_outputs"]

for prompt in prompts:
    break
    print("Starting new set of prompts for " + prompt[0] + "\n")

    for i in range(curr_index, curr_index+100):
        try:
            out_images, out_target = createData(dataset[i], prompt[0], prompt[1], i)
            timestamp = int(time.time() * 1000)

            for img, folder in zip(out_images, folders):
                img.save(f"augmented_data/{folder}/{i}_{timestamp}.png")

            with open(f"augmented_data/labels/{i}_{timestamp}.txt", "w") as f:
                for object in out_target:
                    object['bbox'] = " ".join(map(str, object['bbox']))
                    object['dimensions'] = " ".join(map(str, object['dimensions']))
                    object['location'] = " ".join(map(str, object['location']))

                    f.write(" ".join(map(str, object.values())) + "\n")
        
            num_worked += 1
        except Exception as e:
            print(e)
        
        print()
    
    curr_index += 100
    print(f"{num_worked} images for prompt {prompt[0]}")

print(f"{num_worked} new images created and saved!")

def visualize_augmented(filename):
    test_image = Image.open(f"augmented_data/outputs/{filename}.png")
    gen_labels = open(f"augmented_data/labels/{filename}.txt", "r").readlines()
    target = {'labels': [], 'boxes': []}

    for line in gen_labels:
        things = line.strip().split()
        target['labels'].append(things[0])
        target['boxes'].append(list(map(float, [things[4], things[5], things[6], things[7]])))

    get_tensor = get_transform()
    tensored = get_tensor(test_image)
    visualize_image_with_boxes(tensored, target['boxes'], target['labels'], labels)

# visualize_augmented("830_1717510676652")