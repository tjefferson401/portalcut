import cv2, shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor, Compose

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

    last_obj = None
    last_box = None

    # Draw boxes and labels
    for box, label in zip(boxes, labels):
        box = np.array(box)
        box = box.astype(int)
        color = colors.get(label, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Put label
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        last_obj = label
        last_box = box

    print(last_obj, last_box)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

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

def move_file(source, new_folder, file, sub_folders=["diffused", "labels", "masks", "outputs"]):
    for folder in sub_folders:
        if folder == "labels":
            extension = ".txt"
        else:
            extension = ".png"

        shutil.move(f"{source}/{folder}/{file}{extension}", f"{new_folder}/{folder}")

labels = ['background', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', "Don'tCare"]

directory = Path('augmented_data/outputs')
deleted_list = []
edited_list = []
edited_box_list = []
num_total_images = 1
num_good_images = 0

for filepath in directory.iterdir():
    if filepath.is_file():
        filepath = str(filepath)

        if filepath.endswith(".png"):
            file = filepath.split("/")[-1]
            filename = file.split(".")[0]

            print(f"\n{num_total_images}. Augmentation for {file}", end=": ")
            visualize_augmented(filename)
            # continue # Uncomment to just view all files

            action = input("Delete or edit: ").upper()
            
            if action == "D":
                move_file("augmented_data", "bad_data", filename)
                deleted_list.append(file)
            elif action == "E":
                topleft_x = input("edit topleft x? ")
                topleft_y = input("edit topleft y? ")
                bottomright_x = input("edit bottomright x? ")
                bottomright_y = input("edit bottomright y? ")

                label_file = f"augmented_data/labels/{filename}.txt"
                new_content = ""
                replaced_list = []

                with open(label_file, 'r') as f:
                    content = f.readlines()

                    for idx in range(len(content)):
                        line = content[idx]
                        things = line.strip().split()
                        
                        if idx == len(content)-1:
                            if topleft_x != "":
                                things[4] = topleft_x
                                replaced_list.append(4)
                                
                            if topleft_y != "":
                                things[5] = topleft_y
                                replaced_list.append(5)
                            
                            if bottomright_x != "":
                                things[6] = bottomright_x
                                replaced_list.append(6)

                            if bottomright_y != "":
                                things[7] = bottomright_y
                                replaced_list.append(7)

                        new_content +=  " ".join(things) + "\n"
                
                with open(label_file, 'w') as f:
                    f.write(new_content)
                
                print("Replaced:", replaced_list)
                edited_box_list.append(replaced_list)

                edited_list.append(file)
            elif action == "X":
                break
            else:
                num_good_images += 1
            
            num_total_images += 1

print(f"A total of {num_total_images+1} images were reviewed!")
print(f"A total of {len(deleted_list)} images were deleted!")
print(f"A total of {len(edited_list)} images were edited!")
print(f"A total of {num_good_images} good images found!\n")

print("Deleted:", deleted_list)
print("Edited:", edited_list)
print("Box Edits:", edited_box_list)