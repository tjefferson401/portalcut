from pathlib import Path

def calculate_overlap_ratio(rect1, rect2):
    # Unpack the rectangles
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    # Calculate the intersection coordinates
    intersect_x_min = max(x1_min, x2_min)
    intersect_y_min = max(y1_min, y2_min)
    intersect_x_max = min(x1_max, x2_max)
    intersect_y_max = min(y1_max, y2_max)

    # Check if there is an intersection
    if intersect_x_min < intersect_x_max and intersect_y_min < intersect_y_max:
        # Calculate the area of the intersection
        intersection_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
    else:
        intersection_area = 0

    # Calculate the area of the first rectangle
    area_rect1 = (x1_max - x1_min) * (y1_max - y1_min)

    # Calculate the ratio
    ratio = intersection_area / area_rect1

    return ratio

directory = Path('augmented_data/labels')

with open("obj_deletion_log.txt", "w") as f:
    total_obj_deleted = 0
    total_num_files = 0

    for filepath in directory.iterdir():
        if filepath.is_file():
            total_num_files += 1
            f.write(str(filepath) + "\n")
            new_content = ""
            
            with filepath.open('r') as curr:
                objs = curr.readlines()
                augmented = objs[-1]

                aug_rect = tuple(map(float, augmented.split()[4:8]))

                for obj in objs[:-1]:
                    curr_rect = tuple(map(float, obj.split()[4:8]))
                    r = calculate_overlap_ratio(curr_rect, aug_rect)

                    if r > 0.9:
                        f.write(f"- Deleting {obj[:-1]} (Overlap = {r})\n")
                        total_obj_deleted += 1
                    else:
                        new_content += obj
                
                new_content += augmented
            
            with filepath.open('w') as curr:
                curr.write(new_content)
            
            f.write("\n")

    f.write(f"{total_obj_deleted} objects deleted at an average of {total_obj_deleted / total_num_files} per file!")