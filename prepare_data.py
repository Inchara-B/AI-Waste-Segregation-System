import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

# --- CONFIGURATION ---
# 1. Path to the downloaded TACO dataset folder
# line 8 in prepare_data.py
data_dir = Path("downloads/archive (1)/data")# IMPORTANT: Change this path

# 2. Path to where you want to save your new YOLOv8 dataset
output_dir = Path("taco_yolov8")

# 3. Desired train/validation split ratio
validation_split_ratio = 0.2
# --- END CONFIGURATION ---


# --- SCRIPT LOGIC ---
def convert_taco_to_yolo():
    """
    Converts the TACO dataset from COCO JSON format to YOLOv8 format.
    """
    # Create output directories
    output_images_train = output_dir / "images" / "train"
    output_images_val = output_dir / "images" / "val"
    output_labels_train = output_dir / "labels" / "train"
    output_labels_val = output_dir / "labels" / "val"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_train.mkdir(parents=True, exist_ok=True)
    output_images_val.mkdir(parents=True, exist_ok=True)
    output_labels_train.mkdir(parents=True, exist_ok=True)
    output_labels_val.mkdir(parents=True, exist_ok=True)

    # Load the COCO annotations file
    annotations_file = data_dir / "annotations.json"
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Create mapping from category ID to category name
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    category_names = [cat['name'] for cat in sorted(data['categories'], key=lambda x: x['id'])]

    # Create mapping from image ID to image info (filename, width, height)
    images_info = {img['id']: img for img in data['images']}
    
    # Group annotations by image ID
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    all_image_ids = list(images_info.keys())
    train_ids, val_ids = train_test_split(all_image_ids, test_size=validation_split_ratio, random_state=42)

    print(f"Total images: {len(all_image_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")

    # Process train and validation sets
    for split_name, image_ids in [("train", train_ids), ("val", val_ids)]:
        image_out_path = output_dir / "images" / split_name
        label_out_path = output_dir / "labels" / split_name
        
        for image_id in image_ids:
            img_info = images_info[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            # Copy image to the new split folder
            source_image_path = data_dir / img_filename
            if source_image_path.exists():
                (image_out_path / Path(img_filename).name).write_bytes(source_image_path.read_bytes())
            else:
                print(f"Warning: Image file not found: {source_image_path}")
                continue

            # Create YOLO annotation file
            label_file_path = label_out_path / (Path(img_filename).stem + ".txt")
            with open(label_file_path, 'w') as label_file:
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        category_id = ann['category_id']
                        bbox = ann['bbox']
                        x, y, w, h = bbox

                        # Convert COCO bbox (top-left x, top-left y, width, height) to YOLO format
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        norm_w = w / img_width
                        norm_h = h / img_height
                        
                        # Assuming category IDs in JSON are 1-based, YOLO needs 0-based
                        class_index = category_id -1 

                        label_file.write(f"{class_index} {x_center} {y_center} {norm_w} {norm_h}\n")
    
    # Create the data.yaml file for YOLOv8
    yaml_data = {
        'train': f'../{output_dir.name}/images/train',
        'val': f'../{output_dir.name}/images/val',
        'nc': len(category_names),
        'names': category_names
    }

    yaml_file_path = output_dir / "data.yaml"
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print("\nConversion complete!")
    print(f"YOLOv8 dataset created at: {output_dir.resolve()}")
    print(f"YAML file created at: {yaml_file_path.resolve()}")


if __name__ == "__main__":
    convert_taco_to_yolo()