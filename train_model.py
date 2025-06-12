import torch
from ultralytics import YOLO
import yaml
import os
import cv2
import tqdm
from pathlib import Path


YOLO_PATH = "yolo11n.pt"

SMALL_DATASET_YAML_PATH = "small_data.yaml"
BIG_DATASET_YAML_PATH = "big_data.yaml"
OOD_DATASET_YAML_PATH = "ood_data.yaml"

AUGMENTATION_CONFIG_PATH = "hyp.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_REFINEMENT_ITERATIONS = 3

INITIAL_MODEL_PATH = "initial_model.pt"
TRAINED_MODEL_PATH = lambda i: f"trained_model_{i}.pt"
FINAL_MODEL_PATH = "final_model.pt"

SMALL_DATASET_ROOT = "/datashare/HW1/"
BIG_DATASET_ROOT = "./big_dataset/"
OOD_DATASET_ROOT = "./ood_dataset/"
ID_VIDEOS_PATH = os.path.join(SMALL_DATASET_ROOT, "id_video_data")
OOD_VIDEOS_PATH = os.path.join(SMALL_DATASET_ROOT, "ood_video_data")

CLASSES_FILE_PATH = os.path.join(SMALL_DATASET_ROOT, "labeled_image_data", "classes.txt")
with open(CLASSES_FILE_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]


def train_model(model, save_path, dataset_yaml_path, augmentations_path, epochs=50, batch_size=32, model_name=None):
    """
    Train a YOLO model on the specified dataset.

    Args:
        model: YOLO model to be trained.
        save_path (str): Path to save the trained model.
        dataset_yaml_path (str): Path to the dataset YAML file.
        augmentations_path (str): Path to the augmentations YAML file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    Returns:
        model: The trained YOLO model.
    """
    # Load augmentations from the YAML file
    if augmentations_path is not None:
        with open(augmentations_path, 'r') as f:
            augmentations = yaml.safe_load(f)
    else:
        augmentations = {}

    # Train the model
    model.train(data=dataset_yaml_path, epochs=epochs, batch=batch_size, device=DEVICE, name=model_name, **augmentations)

    # Save the trained model
    model.save(save_path)

    print(f"Training completed and model saved as {save_path}")
    
    return model


def extract_frames(video_path, dataset_base, val_ratio=0.15):
    """
    Extract frames from a video file and save them in the dataset directory.
    Parameters:
    - video_path: Path to the video file.
    - dataset_base: Base directory where the frames will be saved.
    - val_ratio: Ratio of validation data (default is 0.15).
    """
    video_name = Path(video_path).stem
    train_dir = os.path.join(dataset_base, "images", "train")
    val_dir = os.path.join(dataset_base, "images", "val")

    cap = cv2.VideoCapture(video_path)
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    for frame_id in tqdm.tqdm(range(frames_num), desc=f"Extracting frames from {video_name}"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_dir = train_dir if frame_id % int(1 / val_ratio) != 0 else val_dir
        frame_file = os.path.join(frame_dir, f"{video_name}_{frame_id:04d}.jpg")
        cv2.imwrite(frame_file, frame)

    cap.release()


def generate_pseudo_labels(model, dataset_base):
    """
    Generate pseudo-labels for the dataset by running inference on all frames.
    Parameters:
    - model: a YOLOv11 model loaded with ultralytics.YOLO
    - dataset_base: Base directory where the dataset is stored.
    Generates labels in YOLO format and saves them in the labels directory.
    """
    train_frames_dir = os.path.join(dataset_base, "images", "train")
    val_frames_dir = os.path.join(dataset_base, "images", "val")
    train_frames = ["train/" + f for f in os.listdir(train_frames_dir)]
    val_frames = ["val/" + f for f in os.listdir(val_frames_dir)]
    frames_list = sorted(train_frames + val_frames, key=lambda x: Path(x).stem)
    enumerator = enumerate(frames_list)
    num_frames = len(frames_list)

    for frame_id, frame_file in tqdm.tqdm(enumerator, total=num_frames, desc="Processing frames"):
        frame = cv2.imread(os.path.join(dataset_base, "images", frame_file))
        frame_name = Path(frame_file).stem
        subset = "train" if frame_file.startswith("train/") else "val"

        # Inference
        results = model.predict(frame, device=DEVICE, verbose=False)
        # if no detections, print a warning and continue
        if not results or len(results) == 0 or len(results[0].boxes) == 0:
            print(f"Warning: No detections in frame {frame_id}. Skipping...")

        boxes = results[0].boxes
        preds = boxes.xywhn.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        detections = []
        for i in range(len(preds)):
            x_center, y_center, bw, bh = preds[i]
            cls = classes[i]

            detections.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
        
        label_file = os.path.join(dataset_base, "labels", subset, f"{frame_name}.txt")
        
        with open(label_file, "w") as f:
            f.write("\n".join(detections))


def make_big_dataset(model, videos_dir, dataset_base, data_yaml_path, classes, val_ratio=0.15, frames_exist=False):
    """
    Generate a large dataset from a directory of videos, saving images and labels in YOLO format.

    Parameters:
    - model: a YOLOv11 model loaded with ultralytics.YOLO
    - videos_dir: directory containing video files
    - dataset_base: directory where the dataset will be saved
    - data_yaml_path: path to save the dataset YAML file
    - classes: list of class names
    - conf_thresh: minimum confidence threshold for predictions
    - val_ratio: ratio of validation data
    - frames_exist: if True, frames are already extracted and stored in dataset_base/images

    Returns:
    - None
    """
    if not frames_exist:
        # Create directories for images and labels
        os.makedirs(os.path.join(dataset_base, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(dataset_base, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(dataset_base, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(dataset_base, 'labels', 'val'), exist_ok=True)
        
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        for video_file in video_files:
            extract_frames(os.path.join(videos_dir, video_file), dataset_base, val_ratio)
    
    generate_pseudo_labels(model, dataset_base)

    # create the classes.txt file
    with open(os.path.join(dataset_base, 'classes.txt'), 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    # create the dataset YAML file
    dataset_yaml = {
        'train': os.path.join(dataset_base, 'images', 'train'),
        'val': os.path.join(dataset_base, 'images', 'val'),
        'nc': len(classes),
        'names': classes
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)


def main():
    # Step 1: Train on the small dataset
    if os.path.exists(INITIAL_MODEL_PATH):
        print(f"Initial model already exists at {INITIAL_MODEL_PATH}. Skipping initial training.")
        model = YOLO(INITIAL_MODEL_PATH)
    else:
        print("Initial model not found. Training on the small dataset...")
        model = YOLO(YOLO_PATH)
        model = train_model(model, INITIAL_MODEL_PATH, SMALL_DATASET_YAML_PATH, AUGMENTATION_CONFIG_PATH, epochs=100, batch_size=32, model_name="initial_training")

    for i in range(NUM_REFINEMENT_ITERATIONS):
        if os.path.exists(TRAINED_MODEL_PATH(i + 1)):
            print(f"Trained model for iteration {i + 1} already exists at {TRAINED_MODEL_PATH(i + 1)}. Skipping retraining.")
            model = YOLO(TRAINED_MODEL_PATH(i + 1))
            continue

        print(f"Retraining for iteration {i + 1}...")
        # Step 2: Generate pseudo-labels for the big dataset
        print(f"Generating pseudo-labels for iteration {i + 1}...")
        make_big_dataset(model, ID_VIDEOS_PATH, BIG_DATASET_ROOT, BIG_DATASET_YAML_PATH, CLASSES, val_ratio=0.15, frames_exist=(i > 0))

        # Step 3: Retrain on the big dataset
        print(f"Retraining on the big dataset for iteration {i + 1}...")
        train_model(model, TRAINED_MODEL_PATH(i + 1), BIG_DATASET_YAML_PATH, AUGMENTATION_CONFIG_PATH, epochs=10, batch_size=32, model_name=f"refinement_{i + 1}")

    # Step 4: Generate pseudo-labels for the OOD videos
    print("Generating pseudo-labels for the OOD videos...")
    make_big_dataset(model, OOD_VIDEOS_PATH, OOD_DATASET_ROOT, OOD_DATASET_YAML_PATH, CLASSES, val_ratio=0.15, frames_exist=False)

    # Step 5: Final training on the combined dataset
    print("Final training on the OOD dataset...")
    train_model(model, FINAL_MODEL_PATH, OOD_DATASET_YAML_PATH, None, epochs=25, batch_size=32, model_name="final_training")
    

if __name__ == "__main__":
    main()