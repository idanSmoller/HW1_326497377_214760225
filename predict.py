from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path


def predict(model_path, image_path, annotation_save_path, true_labels_path=None):
    """
    Predicts objects in an image using a trained YOLO model and annotates the image with predictions.
    Args:
        model_path (str): Path to the trained YOLO model.
        image_path (str): Path to the image file for prediction.
        true_labels_path (str, optional): Path to the file containing true labels in YOLO format.
        If provided, the function will annotate the image with true labels as well.
    Returns:
        None
    """
    # Load your trained model
    model = YOLO(model_path)

    image_name = Path(image_path).name

    # Load image with OpenCV
    img = cv2.imread(image_path)

    # If true labels are provided, read them
    true_labels = None
    if true_labels_path:
        with open(true_labels_path, 'r') as f:
            true_labels = f.readlines()

    # Run inference (returns results)
    results = model.predict(img)

    # results is a list, get the first (and usually only) prediction
    result = results[0]

    # Display predictions on the image (draw boxes, labels, etc)
    annotated_img = result.plot()

    # If true labels are provided, annotate the image with them
    if true_labels:
        for label in true_labels:
            label = label.strip().split()
            class_id = int(label[0])
            x_center, y_center, width, height = map(float, label[1:])
            # Convert YOLO format to OpenCV rectangle format
            x1 = int((x_center - width / 2) * img.shape[1])
            y1 = int((y_center - height / 2) * img.shape[0])
            x2 = int((x_center + width / 2) * img.shape[1])
            y2 = int((y_center + height / 2) * img.shape[0])
            
            # Draw bigger rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=4)
            
            # Draw bigger and thicker text
            cv2.putText(annotated_img, f'Class {class_id}', (x1, y1 - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(255, 0, 0), thickness=3)

    # Save the annotated image
    if annotation_save_path:
        cv2.imwrite(annotation_save_path, annotated_img)
    else:
        # If no save path is provided, save with a default name
        cv2.imwrite(f'annotated_{image_name}', annotated_img)


def main():
    parser = argparse.ArgumentParser(description="Predict objects in an image using a trained YOLO model.")
    parser.add_argument('model_path', type=str, help='Path to the trained YOLO model.')
    parser.add_argument('image_path', type=str, help='Path to the image file for prediction.')
    parser.add_argument('--annotated_image_path', type=str, default=None, 
                        help='Path to save the annotated image. If not provided, saves as "annotated_<image_name>".')
    parser.add_argument('--true_labels_path', type=str, default=None, 
                        help='Path to the file containing true labels in YOLO format (optional).')

    args = parser.parse_args()

    predict(args.model_path, args.image_path, args.annotated_image_path, args.true_labels_path)


if __name__ == "__main__":
    main()