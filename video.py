from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def video_predict(model_path, video_path):
    """
    Predicts objects in a video using a trained YOLO model and annotates the video with predictions.
    Args:
        model_path (str): Path to the trained YOLO model.
        video_path (str): Path to the video file for prediction.
    Returns:
        None
    """
    # Load your trained model
    model = YOLO(model_path)
    video_name = Path(video_path).name

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    # Get video properties
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'annotated_{video_name}', fourcc, fps, (width, height))
    
    # Process each frame
    for _ in tqdm(range(frames_num), desc=f"Processing {video_name}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference (returns results)
        results = model.predict(frame, verbose=False)

        # results is a list, get the first (and usually only) prediction
        result = results[0]

        # Display predictions on the frame (draw boxes, labels, etc)
        annotated_frame = result.plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)
    # Release resources
    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Predict objects in a video using a trained YOLO model.")
    parser.add_argument("model_path", type=str, help="Path to the trained YOLO model.")
    parser.add_argument("video_path", type=str, help="Path to the video file for prediction.")

    args = parser.parse_args()

    video_predict(args.model_path, args.video_path)


if __name__ == "__main__":
    main()