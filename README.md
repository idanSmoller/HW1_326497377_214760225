# HW1 - Computer Vision in the OR

## environment setup:
```bash
  git clone https://github.com/idanSmoller/HW1_326497377_214760225.git
  cd HW1_326497377_214760225
  pip install -r requirements.txt
```

## training:
```bash
  python train_model.py
```
## inference:
```bash
  python video.py <model_path> <video_path> <(optional) output_path>
```
or
```bash
  python predict.py <model_path> <image_path> <(optional) output_path> <(optional) true_labels_path>
```

## pretrained models:
|model|description|download|
|---|---|---|
|initial_model.pt|Initial model trained on the original dataset|[initial_model.pt](initial_model.pt)|
|trained_model_1.pt|Trained model after the first refinement iteration|[trained_model_1.pt](trained_model_1.pt)|
|trained_model_2.pt|Trained model after the second refinement iteration|[trained_model_2.pt](trained_model_2.pt)|
|trained_model_3.pt|Trained model after the third refinement iteration|[trained_model_3.pt](trained_model_3.pt)|
|final_model.pt|Final model trained on the OOD dataset (not used in final version)|[final_model.pt](final_model.pt)|

