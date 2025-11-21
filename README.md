# Deepfake Detection with Vision Transformers and CNN Hybrid Architecture

## Overview

This project implements a robust deepfake detection pipeline using a combination of Convolutional Neural Networks (CNNs) and Vision Transformers (ViT). The pipeline consists of three main phases:

1. **CNN Model Training**: A custom CNN is trained from scratch to identify deepfake images.
2. **ViT Fine-Tuning**: A pre-trained Vision Transformer is fine-tuned on the CNN’s learned features to boost performance.
3. **Explainability**: Attention rollout visualizations highlight which parts of the image influenced the ViT’s decisions, offering interpretability.

## Features

* End-to-end training and evaluation on image datasets.
* Detailed performance metrics (accuracy, precision, recall, F1-score).
* Visualizations:

  * Training and Validation Loss/Accuracy curves
  * Confusion Matrix
  * Attention Rollout Maps for explainability


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Dataset is take from:
```
"Deepfakes: Medical Image Tamper Detection," UCI Machine Learning Repository, 2020. [Online]. Available: https://doi.org/10.24432/C5J318.
```

Adjust the paths in the notebook or configuration file if needed.

## Usage

1. **Train the CNN model**:

   * Open `deepfake-detection.ipynb` and run the "CNN Training" cells.
   * The model checkpoints and final weights will be saved to `models/cnn_model.h5`.

2. **Fine-tune the ViT**:

   * Continue in the notebook’s "ViT Fine-Tuning" section.
   * The combined model will be saved as `models/deepfake_detection_model.h5`.
  
  ![m1](https://github.com/user-attachments/assets/4b8c0365-7899-4a6a-a734-36dc706f8494)


3. **Evaluate the Model**:

   ```python
   from tensorflow.keras.models import load_model
   from utils.evaluate import evaluate_on_testset

   model = load_model("models/deepfake_detection_model.h5")
   metrics = evaluate_on_testset(model, test_dir)
   print(metrics)
   ```

## Results

* **Test Accuracy**: **97.60%**
* **Precision**: 98%
* **Recall**: 97%
* **F1-score**: 98%

### Training Loss Curves

![g2](https://github.com/user-attachments/assets/6339aa2c-a1c6-4c55-ab85-0e1bd94e91cf)
![g1](https://github.com/user-attachments/assets/1dd44566-19fe-49ed-90ba-907c735fd772)

### Confusion Matrix

![c1](https://github.com/user-attachments/assets/cfdc9194-5f40-412e-81d5-943e8a36c0a1)


## Explainability

The explainability module employs **attention rollout** to trace and visualize how information flows through the Vision Transformer (ViT) layers, highlighting the image regions that most influenced the model’s prediction. Key components include:

1. **Attention Extraction**: We extract the attention weights from each ViT layer’s multi-head attention mechanism during inference.
2. **Rollout Computation**: Attention rollout aggregates attention across layers by multiplying normalized attention matrices, as described in Abnar & Zuidema (2020). This identifies long-range dependencies and composite feature importance.
3. **Heatmap Generation**: The aggregated attention matrix is reshaped and upsampled to the original image dimensions, producing a heatmap overlay. We apply a colormap (e.g., Viridis) to distinguish high-attention (warm colors) from low-attention (cool colors).
4. **Visualization**: Overlay heatmaps on sample images to interpret model focus:

   * **Real**: Attention centers on cells features and its structures, indicating reliance on fine-grained details for authenticity checks.
   * **Fake**: Attention often highlights boundary artifacts and blending regions, revealing the model’s sensitivity to synthesis imperfections.

### How to Generate Attention Rollouts

```python
from explainability import AttentionRollout
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load trained ViT model
model = load_model('models/deepfake_detection_model.h5', compile=False)

# Initialize rollout object
rollout = AttentionRollout(model, head_fusion='mean', discard_ratio=0.9)

# Process input image
img = image.load_img('data/test/fake/fake1.jpg', target_size=(224,224))
img_array = image.img_to_array(img)/255.0

# Generate attention map
mask = rollout(img_array)

```

![exp](https://github.com/user-attachments/assets/0e77f02b-0c33-4d5f-98bf-0920827a0fb1)

### Interpretation & Insights

* **Model Trust**: By observing that the ViT attends to semantically meaningful regions, we gain confidence in the model’s reasoning process.
* **Failure Analysis**: Cases where attention highlights irrelevant background suggest potential dataset biases or need for further data augmentation.
* **Guiding Improvements**: Insights from explainability can inform architecture tweaks (e.g., adjusting patch sizes) or targeted data collection to improve robustness.
