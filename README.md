# Colorectal Histology Classification with CNN 
### A convolutional neural network (CNN) built with TensorFlow to classify colorectal tissue images into 8 histological categories. The project covers the full ML pipeline: data loading, preprocessing, model architecture, training with callbacks, fine-tuning, and evaluation.

Dataset
Source: TensorFlow Datasets — colorectal_histology

Total images: 5,000
Classes (8): Adipose, Background, Debris, Lymphoma, Mucosa, Muscle, Stroma, Tumor
Split: 70% train / 15% validation / 15% test (shuffled with seed=42 for reproducibility)

Pipeline Overview
1. Import Libraries
TensorFlow, TensorFlow Datasets, NumPy, Matplotlib, scikit-learn.

2. Load Dataset
Downloads colorectal_histology via TFDS and prints class distribution.

3. Train/Validation/Test Split
70 / 15 / 15 split applied after shuffling the full dataset.
   
4. Preprocessing & Augmentation
Resize all images to 160 × 160
Normalize pixel values to [0, 1]
Training augmentation: random horizontal flip, rotation (±15%), zoom (±10%), translation (±10%), contrast adjustment

5. CNN Architecture
LayerDetailsConv Block 1Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → BatchNormConv Block 2Conv2D(64, 3×3, ReLU) → MaxPool(2×2) → BatchNormConv Block 3Conv2D(128, 3×3, ReLU) → MaxPool(2×2) → BatchNormPoolingGlobalAveragePooling2D → FlattenDenseDense(128, ReLU) → Dropout(0.4)OutputDense(8, Softmax)

6. Initial Training
Optimizer: Adam (lr=1e-3)
Loss: Sparse Categorical Crossentropy
Epochs: up to 30 with EarlyStopping (patience=3, restores best weights)

7. Fine-Tuning
Re-compiled with reduced learning rate (lr=1e-4)
Runs for an additional 20 epochs to refine the learned weights

8. Training Curves
Combined accuracy and loss plots across both training phases, with a vertical marker showing where fine-tuning begins.

9. Evaluation
Test set accuracy and loss via model.evaluate()
Macro and weighted F1 scores
Full classification report per class

10. Confusion Matrix
Color-coded grid showing true vs. predicted labels across all 8 tissue classes.

11. Sample Predictions
Visual comparison of correctly classified and misclassified examples from the test set, with confidence scores on misclassifications.
