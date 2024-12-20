
# People Clothing Segmentation Project

---

# Table of Sections

| **Section** | **Description**                            |
|-------------|--------------------------------------------|
| 2           | Project Configuration                      |
| 3           | Dataset Loading and Preprocessing          |
| 4           | Dataset Visualization                      |
| 5          | Model Architectures Overview               |
| 6           | UNet Model                                 | 
| 7          | Conclusion and Future Work                 |

---

# Section 1: Introduction

## 1.1 Project Overview
The primary objective of this project is to perform clothing segmentation from images using deep learning models. The implementation includes popular models such as UNet and DeepLabV3, focusing on accuracy and model interpretability.

## 1.2 Expected Outcomes
- **High Accuracy:** Achieving precise segmentation for clothing.
- **Model Interpretability:** Explaining model decisions through visualization.
- **Scalability:** Deploying a robust system for real-world environments.

## 1.3 Technology Stack
- **Frameworks:** TensorFlow, Keras
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Pretrained Models:** DeepLabV3
- **Development Environment:** Jupyter Notebook
- **Deployment:** FastAPI, ONNX

---

# Section 2: Project Configuration

## 2.1 Library Imports
We import essential libraries for model development, evaluation, and visualization.

## 2.2 Hyperparameter Setup
-![alt text](image-1.png)

This code defines hyperparameters and settings for a semantic segmentation task related to people and clothing segmentation. Here's a detailed explanation:

### 1. **Image Dimensions**
```python
H, W = 512, 512
```
- **Purpose**: Sets the height and width of the input images.
- **Reason**: Standardizing image sizes simplifies training and improves computational efficiency.
- **Benefit**: Ensures consistent input size, enabling batch processing and compatibility with deep learning models.

### 2. **Batch Size**
```python
BATCH_SIZE = 2
```
- **Purpose**: Defines the number of samples processed before the model is updated.
- **Reason**: A small batch size reduces memory usage, allowing training on memory-constrained GPUs.
- **Benefit**: Achieves a balance between memory efficiency and model performance.

### 3. **Number of Classes**
```python
N_CLASSES = 24
```
- **Purpose**: Sets the number of segmentation classes for clothing types.
- **Reason**: Each class represents a specific clothing item or body part.
- **Benefit**: Enables multi-class segmentation for complex tasks.

### 4. **Learning Rate**
```python
LR = 5e-5
```
- **Purpose**: Determines the step size at each iteration during optimization.
- **Reason**: A small learning rate stabilizes training, especially for deep models.
- **Benefit**: Reduces the risk of overshooting the minimum of the loss function.

### 5. **Number of Epochs**
```python
N_EPOCHS = 20
```
- **Purpose**: Specifies how many times the entire dataset is passed through the model.
- **Reason**: Sufficient epochs ensure proper convergence.
- **Benefit**: Allows the model to learn complex patterns without overfitting.

### 6. **Weight Decay Rate**
```python
WEIGHT_DECAY_RATE = 0.01
```
- **Purpose**: Adds L2 regularization to the loss function to prevent overfitting.
- **Reason**: Penalizes large weights to improve generalization.
- **Benefit**: Helps reduce model complexity and prevents overfitting.

### 7. **Mean and Standard Deviation**
```python
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
```
- **Purpose**: Normalize the input images by standardizing pixel values.
- **Reason**: Preprocessing ensures consistent model input scaling.
- **Benefit**: Improves training stability and convergence speed.

### 8. **Checkpoint Filepath**
```python
checkpoint_filepath = "/content/drive/MyDrive/fiftyone/segformer_b5_clothing.h5"
```
- **Purpose**: Specifies where the model's weights are saved during training.
- **Reason**: Allows checkpointing to resume training if interrupted.
- **Benefit**: Ensures model progress is preserved and allows evaluation of the best model.



---

# Section 3: Dataset Loading and Preprocessing

## 3.1 Dataset Source
![alt text](image.png)
I'll review the file contents to explain the data preparation process in detail. Give me a moment.

The data preparation process in the notebook involves several key steps. Here’s a breakdown:

### 1. **Dataset Installation and Environment Setup**
```python
!pip install -q datasets evaluate
!pip install fiftyone==0.23.0rc1
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q regex tqdm
!pip install -q diffusers transformers accelerate scipy
!pip install -q -U xformers
```
- **Purpose**: Installs essential libraries for data processing, evaluation, and training.
- **Reason**: Ensures all dependencies are installed for seamless execution.
- **Benefit**: Automates environment setup, saving manual installation effort.

---

### 2. **Dataset Download from Kaggle**
```python
!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d rajkumarl/people-clothing-segmentation
```
- **Purpose**: Downloads the people clothing segmentation dataset from Kaggle.
- **Reason**: Provides access to the required dataset.
- **Benefit**: Automates dataset acquisition, making the process reproducible.

---

### 3. **Data Extraction**
```python
!unzip "/content/people-clothing-segmentation.zip" -d "/content/dataset/"
```
- **Purpose**: Extracts the downloaded dataset into a specified directory.
- **Reason**: Prepares the data files for easy access during loading.
- **Benefit**: Organizes dataset files in a structured format for training and evaluation.

---

### 4. **Data Loading**
```python
df = pd.read_csv("/content/dataset/labels.csv")
```
- **Purpose**: Loads the dataset’s metadata from a CSV file.
- **Reason**: Retrieves image file paths and labels for segmentation tasks.
- **Benefit**: Provides a structured view of the dataset for easy iteration.

---

### 5. **Data Paths Definition**
```python
im_path = "/content/dataset/png_images/IMAGES/"
anno_path = "/content/dataset/png_masks/MASKS/"
val_im_path = "/content/val_dataset/png_images/IMAGES/"
val_anno_path = "/content/val_dataset/png_masks/MASKS/"
```
- **Purpose**: Defines paths for training and validation datasets.
- **Reason**: Segregates image and mask data for efficient loading.
- **Benefit**: Simplifies file management and prevents path-related errors.

---

The preprocessing and data augmentation steps in the notebook are organized into well-defined functions. Here’s a detailed explanation:

---

### **1. Preprocessing Function**
```python
def preprocess(im_path, anno_path):
    img = tf.io.decode_jpeg(tf.io.read_file(im_path))
    img = tf.cast(img, tf.float32)
    img = (img - MEAN) / STD

    anno = tf.io.decode_jpeg(tf.io.read_file(anno_path))
    anno = tf.cast(tf.squeeze(anno, -1), tf.float32)

    return img, anno
```

#### **Explanation:**
- **Image Loading**: Reads the image and annotation mask using TensorFlow’s file I/O functions.
- **Normalization**: Subtracts the mean and divides by the standard deviation to normalize pixel values.
- **Annotation Processing**: Squeezes the last dimension of the annotation mask to match the expected shape.

#### **Reason and Benefit:**
- This function ensures input consistency and normalization, aiding in stable training and faster convergence.

---

### **2. Preprocessing Pipeline Setup**
```python
prep_train_ds = (
    train_dataset
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
)

prep_val_ds = (
    val_dataset
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
)
```

#### **Explanation:**
- **Mapping**: Applies the preprocessing function to the training and validation datasets.
- **Parallel Processing**: Uses TensorFlow's `AUTOTUNE` to parallelize loading and preprocessing.

#### **Reason and Benefit:**
- Ensures efficient and parallelized data loading, minimizing I/O bottlenecks.

---

### **3. Data Augmentation Functions (Using Albumentations)**
```python
def aug_albument(image, mask):
    augmented = transform(image=image, mask=mask)
    return [tf.convert_to_tensor(augmented["image"], dtype=tf.float32),
            tf.convert_to_tensor(augmented["mask"], dtype=tf.float32)]

def val_aug_albument(image, mask):
    augmented = val_transform(image=image, mask=mask)
    return [tf.convert_to_tensor(augmented["image"], dtype=tf.float32),
            tf.convert_to_tensor(augmented["mask"], dtype=tf.float32)]
```

#### **Explanation:**
- **Data Augmentation**: Applies augmentations defined in `transform` and `val_transform` (likely set elsewhere).
- **Conversion**: Converts augmented outputs to TensorFlow tensors.

#### **Reason and Benefit:**
- Increases data variability and prevents overfitting by introducing image transformations such as rotations, flips, and scaling.

---

### **4. Augmentation Wrappers**
```python
def augment(image, mask):
    aug_output = tf.numpy_function(func=aug_albument, inp=[image, mask], Tout=[tf.float32, tf.float32])
    return {"pixel_values": tf.transpose(aug_output[0], (2, 0, 1)), "labels": aug_output[1]}

def val_augment(image, mask):
    aug_output = tf.numpy_function(func=val_aug_albument, inp=[image, mask], Tout=[tf.float32, tf.float32])
    return {"pixel_values": tf.transpose(aug_output[0], (2, 0, 1)), "labels": aug_output[1]}
```

#### **Explanation:**
- **Wrapping**: Uses `tf.numpy_function` to apply Albumentations-based augmentation while preserving TensorFlow compatibility.
- **Transposition**: Rearranges tensor dimensions to match the model’s expected input format.

#### **Reason and Benefit:**
- Integrates a powerful third-party augmentation library while maintaining TensorFlow compatibility.

---

### **5. Dataset Finalization**
```python
BATCH_SIZE = 2

train_ds = (
    prep_train_ds
    .shuffle(10)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    prep_val_ds
    .map(val_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
```

#### **Explanation:**
- **Shuffling**: Randomly shuffles the training data to reduce model bias.
- **Batching**: Groups samples into batches for efficient GPU processing.
- **Prefetching**: Loads data asynchronously to improve training speed.

#### **Reason and Benefit:**
- Optimizes the data pipeline for maximum efficiency, reducing model training latency and improving overall performance.

---

These steps ensure robust data preparation through normalization, augmentation, and pipeline optimization, enabling effective training of the segmentation model.

# Section 4: Dataset Visualization

## 4.1 Sample Visualization
- ![alt text](image-2.png)

The data visualization in the notebook uses Matplotlib to display training samples and their corresponding segmentation labels. Here’s a detailed explanation:
---

### **Explanation:**
1. **Figure Initialization**
   ```python
   plt.figure(figsize=(50,50))
   ```
   - Initializes a large Matplotlib figure for displaying results.

2. **Data Extraction**
   ```python
   for data in train_ds.take(1):
       images = data['pixel_values']
       labels = data['labels']
   ```
   - Extracts one batch from the training dataset.
   - `images` holds the input images.
   - `labels` holds corresponding segmentation masks.

3. **Image and Segmentation Display**
   ```python
   for i in range(BATCH_SIZE*2):
       if i == 4:
           break
       ax = plt.subplot(1, BATCH_SIZE*2, i+1)
   ```
   - Loops through both images and labels, creating subplots for display.
   - `BATCH_SIZE*2` ensures equal number of images and labels.

4. **Conditional Plotting**
   ```python
   if i % 2 == 0:
       plt.imshow(tf.transpose(images[i//2], (1, 2, 0)))
       plt.title("Image")
   else:
       plt.imshow(labels[i//2])
       plt.title("Segmentation")
   ```
   - Displays alternating images and segmentation masks:
     - **Even Indices**: Shows the input image, transposing channels to RGB.
     - **Odd Indices**: Shows the corresponding segmentation mask.

5. **Formatting**
   ```python
   plt.axis("off")
   ```
   - Turns off axes to keep the focus on visual content.

6. **Final Display**
   ```python
   plt.show()
   ```
   - Renders the visualization.

---

### **Reason and Benefit:**
- **Reason**: This visualization step helps verify that data is correctly preprocessed and segmented.
- **Benefit**: Provides immediate feedback on data integrity, aiding in debugging and visual confirmation of the model's expected input/output structure.---

# Section 5: Model Architectures Overview

The notebook uses the **SegFormer** architecture for semantic segmentation. Here's a detailed explanation of the model architecture and related configurations:

---

### **1. Model Import and Initialization**
```python
from transformers import TFSegformerForSemanticSegmentation

model_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_id,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
```

---

### **Model Overview**
**SegFormer** is a transformer-based architecture for semantic segmentation, designed for high accuracy and efficiency. It uses hierarchical transformers for multi-scale feature extraction.

---

### **Model Architecture Breakdown**
1. **Hierarchical Transformer Encoder**:
   - Extracts multi-level features at different scales.
   - Each level processes features independently, using attention mechanisms.

2. **Multi-Scale Feature Fusion (MSF)**:
   - Combines features from all levels into a unified representation.

3. **Segmentation Head**:
   - Applies convolutional layers to generate segmentation masks.

---

### **Model Configuration Details**
- **Pretrained Model ID**: `"nvidia/segformer-b5-finetuned-cityscapes-1024-1024"`
- **Fine-Tuned for Cityscapes Dataset**: This pretrained model is adapted for urban segmentation tasks.
- **Number of Labels**:
  ```python
  num_labels=len(label2id)
  ```
  - Dynamically set based on the dataset being used (24 classes in this task).
- **Label Mappings**:
  ```python
  id2label=id2label
  label2id=label2id
  ```
  - Maps numerical IDs to human-readable class names.

---

### **Model Inference Example**
```python
model(tf.ones([1, 3, 512, 512])).logits.shape
```
- **Input Format**: A dummy tensor of shape `[1, 3, 512, 512]` simulating a single RGB image.
- **Output Shape**: The `logits` tensor's shape indicates the predicted segmentation mask dimensions.

---

### **Reason and Benefit**
- **Reason**: The SegFormer model was chosen for its state-of-the-art performance in semantic segmentation tasks, offering high accuracy and scalability.
- **Benefit**: The hierarchical design and efficient transformer backbone enable real-time segmentation with fewer computational resources compared to traditional CNN-based architectures.

---

# Section 6: Modelling
![alt text](image-3.png)


### **Explanation of Code:**

1. **Loading the Pretrained Model:**
   ```python
   model = TFSegformerForSemanticSegmentation.from_pretrained(
       model_id,
       num_labels=24,
       id2label=id2label,
       label2id=label2id,
       ignore_mismatched_sizes=True
   )
   ```
   This loads the SegFormer model (B5 variant fine-tuned on Cityscapes). The `id2label` and `label2id` are dictionaries that map class labels to integer IDs and vice versa.

2. **Preprocessing and Normalization:**
   The `preprocess_image` function reads, resizes, and normalizes the image and its corresponding segmentation mask using the pre-defined `MEAN` and `STD` values for normalization. This ensures that the input image is appropriately formatted before being passed to the model.

3. **Data Pipeline Setup:**
   The dataset is preprocessed and augmented using TensorFlow's `map` and `batch` methods. You can add shuffling and prefetching for better performance during training.

4. **Model Training:**
   The training loop uses a simple `train_step` function, which calculates the loss (using sparse categorical cross-entropy for segmentation), applies the gradients, and updates the model weights.

5. **Inference:**
   After training, you can use the trained model to perform segmentation on a new image. The result is the predicted segmentation mask.

---

### **Customizing for Your Dataset:**
- Adjust `num_labels` to the number of classes in your dataset.
- Define `id2label` and `label2id` dictionaries according to the labels in your dataset.
- Update the dataset paths and preprocessing logic for your specific dataset.
![alt text](image-4.png)


# Section 7: Conclusion and Future Work

## 12.1 Summary of Achievements
- Successful implementation of segmentation models.
- Significant improvement in prediction accuracy.

## 12.2 Future Enhancements
- Explore additional model architectures.
- Implement real-time segmentation.

---

# Final Thoughts

We hope this guide provides detailed insights into running and deploying the People Clothing Segmentation project. With its modular design, high performance, and flexibility, this project is well-suited for real-world applications, research, and continuous development.

Thank you for exploring the People Clothing Segmentation project. Feel free to contribute, raise issues, or suggest improvements.
