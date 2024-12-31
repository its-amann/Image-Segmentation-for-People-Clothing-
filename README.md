![image](https://github.com/user-attachments/assets/11f3afc6-ab65-4329-8969-d9ba7fa66f40)


<h1 align="center">🎓 Clothing Segmentation and Image Generation 🎓</h1>

<p align="center">
<strong>A project focused on clothing segmentation using SegFormer and augmented image generation with Stable Diffusion.</strong>
</p>

<p align="center">
<a href="https://github.com/your-username/your-repo">
<img src="https://img.shields.io/github/license/its-amann/Image-Segmentation-for-People-Clothing-.svg" alt="License">
</a>
<a href="https://github.com/your-username/your-repo/issues">
<img src="https://img.shields.io/github/issues/its-amann/Image-Segmentation-for-People-Clothing-.svg" alt="Issues">
</a>
<a href="https://github.com/your-username/your-repo/stargazers">
<img src="https://img.shields.io/github/stars/its-amann/Image-Segmentation-for-People-Clothing-.svg" alt="Stars">
</a>
</p>

🚀 Overview

Welcome to the Clothing Segmentation and Image Generation project, a cutting-edge and multifaceted system combining deep learning techniques for image analysis and synthesis. This project utilizes state-of-the-art models like SegFormer for clothing segmentation and Stable Diffusion for generating new, augmented images based on segmentation masks. It is designed to demonstrate a robust workflow for both understanding and modifying visual content.

✨ Wow Factors:

Advanced Segmentation with SegFormer: Leverages the SegFormer model for precise pixel-level segmentation of clothing items.

Augmented Image Generation via Stable Diffusion: Utilizes Stable Diffusion to generate photorealistic images, enhancing datasets and creative possibilities.

Integration with FiftyOne: Employs FiftyOne for comprehensive dataset management, visualization, and evaluation, making the workflow seamless and efficient.

Customizable Pipelines: Offers flexible pipelines for processing and augmenting image datasets, adapting to specific project needs.

![image](https://github.com/user-attachments/assets/f509044f-d9ef-41da-9f21-41ae54b12a32)

🛠 Features

Dataset Preparation: Automated data loading, preprocessing, and splitting for both training and validation sets.

Augmented Dataset Generation: Utilizes albumentations for image and mask augmentation, enhancing model training.

Clothing Segmentation: Employs a pre-trained SegFormer model to predict pixel-wise clothing segmentations.

Stable Diffusion Integration: Generates new images by masking the clothing segmentations and using stable diffusion for inpainting.

FiftyOne Integration: Manages, visualizes, and evaluates the dataset using FiftyOne, including calculation of mean IoU and other evaluation metrics.

Customizable Model Training: Allows for the training and fine-tuning of SegFormer models, adaptable to specific dataset characteristics.

Image and Mask Generation: Includes a process to extract and generate masks from segmentation maps.

Evaluation Metrics: Computes per-class and overall evaluation metrics using FiftyOne and evaluate library.

📸 Screenshots
<![image](https://github.com/user-attachments/assets/386f3e08-a3ee-4d5c-8ef3-b30a9095a869)

<div align="center">
<font size="5">**Segmentation Results with predicted Masks on the Image**</font>
</div>

![image](https://github.com/user-attachments/assets/6b239447-bd44-46ca-a9aa-d3522b694dce)

<div align="center">
<font size="5">**Augmented Image Generation via Stable Diffusion after Segmentation**</font>
</div>

## 🔧 Installation

### Prerequisites

- **Python 3.10+** installed on your machine. [Download Python](https://www.python.org/downloads/)
- **CUDA enabled GPU** for efficient model training and inference.
- **PyTorch** with CUDA support. Install following [PyTorch instructions](https://pytorch.org/get-started/locally/).
- **Transformers, Diffusers, and other required libraries**. Install using pip:

```bash
pip install -q datasets evaluate
pip install fiftyone==0.23.0rc1
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q regex tqdm
pip install -q diffusers transformers accelerate scipy
pip install -q -U xformers
```

### Steps

1. **Clone the Repository:**

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
   (You may need to create a requirements.txt file listing all used libraries)

3. **Download Dataset:**
   You will need to download the "People Clothing Segmentation" dataset from Kaggle. Ensure you have the Kaggle CLI installed and properly configured, then use the following command:
```bash
pip install -q kaggle
mkdir ~/.kaggle
#place your kaggle.json file inside ~/.kaggle directory
cp kaggle.json ~/.kaggle/
chmod 600 /root/.kaggle/kaggle.json
kaggle datasets download -d rajkumarl/people-clothing-segmentation
```
  Unzip the downloaded dataset:
```bash
unzip "/content/people-clothing-segmentation.zip" -d "/content/dataset/"
```

4.  **Run the Notebook**
    Open the downloaded notebook (.ipynb) file with an environment with all dependencies and run through cells sequentially.

---

## 💻 Usage

This project is designed to be run within a Jupyter Notebook environment, making it easy to step through each stage of the workflow. Here are the general steps and key operations:

### Initial Setup:

1.  **Environment Setup:** Start by installing the required libraries in your environment to ensure that there is no dependency conflict.

2. **Dataset Download and Configuration:** Download the required dataset from Kaggle using the CLI and then unpack it to the specified directory.

3. **Data Splitting and Preprocessing:** The notebook automatically handles splitting data into training and validation sets and applying necessary transformations such as resize, normalization, and data augmentation.

### Segmentation Model Training:

1. **Model Loading:** Load the pre-trained SegFormer model, modifying the classification head to match the number of classes in the dataset.
2. **Training and Evaluation:** Train the model with the prepared training dataset, and evaluate the model after every epoch with the evaluation dataset using mean IoU, overall accuracy etc.
3. **Visualization:** Use the visual results shown in the notebook to assess your model performance.

### Image Augmentation with Stable Diffusion:

1.  **Stable Diffusion Pipeline:** Create an instance of the Stable Diffusion pipeline for inpainting.
2. **Mask Generation:** Segmented masks are generated, and those are used as input to the pipeline along with the original image.
3.  **Augmented Image Generation:** Apply inpainting to generate modified versions of the original images.

### FiftyOne Integration:

1. **Dataset Loading:** Create a FiftyOne dataset from the validation images and masks.
2. **Evaluation Metrics:** Compute per-class and overall evaluation metrics with FiftyOne.
3. **Data Vizualisation:** Visualize all samples, masks and the predicted masks using FiftyOne.

### Detailed Steps

1.  **Data Download and Unpacking**:

    ```python
    !pip install -q kaggle
    !mkdir ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json
    !kaggle datasets download -d rajkumarl/people-clothing-segmentation
    !unzip "/content/people-clothing-segmentation.zip" -d "/content/dataset/"
    ```
2.  **Data Preprocessing**

    ```python
    H,W = 512,512
    BATCH_SIZE = 2
    N_CLASSES = 24
    LR = 5e-5
    N_EPOCHS = 20
    WEIGHT_DECAY_RATE = 0.01
    MEAN = [123.675, 116.28, 103.53]
    STD = [58.395, 57.12, 57.375]
    checkpoint_filepath = "/content/drive/MyDrive/fiftyone/segformer_b5_clothing.h5"

    ```
    This code cell sets up the core variables required for the training process. This includes image dimensions (H, W), batch size, learning rate, epochs, mean and standard deviations for normalization and model checkpoint path.

3. **Loading Data**
    ```python
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ([im_path+i for i in os.listdir(im_path)],
        [anno_path+"img"+i[3:] for i in os.listdir(im_path)])
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ([val_im_path+i for i in os.listdir(val_im_path)],
        [val_anno_path+"img"+i[3:] for i in os.listdir(val_im_path)])
    )
    ```
     The above code loads images and masks file paths to dataset objects for training and evaluation.

4. **Data Transformation**

  ```python
    def preprocess(im_path, anno_path):
      img = tf.io.decode_jpeg(tf.io.read_file(im_path))
      img = tf.cast(img,tf.float32)
      img = (img-MEAN)/STD

      anno = tf.io.decode_jpeg(tf.io.read_file(anno_path))
      anno = tf.cast(tf.squeeze(anno,-1),tf.float32)

      return img, anno
  ```

  This code defines a preprocessing function `preprocess` that decodes image and mask files, casts them to floating-point tensors, normalizes the images, and prepares them for model input.

  ```python
  transform = A.Compose([
      A.RandomCrop (H,W, p=1.0),
      A.HorizontalFlip(p=0.3),
      A.VerticalFlip(p=0.3),
      A.RandomRotate90(p=0.3),
      A.Transpose(p=0.3),
      A.Sharpen (alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.1),
      A.RandomShadow (shadow_roi=(0, 0.5, 1, 1),
                      num_shadows_lower=1, num_shadows_upper=2,
                      shadow_dimension=5, p=0.1),
      A.RandomBrightnessContrast(p=0.2),
      #A.Resize(H,W),
  ])

  val_transform = A.Compose([
      A.Resize(H,W),
  ])

  ```
  The above defines the augmentation policy for training and testing using albumentations library.

  ```python
  train_ds = (
    prep_train_ds
    .shuffle(10)
    .map(augment,num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
  )
  val_ds = (
    prep_val_ds
    .map(val_augment,num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
  )
  ```
  This creates final training and testing dataset with all augmentation and batching techniques.

5. **Modeling**
    ```python
      model_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
      model = TFSegformerForSemanticSegmentation.from_pretrained(
      model_id,
      num_labels = len(label2id),
      id2label = id2label,
      label2id = label2id,
      ignore_mismatched_sizes = True)
    ```
    This code will load the required pre-trained SegFormer model and initialize with the desired number of classes.

6.  **Training and Evaluation**:
    ```python
        model.compile(optimizer=optimizer)
        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=callbacks,
    )
    ```
    This will start the model training loop with the given training and validation dataset.

7.  **Stable Diffusion Augmentation**:
    ```python
    prompt = "A photorealistic photo of a woman wearing a green-colored nice looking coat all green high resolution"
    image, mask_image = generate_inputs(
       "/content/val_dataset/png_images/IMAGES/img_0003.png","coat")
    image = pipe(prompt=prompt, image=image, mask_image=mask_image, ).images[0]
    ```
    This part loads the Stable Diffusion model and runs the augmentations on the generated masks. The generate input function will get the required images and masks for running the stable diffusion pipeline.

    ```python
     def augpaint(pipe, prompt, pil_image, pil_mask, guidance_scale, num_inference_steps):

        num_images_per_prompt = 1
        generator = torch.Generator(device="cuda").manual_seed(10)

        encoded_images = []

        for i in range(num_images_per_prompt):
            image = pipe(prompt=prompt, guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps, generator=generator,
                            image=pil_image, mask_image=pil_mask, strength=0.99).images[0]

            encoded_images.append(image.resize((550,825)))
        return encoded_images[0]
    ```
    This function will run the stable diffusion inpainting pipeline using the generated mask and image using the given text prompt.

    ```python
      def transform_sample(sample, select_class, prompt):
        hash = create_hash()
        filename = sample.filepath.split("/")[-1][:-4]+"_"+str(hash)+".png"
        pipe = pipeline
        im,mask = generate_inputs(
        sample.filepath, sample.ground_truth.mask_path,
         label2id[select_class])
    
        out = augpaint(pipe, prompt, im, mask, guidance_scale, num_inference_steps)
        im_saved = out.save(sample.filepath[:-4]+"_"+str(hash)+".png")
        shutil.copy(sample.ground_truth.mask_path,
                    sample.ground_truth.mask_path[:-4]+"_"+str(hash)+".png",
                    )
        display(out)
        new_sample = fo.Sample(
            filepath=sample.filepath[:-4]+"_"+str(hash)+".png",
            ground_truth=fo.Segmentation(
                mask_path=sample.ground_truth.mask_path[:-4]+"_"+str(hash)+".png"),
        )
        return new_sample
    ```
    This function will tie all together, it gets the images and the required masks, and then runs the stable diffusion inpainting pipeline using the prompt. Finally, it creates a new sample with the newly generated image and mask.

8.  **FiftyOne Evaluation**:
    ```python
      session = fo.launch_app(dataset,port=51)
    ```
    This code launches the FiftyOne application to visualize the generated segmentation and augmented images.

---

## 🤝 Contributing

1.  **Fork the Project**
2.  **Create your Feature Branch:** `git checkout -b feature/AmazingFeature`
3.  **Commit your Changes:** `git commit -m 'Add some AmazingFeature'`
4.  **Push to the Branch:** `git push origin feature/AmazingFeature`
5.  **Open a Pull Request**

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments
This project has been made possible with these incredible tools:
- Tensorflow and Keras
- HuggingFace Transformers and Diffusers
- FiftyOne for Data Visualization and Evaluation
- Albumentations for Data Augmentation
- OpenCV for Image processing
- PyTorch
- Many other open source contributors that have helped in their journey.

<p align="center">
  Made with ❤️ by Your Name
</p>

