# 🐱🐶 Image Classification using Convolutional Neural Networks (CNN)

## 📌 Assignment Information

- **Assignment:** Assignment-9
- **Topic:** Image Classification using Convolutional Neural Networks (CNN)
- **Course:** AI-ML
- **Model Used:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras

---

# 📖 Objective

The objective of this project is to develop a Convolutional Neural Network (CNN) model capable of automatically classifying images into two categories: **Cats** and **Dogs**. The project demonstrates the complete deep learning workflow, including data preprocessing, CNN model development, model training, evaluation using multiple performance metrics, and visualization of training results.

---

# 📂 Dataset

**Dataset Name:** Dogs and Cats Image Classification Dataset

**Kaggle Dataset Link:**

https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

> **Note:** The dataset is not included in this repository, as instructed in the assignment. Please download it from Kaggle using the above link.

---

# 🛠 Libraries Used

The following Python libraries were used in this project:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- OS
- Random

Install all dependencies using:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

# 📁 Project Structure

```
Assignment-9/
│
├── Assignment-9.ipynb
├── README.md
├── images/
│   ├── sample_images.png
│   ├── confusion_matrix.png
│   ├── accuracy_graph.png
│   └── loss_graph.png
└── requirements.txt (Optional)
```

---

# 📊 Dataset Overview

The dataset contains two image categories:

- Cat
- Dog

Each image is processed before training by:

- Resizing images to **128 × 128 pixels**
- Normalizing pixel values between **0 and 1**
- Splitting data into:
  - **80% Training**
  - **20% Testing**

---

# ⚙ Data Preprocessing

The preprocessing stage includes:

- Loading the dataset
- Displaying the dataset folder structure
- Displaying five sample images with class labels
- Counting total images
- Identifying the number of classes
- Reading original image dimensions
- Resizing all images to 128 × 128
- Pixel normalization
- Creating TensorFlow/Keras data generators

---

# 🧠 CNN Architecture

The CNN model consists of the following layers:

```
Input Image
(128 × 128 × 3)

        │

Conv2D
32 Filters
3 × 3 Kernel
ReLU

        │

MaxPooling2D
2 × 2

        │

Conv2D
64 Filters
3 × 3 Kernel
ReLU

        │

MaxPooling2D
2 × 2

        │

Conv2D
128 Filters
3 × 3 Kernel
ReLU

        │

MaxPooling2D
2 × 2

        │

Flatten Layer

        │

Dense Layer
128 Neurons
ReLU

        │

Output Layer
1 Neuron
Sigmoid
```

---

# ⚙ Model Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Activation Function | ReLU |
| Output Activation | Sigmoid |
| Batch Size | 32 |
| Epochs | 10 |
| Image Size | 128 × 128 |

---

# 🚀 Methodology

The complete workflow of this project is:

### Step 1

Load the Cats vs Dogs dataset.

### Step 2

Explore the dataset and display sample images.

### Step 3

Preprocess the images by resizing and normalization.

### Step 4

Split the dataset into training and testing sets.

### Step 5

Create image generators using TensorFlow/Keras.

### Step 6

Build the CNN model using:

- Three Convolution Layers
- Three MaxPooling Layers
- Flatten Layer
- Dense Layer
- Output Layer

### Step 7

Compile the model using:

- Adam Optimizer
- Binary Crossentropy Loss
- Accuracy Metric

### Step 8

Train the model for 10 epochs.

### Step 9

Evaluate the trained model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Step 10

Visualize:

- Accuracy vs Epoch
- Loss vs Epoch

---

# 📈 Results

The CNN model was evaluated using multiple performance metrics.

The evaluation includes:

- Test Accuracy
- Precision
- Recall
- F1 Score

The project also generates:

- Confusion Matrix
- Accuracy vs Epoch graph
- Loss vs Epoch graph

These visualizations help analyze the learning behavior of the CNN during training.

---

# 📊 Performance Metrics

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score

These metrics provide a complete understanding of the model's classification performance.

---

# 📉 Visualizations

The notebook generates:

- Five sample images from the dataset
- Confusion Matrix
- Accuracy vs Epoch graph
- Loss vs Epoch graph

---

# 🔍 Observations

1. The CNN successfully learns meaningful image features from cats and dogs.
2. Training accuracy increases steadily over the training epochs.
3. Validation accuracy follows a similar trend, indicating effective learning.
4. The confusion matrix shows that most images are correctly classified with only a small number of misclassifications.
5. Precision, Recall, and F1 Score demonstrate balanced performance across both classes.

---

# ✅ Advantages of CNN

- Automatically extracts important image features.
- Better performance than traditional Artificial Neural Networks (ANNs) for image data.
- Learns spatial information effectively.
- Reduces manual feature engineering.
- Provides high classification accuracy for image recognition tasks.

---

# ⚠ Limitations

- Requires a relatively large dataset.
- Training can be computationally expensive.
- Performance depends on hardware resources.
- May overfit if not properly regularized.

---

# 📚 Conclusion

This project successfully implemented a Convolutional Neural Network (CNN) for binary image classification of cats and dogs. Images were resized, normalized, and used to train a deep learning model consisting of convolutional, pooling, flatten, and dense layers. The model demonstrated good classification performance and was evaluated using Accuracy, Precision, Recall, F1 Score, and a Confusion Matrix. Training and validation graphs showed the learning progress over multiple epochs. Compared with a traditional Artificial Neural Network (ANN), CNN automatically extracts spatial features from images, making it more suitable for image classification tasks. However, CNNs require more computational power and larger datasets for effective training.

---

# 💻 How to Run

1. Download the dataset from Kaggle.
2. Extract the dataset.
3. Update the dataset path inside the notebook.
4. Install required libraries.
5. Run all notebook cells sequentially.
6. Observe the evaluation metrics and generated graphs.



**Topic:** Image Classification using CNN

**Framework:** TensorFlow/Keras
