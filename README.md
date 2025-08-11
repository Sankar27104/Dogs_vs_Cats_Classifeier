# 🐶🐱 Dogs vs Cats Image Classification (CNN)

A deep learning-based image classifier built with **TensorFlow/Keras** that can distinguish between **dog** and **cat** images.  
This implementation is designed for **Google Colab** and downloads the dataset directly from **Kaggle**.

---

## 📌 Features
- Downloads and extracts dataset from Kaggle automatically.
- Uses **image_dataset_from_directory** for efficient loading.
- Includes image normalization for better training performance.
- CNN architecture with **Conv2D**, **BatchNormalization**, **MaxPooling2D**, **Dropout**.
- Model training visualization with accuracy/loss graphs.
- Predicts new images (dog or cat) with confidence scores.

---

## 📂 Dataset
The dataset used is from Kaggle:  
**[Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)**



---

## ⚙️ Installation & Setup (Google Colab)
### **1. Upload Kaggle API Key**
- Go to your [Kaggle Account Settings](https://www.kaggle.com/account)
- Click **Create New API Token** → Downloads `kaggle.json`
- In Colab, upload `kaggle.json` to your working directory.

### **2. Download & Extract Dataset**
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
```
```
!kaggle datasets download -d salader/dogs-vs-cats
```
```
import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

## Model Training
```
import tensorflow as tf
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# Load training & test datasets
train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/dogs-vs-cats/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)
test_ds = keras.utils.image_dataset_from_directory(
    directory='/content/dogs-vs-cats/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)

# Normalize images
def process(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=test_ds)
```
## Training Results
```
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], color='red', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.legend()
plt.show()
```
## 🔍 Prediction
```
import cv2

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 3)
    prediction = model.predict(img)
    label = "Dog" if prediction[0][0] > 0.5 else "Cat"
    print(f"{label} ({prediction[0][0]:.4f})")

predict_image('/content/Dog1.jpg')
predict_image('/content/cat.jpg')
```
---
## 🛠 Requirements
1. Python 3.7+
2. TensorFlow
3. Keras
4. OpenCV
5. Matplotlib
6. Kaggle CLI

---
## 🤝 Contributing
This project is open-source — feel free to fork and improve it.
