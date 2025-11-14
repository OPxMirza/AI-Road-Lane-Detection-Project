
# 🚗 AI Road Lane Detection Project

This project focuses on **road lane detection using semantic segmentation** powered by **Deep Learning**.  
It uses TensorFlow/Keras to train a model on road images, generates lane masks, and exports trained models in `.h5` and `.tflite` formats for deployment on mobile/embedded systems.

---

## 📌 Features

- Automatic lane mask conversion (magenta → binary mask)
- Image preprocessing (resizing, normalization)
- Train/validation split
- Custom CNN-based segmentation model
- Visualization of predictions
- Export model to:
  - `lane_detection.h5`
  - `model.tflite`
- Ready for Android, Raspberry Pi, and edge devices

---


## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **OpenCV**
- **Pillow (PIL)**
- **matplotlib**
- **scikit-learn**

---

## 🧠 How the Project Works

### **1. Load and Prepare Data**
- Training images and masks are loaded.
- All images are resized to **224 × 224**.
- Masks in magenta color `[255, 0, 255]` are converted to **binary lane masks**.

### **2. Train/Validation Split**
Dataset split:

```python
train_test_split(..., test_size=0.2, random_state=42)
````

### **3. Build the Model**

A CNN-based segmentation model using TensorFlow/Keras.

### **4. Train the Model**

Training runs for **5 epochs** with validation monitoring.

### **5. Predict Lane Masks**

Model predicts lane segmentation on test images.

### **6. Export Models**

The final trained model is saved in two formats:

* **lane_detection.h5** → Standard Keras model
* **model.tflite** → Optimized TensorFlow Lite model for deployment

---

## ▶️ How to Run the Project

### **1. Install Dependencies**

```bash
pip install tensorflow numpy pillow matplotlib scikit-learn opencv-python
```

### **2. Prepare Dataset**

Make sure your folders follow this structure:

```
data_road_224/
    training/image_2/
    training/gt_image_2/
    testing/image_2/
```

### **3. Run the Notebook**

Open in Jupyter or VS Code:

```
ai_project.ipynb
```

### **4. Outputs Generated**

After training, these files will be created:

* `lane_detection.h5` — main model
* `model.tflite` — optimized lightweight model

---


## 👤 Author

**Mirza Abdullah Akmal (OPxMirza)**
AI / ML & Full-Stack Developer
GitHub: [https://github.com/OPxMirza](https://github.com/OPxMirza)

