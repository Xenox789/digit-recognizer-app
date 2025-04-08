# 🧠 Handwritten Digit Recognizer Web App (CNN-Powered)

This is a CNN-based handwritten digit recognition app built with **Flask**, **TensorFlow**, and **HTML5 Canvas**.  
Users can draw a digit on a virtual whiteboard, and the app will predict what number they wrote using a Convolutional Neural Network trained on the MNIST dataset.

---

## 🚀 Live Demo

📍 Coming soon – hosted on Render or another platform.

---

## 🖼️ Preview

✍️ Draw a digit (0–9), click "Predict", and receive an instant result powered by a trained neural network!

---

## 🔧 Tech Stack

- 🐍 Python 3.11
- 🧠 TensorFlow + Keras (with a CNN model)
- 🔥 Flask (API backend)
- 🧼 Pillow (for image preprocessing)
- ✨ HTML5 Canvas (for drawing input)

---

## 🧠 Model Info

The app uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset for better spatial understanding of drawn digits.

Model Architecture (example):

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
