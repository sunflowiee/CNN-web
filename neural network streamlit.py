import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageGrab, ImageOps


model_path = "model_neural_network.h5"

if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model berhasil dimuat dari file.")
else:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Training model... (bisa beberapa menit tergantung PC)")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=[early_stop]
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\n Akurasi model di data test: {test_acc*100:.2f}%")

    model.save(model_path)
    print("Model tersimpan sebagai 'model_neural_network.h5'")

st.title("Tulis Angka")
canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Prediksi"):
    if canvas.image_data is not None:
        img = canvas.image_data

        # ambil satu channel aja (grayscale)
        img = img[:, :, 0]

        # resize ke 28x28
        img = Image.fromarray(img.astype("uint8")).resize((28, 28))

        # normalize
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)
        st.subheader(f"Prediksi: {np.argmax(pred)}")
