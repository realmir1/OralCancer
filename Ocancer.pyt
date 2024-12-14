import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


train_dir = '/kaggle/input/multi-cancer/Multi Cancer/Multi Cancer/Oral Cancer'


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,  # %10 test, %90 eğitim
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resim boyutunu sabitle
    batch_size=32,
    class_mode='binary',
    subset='training'
)

test_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # İkili sınıflandırma
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_data, validation_data=test_data, epochs=10)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Eğitim ve Doğrulama Doğruluğu')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Eğitim ve Doğrulama Kaybı')
plt.show()


import numpy as np

sample_images, sample_labels = next(test_data)
predictions = model.predict(sample_images[:4])

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for img, pred, label, ax in zip(sample_images[:4], predictions, sample_labels[:4], axes):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Tahmin: {"Oral SCC" if pred > 0.5 else "Oral Normal"}\nGerçek: {"Oral SCC" if label > 0.5 else "Oral Normal"}')
plt.show()
