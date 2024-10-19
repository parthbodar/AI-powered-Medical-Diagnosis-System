#train.py
import tensorflow as tf
import PIL

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# 1. Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True


)

val_datagen = ImageDataGenerator(rescale=1./255)



train_data = train_datagen.flow_from_directory(
    'D:/download/archive (1)/chest_xray/train', target_size=(224, 224), batch_size=32, class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    'D:/download/archive (1)/chest_xray/val', target_size=(224, 224), batch_size=32, class_mode='binary'
)

# 2. Build CNN Model using Transfer Learning (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the pre-trained layers

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train the Model
history = model.fit(
    train_data, validation_data=val_data, epochs=10, steps_per_epoch=len(train_data),
    validation_steps=len(val_data)
)

# 4. Save the Model
model.save('medical_diagnosis_model.h5')
print("Model saved!")