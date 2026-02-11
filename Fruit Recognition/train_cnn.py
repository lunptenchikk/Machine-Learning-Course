# Diemid Rybchenko
import os
import tensorflow as tf


import keras
from keras import layers, models, callbacks, optimizers
from keras.utils import image_dataset_from_directory


TRAIN_DIR = "C:\\Users\\annar\\Downloads\\Training"
TEST_DIR = "C:\\Users\\annar\\Downloads\\Test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32   # def. parametrow
EPOCHS = 30
VALIDATION_SPLIT = 0.2
SEED = 42


print("Loading naszych datasetow")

train_ds = image_dataset_from_directory(
    
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    
    seed=SEED,
    image_size=IMAGE_SIZE,
    
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    
    
    seed=SEED,
    image_size=IMAGE_SIZE,
    
    
    batch_size=BATCH_SIZE
)

print("Loading test dataset")


test_ds = image_dataset_from_directory(
    TEST_DIR,
    
    image_size=IMAGE_SIZE,
    
    batch_size=BATCH_SIZE,
    
    
    
    shuffle=False
)

class_names = train_ds.class_names

num_classes = len(class_names)

#print(f"Number of classes: {num_classes}")

# Optymizacja wydajnosci
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)



test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Augmentacja danych

data_augmentation = keras.Sequential(
    
    
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        
        layers.RandomContrast(0.1),
        
    ],
    name="data_augmentation"
)

# Definicja modelu CNN

model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 3)),

    data_augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    
    
    
    
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    
    
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

model.summary()

model.compile( optimizer=optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])




# Warunki zatrzymania wczesnego i zmniejszania lr
early_stopping = callbacks.EarlyStopping(
    
    monitor="val_loss",
    
    patience=5,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    
    
    monitor="val_loss",
    
    factor=0.3,
    patience=3,
    
    min_lr=1e-6
)

#Trening

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Robimy ewaluacje


test_loss, test_acc = model.evaluate(test_ds)

print(f"\nTest accuracy: {test_acc:.4f}")

MODEL_PATH = "cnn_fruits_model.h5"


model.save(MODEL_PATH)


print(f"Model saved to {MODEL_PATH}")
