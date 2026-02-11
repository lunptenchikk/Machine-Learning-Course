
import tensorflow as tf
import keras


from keras import layers, models, callbacks, optimizers
from keras.utils import image_dataset_from_directory

TRAIN_DIR = "C:\\Users\\annar\\Downloads\\Training"
TEST_DIR = "C:\\Users\\annar\\Downloads\\Test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_TRANSFER = 15
EPOCHS_FINE = 10
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

print(f"Number of classes: {num_classes}")



AUTOTUNE = tf.data.AUTOTUNE


train_ds = train_ds.prefetch(AUTOTUNE)


val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)



data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation"
)

# Wybieramy EfficientNetB0 jako backbone

base_model = keras.applications.EfficientNetB0(
    
    
    include_top=False,
    weights="imagenet",
    
    
    input_shape=(*IMAGE_SIZE, 3)
    
)

base_model.trainable = False  # zamrazamy warstwy bazowe w celu transfer learningu

# Budowa modelu z transfer learningiem
inputs = layers.Input(shape=(*IMAGE_SIZE, 3))


x = data_augmentation(inputs)

x = keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)


x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="relu")(x)




x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()





 # Kompilacja modelu
model.compile(
    
    optimizer=optimizers.Adam(learning_rate=1e-3),
    
    loss="sparse_categorical_crossentropy",
    
    
    metrics=["accuracy"]
)


early_stopping = callbacks.EarlyStopping(
    
    monitor="val_loss",
    patience=5, restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6
)


# Sam trening modelu

print("\nPierwszy etap: Transfer Learning")

history_transfer = model.fit(
    
    
    train_ds,
    validation_data=val_ds,
    
    
    epochs=EPOCHS_TRANSFER,
    callbacks=[early_stopping, reduce_lr]
)




print("\nDrugi etap: Fine-Tuning")      

# Robimy unfreeze czesci warstw w celu fine-tuningu
base_model.trainable = True

FINE_TUNE_AT = int(len(base_model.layers) * 0.75) 




for layer in base_model.layers[:FINE_TUNE_AT]:
    
    
    
    layer.trainable = False

print(f"Fine-tuning z warstwy {FINE_TUNE_AT} / {len(base_model.layers)}")

model.compile(
    
    
    optimizer=optimizers.Adam(learning_rate=1e-5),
    
    
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    
    
    
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    
    
    callbacks=[early_stopping, reduce_lr]
)


# Ewaluacja modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(test_ds)


print(f"\nUzyskana accuracy: {test_acc:.4f}")


MODEL_PATH = "transfer_fruits_model.h5"
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
