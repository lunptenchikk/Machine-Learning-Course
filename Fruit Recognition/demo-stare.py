import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


MODEL_PATH = "transfer_fruits_model.h5"

DEMO_DIR = "../test-multiple_fruits"

IMAGE_SIZE = (224, 224)
TOP_K = 3


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")


if hasattr(model, "class_names"):
    
    class_names = model.class_names
else:
    print("NIe ma class_names")
    
    
    
    tmp_ds = tf.keras.utils.image_dataset_from_directory(
        
        "../Training",
        image_size=IMAGE_SIZE,
        
        
        
        batch_size=1
    )
    
    
    class_names = tmp_ds.class_names

num_classes = len(class_names)


print(f"Liczba klas: {num_classes}")


def load_image(path): # funkcja do wczytywania i przetwarzania obrazu
    
    img = tf.keras.utils.load_img(path, target_size=IMAGE_SIZE)
    
    
    img = tf.keras.utils.img_to_array(img)
    
    
    
    img = np.expand_dims(img, axis=0)
    return img


image_files = [ f for f in os.listdir(DEMO_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]



if len(image_files) == 0:
    
    raise RuntimeError("No images found in test-multiple_fruits directory.")

for img_name in image_files:
    
    img_path = os.path.join(DEMO_DIR, img_name)
    
    
    img = load_image(img_path)

    preds = model.predict(img, verbose=0)[0]

    top_indices = np.argsort(preds)[-TOP_K:][::-1]


    print(f"Image: {img_name}")
    
    for i, idx in enumerate(top_indices, start=1):
        
        
        print(f"  Top {i}: {class_names[idx]} " f"({preds[idx]*100:.2f}%)"
        )

    plt.imshow(tf.keras.utils.load_img(img_path))
    
    
    
    
    plt.axis("off")

    title = " | ".join(
        
        
        f"{class_names[idx]} ({preds[idx]*100:.1f}%)"
        for idx in top_indices
    )
    plt.title(title)
    plt.show()

print("Finished.")
