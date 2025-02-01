import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# Modeli yükle
model_path = "/Users/haticeaksu/Desktop/mask_detector_model.h5"
model = tf.keras.models.load_model(model_path)

# Veri seti dizinleri
data_dir = "/Users/haticeaksu/Desktop/data"
mask_dir = os.path.join(data_dir, "with_mask")
no_mask_dir = os.path.join(data_dir, "without_mask")

# Rastgele test görüntülerini seç
mask_images = [os.path.join(mask_dir, img) for img in os.listdir(mask_dir) if img.endswith((".jpg", ".png"))]
no_mask_images = [os.path.join(no_mask_dir, img) for img in os.listdir(no_mask_dir) if img.endswith((".jpg", ".png"))]

# Her iki sınıftan 3'er adet rastgele seç (varsa)
test_images = random.sample(mask_images, min(3, len(mask_images))) + random.sample(no_mask_images, min(3, len(no_mask_images)))

# Görüntü işleme fonksiyonu
def load_and_preprocess_image(image_path, target_size=(224, 224)):  # Model giriş boyutuna göre ayarla
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalizasyon (model böyle eğitildiyse)
    return np.expand_dims(image, axis=0)

# Görüntüleri yükleyip tahmin yap
plt.figure(figsize=(10, 5))
for i, img_path in enumerate(test_images):
    image = load_and_preprocess_image(img_path)
    prediction = model.predict(image)[0][0]  # Modelin çıktısını al
    
    label = "Maskesiz" if prediction > 0.5 else "Maskeli"
    
    # Görüntüyü göster
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.imread(img_path)[:, :, ::-1])  # BGR -> RGB dönüşümü
    plt.title(f"Model Tahmini: {label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
