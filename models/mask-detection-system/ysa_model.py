import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, MobileNet
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

# Veri yolu
data_dir = "/Users/haticeaksu/Desktop/data"
mask_dir = os.path.join(data_dir, "with_mask")
no_mask_dir = os.path.join(data_dir, "without_mask")

# Eğitim ve test setlerini oluşturuyoruz
all_images = []
labels = []

for img in os.listdir(mask_dir):
    all_images.append(os.path.join(mask_dir, img))
    labels.append("mask")  # Maskeli

for img in os.listdir(no_mask_dir):
    all_images.append(os.path.join(no_mask_dir, img))
    labels.append("no_mask")  # Maskesiz

# Veriyi eğitim (%80) ve test (%20) olarak ayırıyoruz
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, labels, test_size=0.2, random_state=42
)

# Eğitim ve validasyon setlerini ayırıyoruz (Eğitim %80, Validasyon %20)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Görüntü işleme için ImageDataGenerator kullanıyoruz
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_images, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_images, 'class': val_labels}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = ImageDataGenerator(rescale=1.0/255).flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_images, 'class': test_labels}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# DenseNet121 Modeli oluşturuyoruz
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Modeli dondurmak ve bazı katmanları serbest bırakmak (Önceden eğitilmiş ağırlıkları koruyarak, transfer learning (aktarılmış öğrenme) yöntemini kullanmak için yapıyoruz)
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleme
model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks (erken durdurma ve öğrenme oranı azaltma)
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Modeli eğitiyoruz
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Modeli kaydediyoruz
model_save_path = os.path.join(os.getcwd(), "mask_detector_model.h5")
model.save(model_save_path)
print(f"Model kaydedildi: {model_save_path}")

# Test verisi ile tahmin yapıyoruz
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

# confusion_matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Mask', 'Mask'], yticklabels=['No Mask', 'Mask'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# classification report
print(classification_report(y_true, y_pred_classes))

# ROC Eğrisi
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
