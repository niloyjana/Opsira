"""
Opsira AI — Problem 3: Training Pipeline
Implements EfficientNetB0 2-Phase Training
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Configuration
DATASET_DIR = "dataset"
MODEL_DIR   = "model"
TARGET_SIZE = (224, 224)
BATCH_SIZE  = 32

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

# 1. Data Preparation (ImageDataGenerator)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Handle Class Imbalance
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))

# 2. Build Model (EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# PHASE 1: Train Head Only
print("\n>>> Phase 1: Training Head (Base Model Frozen)")
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_name', metrics=['accuracy'])
# Note: categorization is inferred as categorical_crossentropy
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_p1 = [
    ModelCheckpoint(os.path.join(MODEL_DIR, "opsira_model.h5"), save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3)
]

history_p1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks_p1
)

# PHASE 2: Fine-Tuning
print("\n>>> Phase 2: Fine-Tuning (Unfreezing Top 30 Layers)")
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_p2 = [
    ModelCheckpoint(os.path.join(MODEL_DIR, "opsira_model.h5"), save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=4)
]

history_p2 = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks_p2
)

# 3. Plotting Curves
def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
    print(f"\n>>> Training curves saved to {MODEL_DIR}/training_curves.png")

plot_history(history_p1, history_p2)
print(f"\n>>> Final model saved to {MODEL_DIR}/opsira_model.h5")
