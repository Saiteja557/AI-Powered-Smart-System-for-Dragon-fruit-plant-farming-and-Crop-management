# TRAINING CNN MODEL FOR DRAGON FRUIT DISEASE DETECTION

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

train_dir = "datasets/disease/"  

img_width, img_height = 128, 128
batch_size = 16


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(4, activation='softmax')   # 4 classes
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


history = model.fit(
    train_generator,
    epochs=12,
    validation_data=val_generator,
    callbacks=[early_stop]
)


model.save("models/disease_model_fast.h5")

print("✅ Training Accuracy:", round(history.history['accuracy'][-1]*100, 2), "%")
print("✅ Validation Accuracy:", round(history.history['val_accuracy'][-1]*100, 2), "%")
print("🎉 Model saved successfully!")
