import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("models/disease_model_fast.h5")

# Image path
img_path = "test_images/test.jpg"

# Load & resize image (same size as training)
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

# Class labels (must match training folder order)
class_labels = ['BadFruit', 'BadLeaf', 'GoodFruit', 'GoodLeaf']
predicted_class = class_labels[class_index]

# Convert to your required output
if predicted_class in ['GoodFruit', 'GoodLeaf']:
    final_output = "Healthy"
else:
    final_output = "Diseased"

print("Detected:", predicted_class)
print("Final Result:", final_output)
