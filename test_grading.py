import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained grading model
model = load_model("models/grading_model.keras")

# Class labels (ORDER MUST MATCH TRAINING)
class_labels = ['Grade_A', 'Grade_B']

# Image path
img_path = "test_images/grade_test.jpg"   # put image here

# Load & preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_index = np.argmax(pred)
confidence = pred[0][class_index] * 100

print(f"Fruit Quality: {class_labels[class_index]} (Confidence: {confidence:.2f}%)")
