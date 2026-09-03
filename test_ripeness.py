import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/ripeness_model.keras")

img_path = "test_images/ripeness_test.jpg"   # Put ripe/unripe image here

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

if prediction >= 0.5:
    print(f"Ripeness Status: RIPE (Confidence: {confidence:.2f}%)")
else:
    print(f"Ripeness Status: UNRIPE (Confidence: {confidence:.2f}%)")
