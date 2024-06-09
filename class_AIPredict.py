import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class AIPredict:
    def __init__(self, model_path, class_names, img_size=256):
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.IMG_SIZE = img_size


    def get_frame(self, video, timestamp):
        video.set(cv2.CAP_PROP_POS_MSEC, round(timestamp, 1) * 1000)
        frame = None

        # Check if video is opened successfully
        if not video.isOpened():
            print("Could not open video")
            return None

        while True:
            current_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds

            if current_timestamp < timestamp:
                ret, frame = video.read()
                if not ret:
                    break
            else:
                break

        return frame


    def predict_image(self, equipment, timestamp):
        # Get the frame at given timestamp
        img = self.get_frame(equipment, timestamp)
        if img is None:
            print("No frame at given timestamp")
            return None, None

        # Crop the image
        #img = img[620:2050, 80:1650]  # y1:y2, x1:x2
        img = img[355:1450, 47:1290]  # y1:y2, x1:x2


        # Resize the frame to the target size
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

        # Display the frame
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
        #plt.title(f'Timestamp: {timestamp} seconds')
        #plt.show()

        # Convert the image to a numpy array and normalize pixel values
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0

        # Predict class probabilities for input image
        predictions = self.model.predict(img_array)

        # Get predicted class label by finding the index of the maximum predicted probability
        predicted_label_index = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_label_index]
        predicted_probability = predictions[0][predicted_label_index]

        return predicted_class, predicted_probability
