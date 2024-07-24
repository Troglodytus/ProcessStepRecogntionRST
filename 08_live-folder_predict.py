import time
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os

# Flag to control the flow of the image processing
proceed = False

def on_key(event):
    global proceed
    if event.key == ' ':
        proceed = True

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

model_path = config['Paths']['model_path']
folder_path = config['Paths']['image_predict_path']

# Check if the folder path exists
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"The specified folder path does not exist: {folder_path}")

# Load pre-trained model
model = tf.keras.models.load_model(model_path, compile=True, safe_mode=True)

# Define image size and class labels
IMG_SIZE = 256
class_names = ['diflush', 'med1', 'med2', 'med3', 'n2dry', 'noprocess', 'preflush']

# Initialize the plot with dummy data
fig, ax = plt.subplots()
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3))
im = ax.imshow(dummy)
ax.axis('off')

# Connect the key press event to the handler
fig.canvas.mpl_connect('key_press_event', on_key)

# Add initial instruction text
text_instructions = ax.text(0.5, -0.1, "Press the spacebar to proceed to the next image...", 
                            fontsize=12, ha='center', transform=ax.transAxes)

# Process images from the specified folder
for image_file in os.listdir(folder_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(folder_path, image_file)

        # Load and resize the image
        img = Image.open(image_path)
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert the image to a numpy array and normalize pixel values
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0

        # Predict class probabilities for input image
        predictions = model.predict(img_array)

        # Get predicted class label by finding the index of the maximum predicted probability
        predicted_label_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_label_index]
        predicted_probability = predictions[0][predicted_label_index]

        # Clear the plot and plot the new image with its predicted label and probability
        im.set_data(img)
        ax.set_title(f'Class: {predicted_class}, Probability: {predicted_probability:.2f}')
        text_instructions.set_text("Press the spacebar to proceed to the next image...")
        fig.canvas.draw()
        plt.pause(0.002)

        # Print prediction result
        if predicted_probability < 0.80:
            print(f'{predicted_class}, Probability: {predicted_probability:.2f}  WARNING: Low probability')
        else:
            print(f'{predicted_class}, Probability: {predicted_probability:.2f}')

        # Wait for user to press spacebar to proceed to the next image
        print("Press the spacebar to proceed to the next image...")
        proceed = False  # Reset the flag
        while not proceed:
            plt.pause(0.1)

    # If plot window is closed, break out of the loop
    if not plt.fignum_exists(fig.number):
        break

