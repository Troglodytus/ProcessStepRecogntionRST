# import matplotlib as plt
# matplotlib.use('Qt5Agg')


import time
import tensorflow as tf
print("TensorFlow-Version:", tf.__version__)
from tensorflow import keras
from PIL import ImageGrab
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import configparser


model_used = "model_process_step_recognition.keras"
config = configparser.ConfigParser()
config.read('config.ini')

# Load pre-trained model from H5 file
#model = keras.models.load_model('C:/Users/Eichleitner/Documents/Coding/RST5C_7Class_Model_v05.h5')
#model = keras.models.load_model('C:/Users/Eichleitner/Documents/Coding/RST_7Class_Aug_Model_v02.h5')
#model = keras.models.load_model('C:/Users/Eichleitner/Documents/Coding/RST_7Class_Aug_Model_v03.h5')
#model = keras.models.load_model('C:/Users/Public/ProcessStepRecognitionRST/Model/RST_7Class_256_Aug_Model_v05.h5')
#model = keras.models.load_model("RST_7Class_256_Aug_Model_v05.h5")
#model = keras.models.load_model('C:/Users/Eichleitner/Documents/Coding/RST_13Class_Aug_Model_v01.h5')

model_path = config['Paths']['model_path']
model = tf.keras.models.load_model(model_path,
        compile=compile,
        safe_mode=True,)

# Define image size and class labels
IMG_SIZE = 256
class_names = ['diflush', 'med1', 'med2', 'med3', 'n2dry', 'noprocess', 'preflush']
#class_names = ['diflush','diflusherror', 'med1','med1error', 'med2','med2error', 'med3', 'n2dry','n2dryerror', 'noprocess','noprocesserror', 'preflush']

# Define the dimensions of the area you want to capture
left = 210
#top = 235
top = 275
width = 890
height = 770

# Initialize the plot with dummy data
fig, ax = plt.subplots()
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3))
im = ax.imshow(dummy)
ax.axis('off')

# Continuously loop to take screenshots, classify them, and plot them
while True:
    # Capture the screen region and resize it to 256x256 pixels
    img = ImageGrab.grab(bbox=(left, top, left+width, top+height))
    #cut_left = 120
    #cut_top = 280
    #cut_right = width - 1270
    #cut_bottom = height - 290
    #cropped_image = img.crop((cut_left, cut_top, cut_right, cut_bottom))

    # resize the image to the target size
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
    fig.canvas.draw()

    # Wait for 1 second before taking the next screenshot and updating the plot
    plt.pause(0.002)
    if predicted_probability < 0.80:
        print(f'{predicted_class}, Probability: {predicted_probability:.2f}' + '  WARNING: Low probability')
    else:
        print(f'{predicted_class}, Probability: {predicted_probability:.2f}')
    time.sleep(3.5)

    # If plot window is closed, break out of the loop
    if not plt.fignum_exists(fig.number):
        break
