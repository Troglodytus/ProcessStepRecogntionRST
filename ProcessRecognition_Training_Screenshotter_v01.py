import time
import os
import keyboard
from PIL import ImageGrab

# define the dimensions of the area you want to capture
left = 210
top = 235
width = 870
height = 790

# Define a dictionary to map keypresses to their respective folders
folders_dict = {
    '2': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/med2/",
    '3': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/med3/",
    '4': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/diflush/",
    '1': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/med1/",
    '5': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/n2dry/",
    '6': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/preflush/",
    '7': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/noprocess/",
    '8': "C:/Users/Public/FrameCategorizationAlgorithm/TestValidation/mdwait/"
    }

# define a function to capture screenshot
def capture_screenshot(key):
    folder = folders_dict[key]
    existing_screenshots = [f for f in os.listdir(folder) if f.startswith("Screenshot_")]
    if existing_screenshots:
        numbers = []
        for f in existing_screenshots:
            try:
                # Grab the last part after splitting on "_" and split again to remove the extension
                number = int(f.split("_")[-1].split(".")[0])  # Use -1 to get the last item after split
                numbers.append(number)
            except ValueError:
                # Skip if it can't be converted to an integer
                continue
        j = max(numbers) if numbers else 0
    else:
        j = 0
    
    # capture the screen region and save it to the folder with a sequential name
    filename = f"{folder}Screenshot_{j+1}.png"
    ImageGrab.grab(bbox=(left, top, left+width, top+height)).save(filename)
    print(f"Screenshot_{j+1} saved in {folder}")

# set the keyboard event for each key-folder pair
for key in folders_dict.keys():
    keyboard.on_press_key(key, lambda e: capture_screenshot(e.name))

# This will keep your script running until you stop it manually.
keyboard.wait()
