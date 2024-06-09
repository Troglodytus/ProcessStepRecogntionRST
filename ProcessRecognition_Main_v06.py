from class_pdsf import PDSF
from class_AIPredict import AIPredict
from class_email_messenger import EmailMessenger
from class_video_handler import VideoHandler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import json

destination_path = 'C:\\Users\\Eichleitner\\Documents\\Coding\\pdsf'
model_path = 'C:/Users/Eichleitner/Documents/Coding/RST_7Class_Aug_Model_v03.h5'
LUT = pd.read_excel('C:\\Users\\Eichleitner\\Documents\\Coding\\RecipeStepLUT.xlsx')
video_directory = 'C:\\Users\\Eichleitner\\Documents\\Coding\\video'
video_files = [f for f in os.listdir(video_directory) if f.endswith(".avi") or f.endswith(".mp4")]
video_handler = VideoHandler(video_directory)

for video_file in video_files:
    video = video_handler.load_video(video_file)
    equipment_list = ['RST5C', 'RST20C']
    pdsf = PDSF(destination_path)
    files = os.listdir(destination_path)
    print(f"Found {len(files)} files in the directory: {files}")

    predicted_classes = {}

    for file_name in files:
        timetable = pdsf.process_pdsf(file_name)
        timetable['timestamp'] = pd.to_datetime(timetable['timestamp'])
        timetable['timerel'] = (timetable['timestamp'] - timetable['timestamp'].iloc[0]).dt.total_seconds()
        print(timetable)

        class_names = ['diflush', 'med1', 'med2', 'med3', 'n2dry', 'noprocess', 'preflush']
        ai_predict = AIPredict(model_path, class_names)

        for index, row in timetable.iterrows():
            timestamp = row['timerel']
            EPA = row['A_EPA']
            RecipeStep = row['RecipeStep']
            label = LUT.query(f'EPA == "{EPA}" and RecipeStep == {RecipeStep}')['Label'].values[0]
            duration = timetable['timerel'].iloc[-1]
            cut_video_file = video_handler.cut_video(video_file, timestamp, duration)
            cut_video = video_handler.load_video(cut_video_file)
            predicted_class, predicted_probability = ai_predict.predict_image(cut_video, timestamp)

            if (EPA, RecipeStep) not in predicted_classes:
                predicted_classes[(EPA, RecipeStep)] = []
            if predicted_class is not None:
                predicted_classes[(EPA, RecipeStep)].append(predicted_class)
            if predicted_class is not None and predicted_probability is not None:
                print(f'Equipment: {timetable["A_MES_ENTITY"].iloc[0]}, Timestamp: {timestamp} seconds, Real Label: {label}.  Predicted Class: {predicted_class}, Probability: {predicted_probability:.2f}')
            else:
                print(f'Equipment: {timetable["A_MES_ENTITY"].iloc[0]}, Timestamp: {timestamp} seconds, No prediction available')

        alarm_text = ""
        for (EPA, RecipeStep), predictions in predicted_classes.items():
            label = LUT.query(f'EPA == "{EPA}" and RecipeStep == {RecipeStep}')['Label'].values[0]
            if label == 'noprocess':
                continue
            if len(predictions) >= 2:
                median_prediction = pd.Series(predictions).mode().values[0]
                print(f'EPA: {EPA}, Recipe Step: {RecipeStep}')
                print(f'Label: {label}, Median Prediction: {median_prediction}')
                if label != median_prediction:
                    alarm_text = f"PSR-RST ALARM:\nProcess Step Recognition RST Violation. Webcam footage does not match APC data!\n \nTool: {timetable['A_MES_ENTITY'].iloc[0]}\nLot: {timetable['LOT_ID'].iloc[0]}\nSlot: {timetable['A_SLOT'].iloc[0]}\nEPA: {timetable['A_EPA'].iloc[0]}\nAPC: {label}, PSR: {median_prediction}\nTime: {timetable['timestamp'].iloc[0]}\n\nPlease verify equipment and lot status manually, actions might be required.\n\n\nThis e-mail was automatically generated."
                    if alarm_text:
                        with open('emailconfig.json') as f:
                            data = json.load(f)
                            subjecttext =  f"PSR Alarm - Process Step Error - {timetable['A_MES_ENTITY'].iloc[0]}"
                            password = data['EMAIL_PASSWORD']
                            recipients = data['RECIPIENTS']
                            messenger = EmailMessenger("infineon\Eichleitner",  password, "Daniel.Eichleitner2@infineon.com")
                            messenger.send_email(subjecttext, alarm_text, recipients)
                            print(alarm_text)
            else:
                print(f'EPA: {EPA}, Recipe Step: {RecipeStep}')
                print('Not enough data to calculate a median prediction.')
