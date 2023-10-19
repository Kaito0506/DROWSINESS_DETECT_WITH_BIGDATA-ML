import cv2
import numpy as np
import time

import pygame
pygame.init()
import torch
pygame.mixer.init()
alert= pygame.mixer.Sound("./sound/alert.mp3")



RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


def count_drowsy(frames):
    count = 0 
    for i in frames:
        if i == 0 :
            count +=1
    return count
            

def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Face", image)
    # image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image


def get_camera_detect(model_face, model):
    frame_count = 0
    limit_frame = 12
    drowsy_count = 0 
    drowsy=False
    camera = cv2.VideoCapture(r"C:\Users\homin\Videos\test_video3.mp4")
    # camera = cv2.VideoCapture(1)
    # Set the desired width and height
    width = 720  # Replace with your desired width
    height = 480  # Replace with your desired height
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    drowsy_frames = []
    while True:
        _, frame = camera.read()
        
        frame = cv2.resize(frame, (width, height))
        
        prev_time = time.time()
        result = model_face(frame, show=False)
        print(result) # print result
        if result is not None and len(result[0].boxes) > 0:
            # print(result[0].boxes.xyxy.tolist()[0][1])
            boxes = result[0].boxes
            ##### frame count
            frame_count +=1
            box = boxes[0]
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bot_right_x = int(box.xyxy.tolist()[0][2])
            bot_right_y = int(box.xyxy.tolist()[0][3])
            if drowsy:
                cv2.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), RED, 2) 
            else:
                cv2.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), GREEN, 2)
            
                    
            print(top_left_x, top_left_y, bot_right_x, bot_right_y)
            print("****************")
            face = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
            
            
            if face.shape[0] > 0 and face.shape[1] > 0:
                face = preprocess(face)
                print(face.shape)
                result = model.predict(face)
                index = np.argmax(result)
                label = ["DROWSY", "NON DROWSY"]	
                label = label[index]
                if index == 0:
                    drowsy_count += 1
                    drowsy_frames.append(0)
                    drowsy = True
                else:
                    drowsy_frames.append(1)
                    drowsy = False

                print(label)

            
            # give prediction
            if frame_count >= limit_frame:
                print(drowsy_frames)
                rate = (count_drowsy(drowsy_frames)/limit_frame)*100
                cv2.putText(frame, "Asleep: {:.2f}%".format(rate), (10,200), cv2.FONT_HERSHEY_DUPLEX, 1, RED, 1)
                if count_drowsy(drowsy_frames) >= limit_frame*0.75:
                    cv2.putText(frame, "DROWSY DANGER!!!!", (50,100), cv2.FONT_HERSHEY_DUPLEX, 1.5, RED, 2)
                    alert.play()
                drowsy_frames.pop(0)
        
    
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)  
        # cv2.putText(frame, str(count), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 1)    
        cv2.imshow("Frame", frame)
        if cv2.waitKey(10) & 0xFF==27:
            break    
        
        
        
#############################
# def get_camera_frame(model_face, model):
#     camera = cv2.VideoCapture(r"C:\Users\homin\Videos\test_video.mp4")
#     while True:
#         _, frame = camera.read()
#         prev_time = time.time()
#         result = model_face(frame, show=False)
#         # print(result[0].boxes.xyxy.tolist()[0][1])
#         boxes = result[0].boxes
#         ##### frame count
        
#         for box in boxes:
#             top_left_x = int(box.xyxy.tolist()[0][0])
#             top_left_y = int(box.xyxy.tolist()[0][1])
#             bot_right_x = int(box.xyxy.tolist()[0][2])
#             bot_right_y = int(box.xyxy.tolist()[0][3])
#             cv2.rectangle(frame, (top_left_x, top_left_y), (bot_right_x, bot_right_y), GREEN, 2)
            
                   
#             print(top_left_x, top_left_y, bot_right_x, bot_right_y)
#             print("****************")
#             face = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
            
            
#             if face.shape[0] > 0 and face.shape[1] > 0:
#                 face = preprocess(face)
#                 print(face.shape)
#                 result = model.predict(face)
#                 index = np.argmax(result)
#                 label = ["DROWSY", "NON DROWSY"]	
#                 label = label[index]
#                 cv2.putText(frame, label, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 2)

#                 print(label)
#         # Calculate FPS
#         current_time = time.time()
#         fps = 1.0 / (current_time - prev_time)
#         prev_time = current_time
        
#         # give prediction        
#         # Display FPS on the frame
#         cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)
            
            
#         # cv2.putText(frame, str(count), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 1)    
#         cv2.imshow("Frame", frame)
#         if cv2.waitKey(10) & 0xFF==27:
#             break    