from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import os 
import keras
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_addons as tfa
# from keras.preprocessing import image
from tensorflow_addons.optimizers import AdamW
import camera


model_face = YOLO("model\yolov8n-face.pt")

vgg_local = "model\local_vgg_official.h5"   # good
# vgg_cluster1 = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\VGG\cluster1_vgg_500.h5" # good
# vgg_cluster2 = r"E:\HOC_TAP\NCKH\model\VGG\cluster2_vgg_official.h5"

# lstm_local = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\resnet\local_resnet_official.h5"  # not good
# lstm_cluster1 = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\LSTM\cluster1_LSTM_official.h5"
# lstm_cluster2 = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\LSTM\cluster2_lstm_official.h5"

# resnet_local = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\resnet\local_resnet_official.h5"  
# resnet_cluster1 = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\resnet\cluster1_resnet_official.h5" # not good

# den_local = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\dense\local2_densenet_500.h5"    # not good
# den_cluster1 = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\dense\cluster1_densenet_official.h5"
# den_cluster2 = r"D:\\DATA_FOLDER\\HOC_TAP\NCKH\model\dense\cluster2_densenet_official.h5"

custom_objects = {
    'AdamW': AdamW,  # Add the custom optimizer if it
}
model = tf.keras.models.load_model(vgg_local, custom_objects=custom_objects,compile=True)#Load model with DNN:

camera.get_camera_detect(model_face, model)





