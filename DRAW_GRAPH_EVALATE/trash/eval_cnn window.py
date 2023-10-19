from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
import cv2
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras.preprocessing import image

images_drowsy= glob.glob(r'D:\\BIG_PROJECT\face-detection-yolov8\test_face_images\test_cnn\drowsy\*.*')
images_non_drowsy= glob.glob(r'D:\\BIG_PROJECT\face-detection-yolov8\test_face_images\test_cnn\non-drowsy\*.*')
test = []
label = []

vgg_local = r"E:\\HOC_TAP\NCKH\model\VGG\local_vgg_official.h5"
vgg_cluster1 = r"E:\\HOC_TAP\NCKH\model\VGG\cluster1_vgg_500.h5"

lstm_local = r"E:\\HOC_TAP\NCKH\model\resnet\local_resnet_official.h5"
lstm_cluster1 = r"E:\\HOC_TAP\NCKH\model\LSTM\cluster1_LSTM_official.h5"
lstm_cluster2 = r"E:\\HOC_TAP\NCKH\model\LSTM\cluster2_lstm_official.h5"

resnet_local = r"E:\\HOC_TAP\NCKH\model\resnet\local_resnet_official.h5"
resnet_cluster1 = r"E:\\HOC_TAP\NCKH\model\resnet\cluster1_resnet_official.h5"

den_local = r"E:\\HOC_TAP\NCKH\model\dense\local2_densenet_500.h5"
den_cluster1 = r"E:\\HOC_TAP\NCKH\model\dense\cluster1_densenet_official.h5"
den_cluster2 = r"E:\\HOC_TAP\NCKH\model\dense\cluster2_densenet_official.h5"



model = tf.keras.models.load_model(den_local,compile=True)#Load model with DNN:

def preprocess(image):
    image = cv2.resize(image, (224, 224))
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



for i in images_drowsy:
    # image = cv2.imread(i)  # Read the image using OpenCV
    # if image is not None:
    #     image = preprocess(image)
    #     test.append(image)
    #     label.append(0)
        # cv2.waitKey(1000)
        ####################
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))
    image=np.array(image)
    test.append(image)
    label.append(0)
        
for i in images_non_drowsy:
    # image = cv2.imread(i)  # Read the image using OpenCV
    # if image is not None:
    #     image = preprocess(image)
    #     test.append(image)
    #     label.append(1)
        # cv2.waitKey(1000)
        ######################
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= (224,224))
    image=np.array(image)
    test.append(image)
    label.append(1)
        
        
test = np.array(test)
label = np.array(label)

result = model.predict(test)
pre_label = np.argmax(result, axis=-1)
print(pre_label)

labels = label.reshape(-1, 1)
# print(y_test_label)
# print(pre_labeld)
# Tính accuracy: (tp + tn) \ (p + n)
accuracy = accuracy_score(labels, pre_label)
print('Accuracy: %f' % accuracy)
# Tính precision tp \ (tp + fp)
precision = precision_score(labels, pre_label, average='macro')
print('Precision: %f' % precision)
# Tính recall: tp \ (tp + fn)
recall = recall_score(labels, pre_label, average='macro')
print('Recall: %f' % recall)
# Tính f1: 2 tp \ (2 tp + fp + fn)
f1 = f1_score(labels, pre_label, average='macro')
print('F1 score: %f' % f1)
