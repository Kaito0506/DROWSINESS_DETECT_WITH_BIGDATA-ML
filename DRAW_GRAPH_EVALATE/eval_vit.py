import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras.preprocessing import image
import matplotlib.pyplot as plt
import glob
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#Patches
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
        images=images,
        sizes=[1,self.patch_size, self.patch_size, 1],
        strides=[1,self.patch_size, self.patch_size, 1],
        rates=[1,1,1,1],
        padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

####Patchencoder
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
      super(PatchEncoder, self).__init__()
      self.num_patches = num_patches
      self.projection = layers.Dense(units=projection_dim)
      self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
      position = tf.range(start=0, limit=self.num_patches, delta=1)
      encoded = self.projection(patch) + self.position_embedding(position)
      return encoded


# vit_local = r"E:\HOC_TAP\NCKH\model\ViT\local_vit_500.h5"
# vit_cluster1 = r"E:\HOC_TAP\NCKH\model\ViT\cluster1_vit_500.h5"
# vit_cluster2 = r"E:\HOC_TAP\NCKH\model\ViT\cluster2_vit_official2.h5"
vit_local = r"/mnt/e/HOC_TAP/NCKH/model/ViT/local_vit_500.h5"
vit_cluster1 = r"/mnt/e/HOC_TAP/NCKH/model/ViT/cluster1_vit_500.h5"
vit_cluster2 = r"/mnt/e/HOC_TAP/NCKH/model/ViT/cluster1_vit_500.h5"

custom_objects = {'Patches': Patches, 'PatchEncoder': PatchEncoder, 'Addons>AdamW': tfa.optimizers.AdamW}

model = tf.keras.models.load_model(vit_cluster2, custom_objects=custom_objects)
#funtion to preprocess

# images= glob.glob(r'D:\BIG_PROJECT\face-detection-yolov8\test_face_images\test_cnn\drowsy\*.*')
images_drowsy= glob.glob(r'/mnt/d/BIG_PROJECT/face-detection-yolov8/test_face_images/test_vit/drowsy/*.*')
images_non_drowsy= glob.glob(r'/mnt/d/BIG_PROJECT/face-detection-yolov8/test_face_images/test_vit/non-drowsy/*.*')
test = []
label = []



def preprocess(image):
    # image=tf.keras.preprocessing.image.load_img(i, color_mode="rgb" ,target_size= (224,224))
    # image=np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # # cv2.imshow('Image', image)
    # cv2.waitKey(1000)
    ###########################other way
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



for i in images_drowsy:
    image = cv2.imread(i)  # Read the image using OpenCV
    if image is not None:
        image = preprocess(image)
        test.append(image)
        label.append(0)
        # cv2.waitKey(1000)
        
for i in images_non_drowsy:
    image = cv2.imread(i)  # Read the image using OpenCV
    if image is not None:
        image = preprocess(image)
        test.append(image)
        label.append(1)
        # cv2.waitKey(1000)

test = np.array(test)
label = np.array(label)

result = model.predict(test)
pre_label = np.argmax(result, axis=-1)
print(pre_label)

labels = label.reshape(-1, 1)
# print(y_test_label)
# print(pre_labeld)
# Tính accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(labels, pre_label)
print('Accuracy: %f' % accuracy)
# Tính precision tp / (tp + fp)
precision = precision_score(labels, pre_label, average='macro')
print('Precision: %f' % precision)
# Tính recall: tp / (tp + fn)
recall = recall_score(labels, pre_label, average='macro')
print('Recall: %f' % recall)
# Tính f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(labels, pre_label, average='macro')
print('F1 score: %f' % f1)
