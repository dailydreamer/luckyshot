import tensorflow as tf
import numpy as np
import cv2

class Model(object):
  """
  mode: One of "caffe", "tf".
    - caffe: BGR, [-127, 128], distract mean pixel respect to the ImageNet dataset
    - tf: RGB [-1,1]
  """
  def preprocess(self, frame, mode="caffe"):
    cv2.imshow('resized_img',frame)
    if mode == "caffe":
      x = np.asarray(frame, dtype=np.float32)
      # add dim for batch size
      x = np.expand_dims(x, axis=0)
      # Zero-center by mean pixel
      x[..., 0] -= 103.939
      x[..., 1] -= 116.779
      x[..., 2] -= 123.68
      return x
    elif mode == "tf":
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      x = np.asarray(frame, dtype=np.float32)
      # add dim for batch size
      x = np.expand_dims(x, axis=0)
      # center to [-1,1]
      x /= 255.
      x -= 0.5
      x *= 2.
      return x
    else:
      raise ValueError("Mode must be caffe or tf")
  
  # ImageNet decoding
  def predict(self, x):
    preds = self.model.predict(x)
    print('Predicted:', tf.keras.applications.vgg16.decode_predictions(preds)[0][:2])

class Xception(Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.applications.xception.Xception(weights='imagenet')

  def preprocess(self, frame):
    frame = cv2.resize(frame, (299, 299), interpolation = cv2.INTER_CUBIC)
    return super().preprocess(frame, mode="tf")


class Vgg16(Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.applications.vgg16.VGG16(weights='imagenet')

  def preprocess(self, frame):
    frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC)
    return super().preprocess(frame, mode="caffe")
    

class Resnet50(Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

  def preprocess(self, frame):
    frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC)
    return super().preprocess(frame, mode="caffe")

class InceptionV3(Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')

  def preprocess(self, frame):
    frame = cv2.resize(frame, (299, 299), interpolation = cv2.INTER_CUBIC)
    return super().preprocess(frame, mode="tf")

class InceptionResNetV2(Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet')
  
  def preprocess(self, frame):
    frame = cv2.resize(frame, (299, 299), interpolation = cv2.INTER_CUBIC)
    return super().preprocess(frame, mode="tf")

class MobileNet(Model):
  def __init__(self):
    super().__init__()
    self.model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet')

  def preprocess(self, frame):
    frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC)
    return super().preprocess(frame, mode="tf")