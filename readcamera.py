import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2
import time


def init():
  model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
  return model

def predict(model, frame):
  #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  img = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC)
  cv2.imshow('resized_img',img)
  x = np.asarray(img, dtype=np.float64)
  x = np.expand_dims(x, axis=0)

  # Zero-center by mean pixel
  #x = preprocess_input(x)
  x[..., 0] -= 103.939
  x[..., 1] -= 116.779
  x[..., 2] -= 123.68


  preds = model.predict(x)
  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  print('Predicted:', decode_predictions(preds, top=3)[0])

if __name__ == "__main__":
  model = init()
  cap = cv2.VideoCapture(0)
  while(True):
    start = time.process_time()
    ret, frame = cap.read()
    predict(model, frame)
    #cv2.imshow('frame',frame)
    end = time.process_time()
    frame_rate = 1 / (end - start)
    print("frame rate: ", frame_rate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()