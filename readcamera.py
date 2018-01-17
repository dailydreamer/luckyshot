import cv2
import time
from models import Xception, Vgg16, Resnet50, InceptionV3, InceptionResNetV2, MobileNet


if __name__ == "__main__":
  model = Vgg16()
  cap = cv2.VideoCapture(0)
  while(True):
    start = time.process_time()

    ret, frame = cap.read()
    read_time = time.process_time()

    x = model.preprocess(frame)
    preprocess_time = time.process_time()    

    model.predict(x)
    predict_time = time.process_time()

    print("read time: ", read_time - start)
    print("pre process time: ", preprocess_time - read_time)
    print("predict time: ", predict_time - preprocess_time)

    frame_rate = 1 / (predict_time - preprocess_time)
    print("frame rate: ", frame_rate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()