## Build image

```sh
sudo docker build -t dailydreamer/tensorflow-opencv-py3 .
```

## Run image

```sh
xhost +

sudo docker run -it --rm \
  --name devtest \
  --env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/dailydreamer/.keras:/root/.keras \
  -v $(pwd):$(pwd) -w $(pwd) \
  --device /dev/video0 \
  -p 8888:8888 \
  dailydreamer/tensorflow-opencv-py3

xhost -
```