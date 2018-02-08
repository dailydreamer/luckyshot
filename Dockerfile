FROM gcr.io/tensorflow/tensorflow:1.5.0-py3

ENV OPENCV_VERSION 3.4.0
ENV NUM_CORES 4

# install cython and keras
RUN pip3 --no-cache-dir install Cython keras

# Install OpenCV
RUN apt-get -y update -qq && \
    apt-get -y install wget \
                       unzip \
                       # Required
                       build-essential \
                       cmake \
                       git \
                       pkg-config \
                       libatlas-base-dev \
                       libgtk2.0-dev \
                       libavcodec-dev \
                       libavformat-dev \
                       libswscale-dev \
                       # Optional
                       libtbb2 libtbb-dev \
                       libjpeg-dev \
                       libpng-dev \
                       libtiff-dev \
                       libv4l-dev \
                       libdc1394-22-dev \
                       qt4-default \
                       # Missing libraries for GTK
                       libatk-adaptor \
                       libcanberra-gtk-module \
                       # Tools
                       imagemagick \
                       # For use matplotlib.pyplot in python
                       python3-pip \
                       python3-tk \
                       python3-skimage \
                       && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

    # install cocoapi
RUN git clone https://github.com/cocodataset/cocoapi.git &&\
    cd cocoapi/PythonAPI &&\
    python3 setup.py build_ext install &&\
    cd / &&\
    rm -r cocoapi

    # Get OpenCV
RUN git clone https://github.com/opencv/opencv.git &&\
    cd opencv &&\
    git checkout $OPENCV_VERSION &&\
    cd / &&\
    # Get OpenCV contrib modules
    git clone https://github.com/opencv/opencv_contrib &&\
    cd opencv_contrib &&\
    git checkout $OPENCV_VERSION &&\
    # Build OpenCV
    mkdir /opencv/build &&\
    cd /opencv/build &&\
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D WITH_IPP=ON \
      -D WITH_CSTRIPES=ON \
      -D WITH_OPENCL=ON \
      -D WITH_V4L=ON \
      .. &&\
    make -j$NUM_CORES &&\
    make install &&\
    ldconfig &&\
    # Clean the install from sources
    cd / &&\
    rm -r /opencv &&\
    rm -r /opencv_contrib



# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

CMD ["/bin/bash"]