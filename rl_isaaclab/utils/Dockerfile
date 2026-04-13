FROM sharpadev/sharpawavesdk:4.6.6.357f

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-venv \
    libevdev-dev

RUN pip3 install --break-system-packages \
    gymnasium==1.2.1 \
    omegaconf==2.3.0 \
    pynput \
    termcolor \
    tensorboardX

RUN pip3 install --break-system-packages --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.0 \
    torchvision==0.24.0

RUN pip3 cache purge

RUN apt-get remove -y \
    build-essential \
    libevdev-dev

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
