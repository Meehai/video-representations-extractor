# Use the latest Ubuntu base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive

# Update the package repository and install Python 3.9
RUN apt-get update && apt-get upgrade -y
RUN apt install -y iproute2 ffmpeg wget curl git fonts-open-sans imagemagick vim
RUN apt-get update
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install gdown pytest

# Open the SSH port and expose it
EXPOSE 22

# Start the SSH service
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Start the SSH service
CMD ["/usr/sbin/sshd", "-D"]

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

