# Run commands:
# docker build . -f Dockerfile -t meehai/vre
# docker push meehai/vre

# Use the latest Ubuntu base image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set the working directory to /app
WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive

# Update the package repository and install Python 3.9
RUN apt-get update
RUN apt-get install -y iproute2 ffmpeg wget curl git fonts-open-sans imagemagick vim

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install pytest

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Copy the rest of the files to the container
COPY vre vre
COPY bin bin
ENV PYTHONPATH "${PYTHONPATH}:/app/vre"
ENV VRE_WEIGHTS_DIR /app/resources/weights
ENTRYPOINT ["python", "bin/vre"]

#### LOCAL DEVELOPMENT BELOW ####

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

## === For local development -- connect via SSH to this image: ===
# Decomment the lines below, start the container and then
# Find the ip: docker container inspect -f '{{.NetworkSettings.IPAddress}}' <container_id>
# Finally, ssh to the container: ssh root@<ip>
# When inside the SSH, do this:
# cd /app
# rm -rf *
# git clone https://gitlab.com/mihaicristianpirvu/video-representations-extractor [-b some_branch]
# cd video-representations-extractor/
# export VRE_ROOT=`pwd`
# export VRE_WEIGHTS_DIR=$VRE_ROOT/resources/weights
# export PYTHONPATH="$PYTHONPATH:$VRE_ROOT"
# export PATH="$PATH:$VRE_ROOT/bin"
# python3 -m pip install -r requirements.txt
# ... run vre manually like the gitlab CI does ...
