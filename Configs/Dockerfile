FROM nvidia/cuda:11.1.1-base-ubuntu20.04
RUN apt update && apt install -y python3.8 && apt install -y python3-pip
RUN apt install -y vim
RUN apt install -y git
RUN mkdir /configfiles
COPY packages.txt /configfiles
RUN pip3 install -r /configfiles/packages.txt
