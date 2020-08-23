FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

COPY src /home/app
COPY docker/api/requirements.txt /tmp/requirements.txt

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3.7 python3.7-dev python3-pip python3-tk git g++ libncursesw5
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 0
RUN update-alternatives --set python /usr/bin/python3.7

RUN /home/app/XLM/install-tools.sh
RUN python -m pip install -r /tmp/requirements.txt --no-cache-dir
RUN ln -s /usr/local/lib/python3.7/dist-packages/clang/native/libclang.so /lib/x86_64-linux-gnu/libclang.so

WORKDIR /home/app

ENV PATH /home/app/.local/bin/:$PATH

#CMD ["/bin/bash"]
CMD ["python", "-m" ,"api.run"]
