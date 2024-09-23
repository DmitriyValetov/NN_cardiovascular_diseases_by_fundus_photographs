# from ubuntu:22.04
from pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# docker build -f ./docker/Dockerfile -t base_torch_img ./docker
# docker run --user="root" --rm -it -v $PWD:/develop --gpus=all -p 8888:8888 base_torch_img bash
# docker run --user="root" --rm -it -v $PWD:/develop --gpus=all base_torch_img bash
# python3 -m jupyterlab --allow-root --port=8888 --ip=0.0.0.0

# python3 train_over_tgts_lrs.py


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 python3 python3-pip -y
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install segmentation-models-pytorch
RUN pip3 install jupyterlab scikit-learn pillow pycm matplotlib wandb opencv-python
RUN pip3 install nibabel seaborn pydicom onnx onnxruntime dicom2nifti
RUN pip3 install openpyxl
RUN pip3 install pytorch_warmup transformers densenet_pytorch

WORKDIR /develop