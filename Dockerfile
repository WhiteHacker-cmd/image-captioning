FROM python:3.9-slim-buster

# Install necessary packages
# RUN apk add --no-cache \
#     build-base \
#     libstdc++ \
#     python3-dev \
#     openblas-dev \
#     && rm -rf /var/cache/apk/*

# # Install PyTorch CPU version
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html 

WORKDIR /app

COPY . /app

RUN pip install -r requirement.txt


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
