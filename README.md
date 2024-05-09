# CLIP Service Network Application

## Quick start

Run CLIP service in docker:

```bash
sudo docker run -p 5896:5896 --network host --gpus all but5gera/openclip_service:latest 
```

Docker build:

```bash
git clone https://github.com/5G-ERA/CLIPService.git
cd CLIPService 
sudo docker build -f docker/openclip_service.Dockerfile -t but5gera/openclip_service:latest .
```

Download sample files:\
https://raw.githubusercontent.com/5G-ERA/CLIPService/main/videos/video3.mp4 \
https://raw.githubusercontent.com/5G-ERA/CLIPService/main/config/config.yaml \

Install example client package:

```bash
python3 -m venv myvenv
myvenv\Scripts\activate
pip install openclip-client
```

Run client example:

```bash
openclip_client_python -c config.yaml video3.mp4
```

## Complete installation

Create python virtual environment, e.g.:

```bash
python3 -m venv myvenv
myvenv\Scripts\activate
```

and install openclip packages:

```bash
pip install openclip-client openclip-service
```

For CUDA accelerated version, on Windows may be needed e.g.:

```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu123
```

It depends on the version of CUDA on the system [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

## Network Application for 5G-ERA

### Run CLIP service / 5G-ERA Network Application

#### Run in Docker

The CLIP service can be started in docker ([docker/openclip_service.Dockerfile](docker/openclip_service.Dockerfile)).
The image can be built:

```bash
git clone https://github.com/5G-ERA/CLIPService.git
cd CLIPService 
sudo docker build -f docker/openclip_service.Dockerfile -t but5gera/openclip_service:latest .
```

or the image directly from the Docker Hub can be used.
 
The startup can be like this, where the GPU of the host computer is used and 
TCP port 5896 are mapped to the host network.

```bash
docker run -p 5896:5896 --network host --gpus all but5gera/openclip_service:latest 
```

To change port, e.q. to 5897:

```bash
docker run -p 5897:5897 -e NETAPP_PORT=5897 --network host --gpus all but5gera/openclip_service:latest 
```

#### Local startup

Download pretrained binary files into CLIPService/data/:
- laion2b_s34b_b88k
https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/blob/main/open_clip_pytorch_model.bin
- laion2b_s34b_b79k
https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/blob/main/open_clip_pytorch_model.bin

The CLIP service can also be started locally using [openclip-service/openclip_service/interface.py](openclip-service/openclip_service/service.py), 
but the openclip-service package must be installed and the NETAPP_PORT environment variable should be set
(default is 5896). Run CLIP service in same virtual environment as standalone example:

```bash
openclip_service
```

## Run client

In other terminal and in same virtual environment, set NETAPP_ADDRESS environment 
variable (default is http://localhost:5896) and run CLIP python client example:

```bash
openclip_client_python -c config/config.yaml videos/video3.mp4
```

## Configuration

The yaml configuration file contains:
- reduced_image_height: Images are reduced to this size before sending to the service, e.g.:
    ```yaml
    reduced_image_height: 720
    ```
- models: OpenCLIP model names and pretrained checkpoints names (or filenames), binary files must be downloaded 
beforehand, e.g.:
    ```yaml
    models:
      ViT-B-32:
        pretrained: ../../data/laion2b_s34b_b79k.bin
      ViT-B-16:
        pretrained: laion2b_s34b_b88k
    ```
- init_texts: List of initial texts to evaluate, e.g.:
    ```yaml
    init_texts:
      ViT-B-32:
        - car
        - bus
        - truck
        - bicycle
        - motorcycle
        - person
      ViT-B-16:
        - car
        - text
        - light
        - red
    ```

## Notes

If poetry lock hangs:

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

ROS 2 version is not functional yet.
Run ROS 2 service node with ROS params

```bash
python openclip_service_ros2/openclip_service_node.py --ros-args  --params-file ../config/config_ros_2.yaml
```