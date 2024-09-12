# nvidia-triton-inference
This repository contains setup examples for hosting model inference using NVIDIA triton

## How to run

1. Start Triton Server and attach shell

    ```
    docker run --shm-size=16g --gpus all -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /onnx-server-side-hf-tokenizer:/models nvcr.io/nvidia/tritonserver:24.08-py3 bash
    ```


2. Run inside the Triton Container

    ```
    pip install transformers

    tritonserver --model-repository=/models
    ```

3. Run client 

    ```
    pip install tritonclient[http]

    python client.py
    ```