# nvidia-triton-inference
This repository contains setup examples for hosting model inference using NVIDIA triton

## How to build a triton embedding image

1. Setup your model and tokenizer files

    - move model.onnx to `hf-embedding-template/onnx_model/1/`
    - move any other model files (model and tokenizer config) to `hf-embedding-template/preprocessing/1/`

2. Start Triton Server and attach shell

    ```
    docker run --shm-size=16g --gpus all -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /hf-embedding-template:/models nvcr.io/nvidia/tritonserver:24.08-py3 bash
    ```


3. Run inside the Triton Container

    ```
    pip install transformers

    tritonserver --model-repository=/models
    ```

4. Run client 

    ```
    pip install tritonclient[http]

    python client.py
    ```

## helm charts

This repo comes with ready to run helm charts. They can be found under `/helm`. E.g. `text-embedder-trion` is readily configured to run a triton embedding server.
