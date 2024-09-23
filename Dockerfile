# Use NVIDIA Triton Inference Server as base image
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Copy all files and directories from the host folder to /models in the container
COPY ./hf-embedding-template /models

# Install the transformers library
RUN pip install transformers tritonclient

# Set the startup command to run Triton Inference Server with the specified model repository
CMD ["tritonserver", "--model-repository=/models"]
