import os
from typing import Dict
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        path: str = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.logger = pb_utils.Logger

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            self.logger.log_info(f"input text shape: {input_tensor.shape()}")
            
            texts = [text.decode("utf-8") for text in input_tensor.as_numpy().tolist()]

            tokenizer_output = self.tokenizer(texts, padding=True)
            input_ids = np.array(tokenizer_output["input_ids"])
            attention_mask = np.array(tokenizer_output["attention_mask"])

            self.logger.log_info(f"output input_ids shape: {input_ids.shape}")
            self.logger.log_info(f"output attention_mask shape: {attention_mask.shape}")

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "input_ids",
                        input_ids,
                    ),
                    pb_utils.Tensor(
                        "attention_mask",
                        attention_mask,
                    ),
                ]
            )
            responses.append(inference_response)
        return responses
