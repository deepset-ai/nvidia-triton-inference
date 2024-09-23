import json
import os
from typing import Dict
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
from tritonclient.utils import triton_to_np_dtype


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        self.logger = pb_utils.Logger
        self.logger.log_info(f"Initializing tokenizer with args: {args}")
        path: str = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # inspect config.pbtxt
        config = json.loads(args["model_config"])

        # Output type mapping
        self.output_types = {
            output["name"]: triton_to_np_dtype(output["data_type"].removeprefix("TYPE_"))
            for output in config["output"]
        }

    def execute(self, requests):
        responses = []
        for request in requests:

            if request.is_cancelled():
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "Request cancelled", pb_utils.TritonError.CANCELLED
                        )
                    )
                )
                continue

            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            self.logger.log_info(f"input text shape: {input_tensor.shape()}")
            
            texts = [text.decode("utf-8") for text in input_tensor.as_numpy()]

            tokenizer_output = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")

            output_tensors = []
            for output_name in self.tokenizer.model_input_names:
                output_type = self.output_types[output_name]
                output_value = tokenizer_output[output_name].astype(output_type)
                self.logger.log_info(f"output {output_name} shape: {output_value.shape}")
                output_tensor = pb_utils.Tensor(output_name, output_value)
                output_tensors.append(output_tensor)

            inference_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(inference_response)
        return responses
