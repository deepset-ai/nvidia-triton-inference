import numpy as np
import tritonclient.http as httpclient

inputs = []
outputs = []
input_texts = [" ".join(["The"] * 256)] * 256
input = httpclient.InferInput("text", [len(input_texts)], "BYTES")
input.set_data_from_numpy(np.array(input_texts, dtype=object), binary_data=False)
inputs.append(input)

triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=False
            )
results = triton_client.infer(
    "ensemble_model",
    inputs,
)

embeddings = results.as_numpy("sentence_embedding")

print(embeddings.shape)