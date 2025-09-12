import os
from trl.extras.vllm_client import VLLMClient
from transformers import AutoModelForCausalLM

vllm_node = os.environ["VLLM_NODE"]
print("Trying to call connect to vLLM on", vllm_node)

client = VLLMClient(host=vllm_node, server_port=8000)
client.generate(["Hello, vLLM server!", "Are you there?"])

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B",
                                             device_map="cuda")
client.init_communicator(device="cuda")
client.update_model_params(model)
