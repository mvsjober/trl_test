# train_grpo.py
import os

# local_rank = os.environ.get('LOCAL_RANK',-1)
# print("LOCAL_RANK=", local_rank)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

import trl
print("trl version", trl.__version__)

dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    bf16=True,
    use_vllm=True,
    max_steps=10,
    vllm_server_host=os.environ["VLLM_NODE"],
    vllm_server_port=8000
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
