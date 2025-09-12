#!/bin/bash
#SBATCH --job-name=test-vllm-trl-grpo
#SBATCH --cpus-per-task=56
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000007

export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

# export HF_TOKEN_PATH=~/.cache/huggingface/token
# mkdir -p logs

# AMD PyTorch
module purge
#SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0.sif
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"

# Created venv with trl like this:
# singularity shell $SIF
# python3 -m venv --system-site-packages venv-amd-2.7.1-trl
# source venv-amd-2.7.1-trl/bin/activate
# pip install trl
# pip install --upgrade aiter
# vim venv-amd-2.7.1-trl/lib/python3.12/site-packages/trl/scripts/vllm_serve.py
# 
# add these lines at the beginning of the file:
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)

VENV_ACTIVATE="venv-amd-2.7.1-trl/bin/activate"

export NCCL_DEBUG=INFO
#export NCCL_DEBUG=TRACE
#export NCCL_DEBUG_SUBSYS=ALL

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_ENABLE_DMABUF_SUPPORT=1

export VLLM_USE_TRITON_FLASH_ATTN=0

singularity exec $SIF bash -c "source $VENV_ACTIVATE && trl env"

export OMP_NUM_THREADS=7

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export VLLM_NODE=${NODELIST[-1]}

echo "VLLM_NODE=$VLLM_NODE"
echo "TRAIN_NODES=$TRAIN_NODES"

MODEL="Qwen/Qwen2-0.5B-Instruct"

# start vllm server
(set -x
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE --label singularity exec $SIF \
     bash -c "source $VENV_ACTIVATE && trl vllm-serve --model=$MODEL --tensor_parallel_size=1 --data_parallel_size=1 --enforce-eager=True" &
)

# wait until vLLM is running properly
sleep 10
while ! curl $VLLM_NODE:8000 >/dev/null 2>&1
do
    sleep 10
done

# simple connect back to vllm using trl.extras.vllm_client.VLLMClient
singularity exec $SIF bash -c "source $VENV_ACTIVATE && python3 trl_vllm_connect.py"
