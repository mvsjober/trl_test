#!/bin/bash
#SBATCH --job-name=test-vllm-trl-grpo
#SBATCH --cpus-per-task=56
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=3
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
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0.sif
#SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"

# Created venv with trl like this:
# singularity shell $SIF
# python3 -m venv --system-site-packages $VENV_NAME
# source $VENV_NAME/bin/activate

# pip install --upgrade trl "transformers<4.54.0"
VENV_ACTIVATE="venv-amd-trl-0.20.0/bin/activate"


#export NCCL_DEBUG=INFO
export NCCL_DEBUG=TRACE
#export NCCL_DEBUG_SUBSYS=ALL

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_ENABLE_DMABUF_SUPPORT=1

# export NCCL_CUMEM_ENABLE=0
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1

export VLLM_USE_TRITON_FLASH_ATTN=0

singularity exec $SIF bash -c "source $VENV_ACTIVATE && trl env"

export OMP_NUM_THREADS=7
NUM_NODES=$((SLURM_NNODES - 1))
NUM_PROCESSES=$(expr $NUM_NODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TRAIN_NODES="${NODELIST[@]:0:$((SLURM_NNODES - 1))}"
export VLLM_NODE=${NODELIST[-1]}

echo "VLLM_NODE=$VLLM_NODE"
echo "TRAIN_NODES=$TRAIN_NODES"

MODEL="Qwen/Qwen2-0.5B-Instruct"

# gpu monitoring
# srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 /appl/local/csc/soft/ai/bin/gpu-energy --save

# start vllm server
# TODO parameterize model
(set -x
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE --label singularity exec $SIF \
     bash -c "source $VENV_ACTIVATE && trl vllm-serve --model=$MODEL --tensor_parallel_size=1 --data_parallel_size=1 --enforce-eager=True" &
)
VLLM_PID=$!

# start GRPO training

CMD="source $VENV_ACTIVATE && accelerate launch --config-file=./accelerate_config.yaml --num_processes=$NUM_PROCESSES --num_machines=$NUM_NODES --machine_rank=\$SLURM_NODEID --main_process_ip=$MAIN_PROCESS_IP --rdzv_backend=c10d train_vllm_grpo.py"
(set -x
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --nodelist="$TRAIN_NODES" singularity exec $SIF \
     bash -c "$CMD"
)

kill $VLLM_PID

# srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 /appl/local/csc/soft/ai/bin/gpu-energy --diff
