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

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

set -exuo pipefail

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=7
NUM_NODES=$((SLURM_NNODES - 1))
NUM_PROCESSES=$(expr $NUM_NODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TRAIN_NODES="${NODELIST[@]:0:$((SLURM_NNODES - 1))}"
VLLM_NODE=${NODELIST[-1]}

MODEL="Qwen/Qwen2-0.5B-Instruct"

# gpu monitoring
srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 /appl/local/csc/soft/ai/bin/gpu-energy --save

# start vllm server
# TODO parameterize model
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE --label singularity_wrapper exec \
     bash -c "trl vllm-serve --model=$MODEL --tensor_parallel_size=8 --data_parallel_size=1 --enforce-eager=True" &
VLLM_PID=$!

# start GRPO training
echo $NODELIST
echo $TRAIN_NODES
echo "${NODELIST[@]:0:$((SLURM_NNODES - 1))}"
echo "${TRAIN_NODES[*]}"

CMD="accelerate launch --config-file=./accelerate_config.yaml --num_processes=$NUM_PROCESSES --num_machines=$NUM_NODES --machine_rank=\$SLURM_NODEID --main_process_ip=$MAIN_PROCESS_IP --rdzv_backend=c10d train_grpo.py"
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --nodelist="$TRAIN_NODES" bash -c "$CMD"

kill VLLM_PID

srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 /appl/local/csc/soft/ai/bin/gpu-energy --diff
