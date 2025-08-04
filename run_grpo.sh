#!/bin/bash
#SBATCH --job-name=test-trl-grpo
#SBATCH --cpus-per-task=56
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000827
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err


export HF_HOME=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf-cache
mkdir -p $HF_HOME

export HF_TOKEN_PATH=~/.cache/huggingface/token
mkdir -p logs

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

set -exuo pipefail

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

# When slurm reschedules a job that ended on node failure, it will run
# with the same job ID, clobbering the original logs. Rename the logs
# and include timestamp to avoid this.
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logfile_basename="${SLURM_JOB_NAME}-${SLURM_JOBID}-${timestamp}"
mv -f "logs/${SLURM_JOBID}.out" "logs/${logfile_basename}.out"
mv -f "logs/${SLURM_JOBID}.err" "logs/${logfile_basename}.err"

# Check if this is a restared run and if so, print the failure
# events/reasons for failed nodes. (This relies on "logs/latest.err"
# pointing to the error log of the failed run.)
if [[ -v SLURM_RESTART_COUNT ]]; then
    failed_node=$(grep 'Node failure' logs/latest.err | awk '{print $NF}')
    if [[ -z ${failed_node:+x} ]]; then
        echo "RUN RESTARTED but no node failure logged"
    else
        failed_node="${failed_node//$'\n'/ }"
        echo "RUN RESTARTED AFTER FAILURE OF NODE(s) $failed_node. Reason:"
        sacctmgr show event where node="$failed_node" format="NodeName,TimeStart,TimeEnd,State,Reason%100"
    fi
fi

# Symlink logs/latest.out and logs/latest.err for convenience and to
# support the above check.
ln -sf "${logfile_basename}.out" "logs/latest.out"
ln -sf "${logfile_basename}.err" "logs/latest.err"

# gpu monitoring
srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 /appl/local/csc/soft/ai/bin/gpu-energy --save

CMD="accelerate launch --config-file=./accelerate_config.yaml --num_processes=$NUM_PROCESSES --num_machines=$SLURM_NNODES --machine_rank=\$SLURM_NODEID --main_process_ip=$MAIN_PROCESS_IP --rdzv_backend=c10d train_grpo.py"
srun bash -c "$CMD"

srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 /appl/local/csc/soft/ai/bin/gpu-energy --diff
