#!/bin/bash
# WARNING: You MUST use bash to prevent errors
. /idiap/resource/software/initfiles/shrc
export PYTHONPATH=./
export TORCH_HOME=/idiap/temp/mjohari/TORCH_HOME/
export CUDA_VISIBLE_DEVICES=0 

grid_gpu="-l sgpu -l h=vgn[ji]*"
grid_args="${grid_gpu} -P ams -S $(which python3) -cwd -V"

python_command="run.py configs/Replica/office0.yaml"

expname="o0_sigin"
expname="e"$RANDOM

if [ $(qstat | grep ${expname} -c) == 0 ]; then
  log_dir="grid_logs"
  log_file="${log_dir}/${expname}.log"
  mkdir -p "${log_dir}"
  rm -f ${log_file}

  qsub ${grid_args} -e ${log_file} -o ${log_file} -N ${expname} ${python_command}
else
  echo "The job is already running"
fi
