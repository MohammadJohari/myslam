#!/bin/bash
# WARNING: You MUST use bash to prevent errors
. /idiap/resource/software/initfiles/shrc
export PYTHONPATH=./
export TORCH_HOME=/idiap/temp/mjohari/TORCH_HOME/
export LD_LIBRARY_PATH=/idiap/temp/mjohari/lib
export CUDA_VISIBLE_DEVICES=0 

grid_gpu="-l sgpu -l h=vgn[ji]*"
grid_args="${grid_gpu} -P ams -S $(which python3) -cwd -V"

output='/idiap/temp/mjohari/outputs/slam/Replica/r0_ncoa'
#python_command="run.py configs/Replica/room0.yaml --output "
python_command="run.py configs/ScanNet/scene0059.yaml"
# python_command="run.py configs/Apartment/apartment.yaml"

expname="o0_sigin"
expname="ex"$RANDOM

if [ $(qstat | grep ${expname} -c) == 0 ]; then
  log_dir="grid_logs"
  log_file="${log_dir}/${expname}.log"
  mkdir -p "${log_dir}"
  rm -f ${log_file}

  qsub ${grid_args} -e ${log_file} -o ${log_file} -N ${expname} ${python_command}
else
  echo "The job is already running"
fi