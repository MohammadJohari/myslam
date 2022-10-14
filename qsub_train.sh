#!/bin/bash
# WARNING: You MUST use bash to prevent errors
. /idiap/resource/software/initfiles/shrc
export PYTHONPATH=./
export TORCH_HOME=/idiap/temp/mjohari/TORCH_HOME/
export LD_LIBRARY_PATH=/idiap/temp/mjohari/lib
export CUDA_VISIBLE_DEVICES=0 

grid_gpu="-l sgpu -l h=vgn[ji]*"
grid_args="${grid_gpu} -P ams -S $(which python3) -cwd -V"

#for i in 0 1 2 3 4
for i in 0 1
do
  room="scene0207"

#  input_folder="/idiap/resource/database/ETHZ-Replica/${room}"
#  output="/idiap/temp/mjohari/outputs/slam/CVPR/Replica/${room}/ours_${i}"
#  python_command="run.py configs/Replica/${room}.yaml --input_folder ${input_folder} --output ${output}"

  input_folder="/idiap/temp/mjohari/Datasets/scannet/scans/${room}_00"
  output="/idiap/temp/mjohari/outputs/slam/CVPR_6d/ScanNet/${room}/ours_${i}"
  python_command="run.py configs/ScanNet/${room}.yaml --input_folder ${input_folder} --output ${output}"


  expname="ex"$RANDOM
  log_dir="grid_logs"
  log_file="${log_dir}/${expname}.log"
  mkdir -p "${log_dir}"
  rm -f ${log_file}

  qsub ${grid_args} -e ${log_file} -o ${log_file} -N ${expname} ${python_command}
done