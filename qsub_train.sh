#!/bin/bash
# WARNING: You MUST use bash to prevent errors
. /idiap/resource/software/initfiles/shrc
export PYTHONPATH=./
export TORCH_HOME=/idiap/temp/mjohari/TORCH_HOME/
export LD_LIBRARY_PATH=/idiap/temp/mjohari/lib
export CUDA_VISIBLE_DEVICES=0 

grid_gpu="-l sgpu -l h=vgn[jig]*"
grid_args="${grid_gpu} -P ams -S $(which python3) -cwd -V"

#for room in "scene0000" "scene0059" "scene0106" "scene0169" "scene0181" "scene0207" "scene0472"
for room in "scene0000" "scene0059" "scene0106" "scene0169" "scene0181" "scene0207"
do
  for i in 0 1 2
#  for i in 0
  do
#    room="scene0106"
  #  python_command="run.py configs/ScanNet/${room}.yaml"

  #  input_folder="/idiap/resource/database/ETHZ-Replica/${room}"
  #  output="/idiap/temp/mjohari/outputs/slam/CVPR_nadam21/Replica/${room}/ours_${i}"
  #  python_command="run.py configs/Replica/${room}.yaml --input_folder ${input_folder} --output ${output}"

    input_folder="/idiap/temp/mjohari/Datasets/scannet/scans/${room}_00"
    output="/idiap/temp/mjohari/outputs/slam/CVPR_w20last2c3over/ScanNet/${room}/ours_${i}"
    python_command="run.py configs/ScanNet/${room}.yaml --input_folder ${input_folder} --output ${output}"

    expname="ex"$RANDOM
    log_dir="grid_logs"
    log_file="${log_dir}/${expname}.log"
    mkdir -p "${log_dir}"
    rm -f ${log_file}

    qsub ${grid_args} -e ${log_file} -o ${log_file} -N ${expname} ${python_command}
    done
  done
