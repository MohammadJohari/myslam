#!/bin/bash
# WARNING: You MUST use bash to prevent errors
. /idiap/resource/software/initfiles/shrc
export PYTHONPATH=./
export TORCH_HOME=/idiap/temp/mjohari/TORCH_HOME/
export LD_LIBRARY_PATH=/idiap/temp/mjohari/lib
export CUDA_VISIBLE_DEVICES=0 

grid_gpu="-l sgpu -l h=vgn[gij]*"
grid_args="${grid_gpu} -P ams -S $(which python3) -cwd -V"

#for room in "scene0000" "scene0059" "scene0106" "scene0169" "scene0181" "scene0207" "scene0472"
#for room in "scene0059" "scene0106" "scene0169"
#for room in "scene0000" "scene0181" "scene0207"
for room in "scene0000"
#for room in "scene0059" "scene0106" "scene0169" "scene0181" "scene0207"
#for room in "room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4"
#for room in "room0" "room1"
do
#  for i in 0 1 2 3 4
  for i in 0 1
  do
#    python_command="run.py configs/Replica/${room}.yaml"

#    input_folder="/idiap/resource/database/ETHZ-Replica/${room}"
#    output="/idiap/temp/mjohari/outputs/slam/CVPR_md01td5mc1tc5_s32/Replica/${room}/ours_${i}"
#    python_command="run.py configs/Replica/${room}.yaml --input_folder ${input_folder} --output ${output}"

    input_folder="/idiap/temp/mjohari/Datasets/scannet/scans/${room}_00"
    output="/idiap/temp/mjohari/outputs/slam/CVPR_md01td1mc5tc5/ScanNet/${room}/ours_${i}"
    python_command="run.py configs/ScanNet/${room}.yaml --input_folder ${input_folder} --output ${output}"

    expname="ex"$RANDOM
    log_dir="grid_logs"
    log_file="${log_dir}/${expname}.log"
    mkdir -p "${log_dir}"
    rm -f ${log_file}

    qsub ${grid_args} -e ${log_file} -o ${log_file} -N ${expname} ${python_command}
    done
  done
