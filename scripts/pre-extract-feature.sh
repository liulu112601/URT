#!/bin/bash
# bash ./scripts/pre-extract-feature.sh resnet18 ./outputs/extract-feature
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 args: path"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  export TORCH_HOME="${HOME}/.torch"
else
  echo "TORCH_HOME : $TORCH_HOME"
fi
if [ "$DATASET_DIR" = "" ]; then
  export DATASET_DIR="${HOME}/scratch/meta-dataset-x"
else
  echo "DATASET_DIR : $DATASET_DIR"
fi
echo "DATASET_DIR : $DATASET_DIR"

export META_DATASET_ROOT="${HOME}/scratch/git/meta-dataset-v1"
export RECORDS="${HOME}/scratch/meta-dataset-records"

backbone=$1
save_dir=$2

echo "ROOT: $(pwd)"

ulimit -n 100000

python exps/pre-extract-feature.py --save_dir ${save_dir} \
	--model.backbone=${backbone}
