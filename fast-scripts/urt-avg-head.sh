#!/bin/bash
# bash ./fast-scripts/urt-avg-head.sh ./fast-outputs/urt-avg-head 2 ${cache_dir}

echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 args: path"
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


save_dir=$1
n_head=$2
penalty_coef=$3
cache_dir=$4
temp=1
optimizer=adam
scheduler=cosine
test_interval=10000

export META_DATASET_ROOT="${HOME}/scratch/git/meta-dataset-v1"
export RECORDS="${HOME}/scratch/meta-dataset-records"
echo "ROOT: $(pwd)"

python fast-exps/urt-avg-head.py --save_dir ${save_dir} --cache_dir ${cache_dir} --urt.head ${n_head} --urt.penalty_coef ${penalty_coef} \
	--train.max_iter=10000 --train.weight_decay=1e-5 --interval.train 100 --interval.test ${test_interval} \
	--train.learning_rate=1e-2 --train.lr_decay_step_gamma=0.9 \
	--urt.temp=${temp} --train.optimizer=${optimizer} --train.scheduler=${scheduler}	
