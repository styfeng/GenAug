export GLUE_DIR=$1 #"/data2/vgangal/yelp"
export MODEL_PATH=$2 #"/data2/vgangal/yelp/ckpts"
export TASK_NAME="STS-B"

#bert,bert-base-uncased,2e-5
CUDA_VISIBLE_DEVICES=1 python ./examples/run_glue.py \
                              --model_type bert \
                              --model_name_or_path bert-base-uncased \
                              --task_name $TASK_NAME \
                              --do_train \
                              --do_eval \
                              --do_lower_case \
                              --data_dir $GLUE_DIR \
                              --max_seq_length 128 \
                              --per_gpu_eval_batch_size 32 \
                              --per_gpu_train_batch_size 32 \
                              --learning_rate 2e-5 \
                              --num_train_epochs 3.0 \
                              --logging_steps 1563 \
                              --save_steps 4689 \
                              --output_dir ${MODEL_PATH}/
