#export TRAIN_FILE="/home/vgangal/libs_from_src/transformers/wikitext2_raw/wiki.valid.raw"
#export TEST_FILE="/home/vgangal/libs_from_src/transformers/wikitext2_raw/wiki.test.raw"

#CUDA_VISIBLE_DEVICES=0,1 python examples/run_language_modeling.py \
#                        --output_dir tmp/wikitext2 \
#                        --model_type gpt2 \
#                        --model_name_or_path gpt2 \
#                       --do_train \
#                       --train_data_file $TRAIN_FILE \
#                       --num_train_epochs 2 \
#                       --per_gpu_train_batch_size 1 \
#                       --gradient_accumulation_steps 1 \
#                       --eval_data_file $TEST_FILE

#export TRAIN_FILE="./data/tongue_twisters/train.raw"
#export TEST_FILE="./data/tongue_twisters/valid.raw"

#CUDA_VISIBLE_DEVICES=0,1 python examples/run_language_modeling.py \
#                        --output_dir tmp/tongue_twisters \
#                        --model_type gpt2 \
#                        --model_name_or_path gpt2 \
#                       --do_train \
#                       --line_by_line \
#                       --train_data_file $TRAIN_FILE \
#                       --learning_rate 5e-5 \
#                       --num_train_epochs 4 \
#                       --per_gpu_train_batch_size 1 \
#                       --gradient_accumulation_steps 1 \
#                       --eval_data_file $TEST_FILE


# Configuration 201
# You need to set TRAIN_FILE, TEST_FILE, MODEL_DIR to point to the correct paths as per your system
# Format of TRAIN_FILE/TEST_FILE: Each line is a domain sentence (prompt space continuation, or just directly the domain sentence if there's no explicit prompt-continuation separation)
export TRAIN_FILE=$1 # "/data2/vgangal/fanstories/train.raw.seed42.54K" #"/data2/vgangal/fanstories/train.raw.seed42.20pc"
export TEST_FILE=$2 #"/data2/vgangal/fanstories/valid.raw.seed42.3K" #"/data2/vgangal/fanstories/valid.raw.seed42.20pc"
export MODEL_DIR=$3 #"/data2/vgangal/fanstories/ckpts/1e-6_3epochs.seed42.20pc" 

export MODEL_TYPE="gpt2" #gpt2 #This is the specific arch. For now only gpt2 is supported
export MODEL_SIZE="gpt2" #gpt2 #This is the specific arch you want to use within this type e.g gpt2 [gpt2-small] , gpt2-medium etc etc

export LR="5e-5"
export NUM_TRAIN_EPOCHS=4
export PER_GPU_TRAIN_BATCH_SIZE=2
export GRAD_ACCUMULATION_STEPS=1 #Increase this in case you're getting out of memory errors [This has to be a factor of the batch size, and cannot exceed the batch size itself]

export SEED=24

# Setting TOTAL_STEPS (to be given to --save_steps)
NUMBER_OF_LINES=`wc -l $TRAIN_FILE | cut -f1 -d' '`
TOTAL_STEPS=$(echo "scale = 0; $NUMBER_OF_LINES/$PER_GPU_TRAIN_BATCH_SIZE" | bc)
#TOTAL_STEPS=$(echo "scale = 0; $TOTAL_STEPS*$NUM_TRAIN_EPOCHS" | bc) #Uncomment this if you wanna save only one checkpoint at the very end of all finetuning epochs, and nothing else
echo $TOTAL_STEPS

CUDA_VISIBLE_DEVICES=0  python examples/run_language_modeling.py \
                        --output_dir ${MODEL_DIR} \
                        --model_type ${MODEL_TYPE} \
                        --model_name_or_path ${MODEL_SIZE} \
                        --do_train \
                       --line_by_line \
                       --train_data_file ${TRAIN_FILE} \
                       --do_eval \
		       --evaluate_during_training \
		       --learning_rate ${LR} \
		       --seed ${SEED} \
                       --num_train_epochs ${NUM_TRAIN_EPOCHS} \
                       --save_steps ${TOTAL_STEPS} \
		       --logging_steps ${TOTAL_STEPS} \
		       --save_total_limit 4 \
                       --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
                       --per_gpu_eval_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
		       --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
                       --eval_data_file ${TEST_FILE}