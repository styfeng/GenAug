PROMPT_FILE=$1 # path to file with the prompt portions of the text (fed into GPT-2 as input)
CONTINUATIONS_FILE=$2 # path to file with the generated continuations

echo "Combining and flattening prompt + continuation files"
python flatten_generations.py $PROMPT_FILE $CONTINUATIONS_FILE "${CONTINUATIONS_FILE}_combined.txt"

TEMP_DIR="intermediate_${CONTINUATIONS_FILE}"
mkdir -p $TEMP_DIR
export MODEL_PATH=$3 # directory with the sentiment regressor
export TASK_NAME="STS-B"

python prepare_output_for_classifier.py "${CONTINUATIONS_FILE}_combined.txt" "${TEMP_DIR}/dev.tsv"

CUDA_VISIBLE_DEVICES=0 python ./examples/run_glue.py \
				  --model_type bert \
				  --model_name_or_path ${MODEL_PATH}/ \
				  --task_name $TASK_NAME \
				  --do_eval \
				  --do_lower_case \
				  --data_dir ${TEMP_DIR} \
				  --max_seq_length 128 \
				  --per_gpu_eval_batch_size 32 \
				  --output_dir ${MODEL_PATH}/

rm -f "${TEMP_DIR}/cached_dev_sentiment_classifier_128_sts-b"	
mv ${TEMP_DIR}/preds.json "${CONTINUATIONS_FILE}_sentiment_scores_GENERATED.json"
rm -f "${TEMP_DIR}/cached_dev_sentiment_classifier_128_sts-b"
echo "Done"