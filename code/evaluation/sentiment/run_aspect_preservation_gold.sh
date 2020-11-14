GOLD_FILE=$1 # path to gold text (full original text - both prompt and continuation portions)

TEMP_DIR="intermediate_${GOLD_FILE}"
mkdir -p $TEMP_DIR
export MODEL_PATH=$2 # directory with the sentiment regressor
export TASK_NAME="STS-B"

python prepare_output_for_classifier.py "${GOLD_FILE}" "${TEMP_DIR}/dev.tsv"

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
mv ${TEMP_DIR}/preds.json "${GOLD_FILE}_sentiment_scores_GOLD.json"
rm -f "${TEMP_DIR}/cached_dev_sentiment_classifier_128_sts-b"
echo "Done"