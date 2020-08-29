DATA_DIR=$1

for SPLIT in "train" "dev" "test" 
do
	python prepare_output_for_classifier.py ${DATA_DIR}/yelp_${SPLIT}_stars.txt ${DATA_DIR}/${SPLIT}.tsv
done
