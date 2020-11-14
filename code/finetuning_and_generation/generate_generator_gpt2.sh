#CUDA_VISIBLE_DEVICES=0 python examples/run_generation.py \
#    --model_type=gpt2 \
#    --length=100 \
#    --prompt="marilyn monroe starred for" \
#    --model_name_or_path="./tmp/wikitext2" \

#CUDA_VISIBLE_DEVICES=0 python examples/run_generation.py \
#    --model_type=gpt2 \
#    --length=100 \
#    --model_name_or_path="./tmp/retweet_more_2epochs" \

#CUDA_VISIBLE_DEVICES=0 python examples/run_generation.py \
#    --model_type=gpt2 \
#    --length=100 \
#    --multi_prompt \
#    --multi_prompt_file_name="./data/retweet_more/dev.input.raw" \
#    --output_gen_file_name="./tmp/retweet_more_4epochs/dev.output.raw" \
#    --model_name_or_path="./tmp/retweet_more_4epochs" \

#CUDA_VISIBLE_DEVICES=0 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=340 \
#    --stop_token="[STOP]" \
#    --multi_prompt \
#    --multi_prompt_file_name="./data/muffin_captions_300/dev.input.raw" \
#    --output_gen_file_name="./tmp/muffin_captions_300_3epochs/dev.output.raw" \
#    --model_name_or_path="./tmp/muffin_captions_300_3epochs" \

# Note: Run this configuration to run with default nucleus decoding parameters and using the non-finetuned GPT-2 model.
# Configuration 101
# Don't forget to set multi_prompt_file_name, output_gen_file_name to point appropriately as per your system.
#echo "p=0.9"
#CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.nonfinetuned.nucleus.p0.9.T1.00.50" \
#    --model_name_or_path=gpt2 \

# Note: Run this configuration to run with default nucleus decoding parameters and using the finetuned GPT-2 model
# Configuration 102
# Don't forget to set multi_prompt_file_name, output_gen_file_name and model_name_or_path to point appropriately as per your system.
#echo "p=0.9"
#CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.finetuned.nucleus.p0.9.T1.00.50" \
#    --model_name_or_path="/data2/vgangal/fanstories/ckpts/5e-5_3epochs.seed42.20pc" \




#echo "p=0.9"
#CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.finetuned.nucleus.p0.9.T1.00.50" \
#    --model_name_or_path="/data2/vgangal/fanstories/ckpts/5e-5_3epochs.seed42.20pc" \

#echo "p=0.85"
#CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.85 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.nucleus.p0.85.T1.00.50" \
#    --model_name_or_path=gpt2 \

#echo "p=0.80"
#CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.80 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.nucleus.p0.80.T1.00.50" \
#    --model_name_or_path=gpt2 \



#echo "T=1.75"
#CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.75 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.nucleus.p0.9.T1.75.50" \
#    --model_name_or_path=gpt2 \

#echo "5Epochs, 1e-6"
#CUDA_VISIBLE_DEVICES=1 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.finetuned.1e-6.nucleus.p0.9.T1.00.50.5Epochs" \
#    --model_name_or_path="/data2/vgangal/fanstories/ckpts/1e-6_5epochs.seed42.20pc" \

#echo "4Epochs, 1e-6"
#CUDA_VISIBLE_DEVICES=1 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.finetuned.1e-6.nucleus.p0.9.T1.00.50.4Epochs" \
#    --model_name_or_path="/data2/vgangal/fanstories/ckpts/1e-6_5epochs.seed42.20pc/checkpoint-109040" \

#echo "3Epochs, 1e-6"
#CUDA_VISIBLE_DEVICES=1 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.finetuned.1e-6.nucleus.p0.9.T1.00.50.3Epochs" \
#    --model_name_or_path="/data2/vgangal/fanstories/ckpts/1e-6_5epochs.seed42.20pc/checkpoint-81780" \

#echo "2Epochs, 1e-6"
#CUDA_VISIBLE_DEVICES=1 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=50 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source" \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_output.finetuned.1e-6.nucleus.p0.9.T1.00.50.2Epochs" \
#    --model_name_or_path="/data2/vgangal/fanstories/ckpts/1e-6_5epochs.seed42.20pc/checkpoint-54520" \

#echo "3Epochs, 1e-6,Gold"
#CUDA_VISIBLE_DEVICES=2 python examples/run_generation.py \
#   --model_type=gpt2 \
#    --length=750 \
#    --multi_prompt \
#    --stop_token="<|endoftext|>" \
#    --num_return_sequences=500 \
#    --p=0.9 \
#    --temperature=1.00 \
#    --multi_prompt_file_name="/data2/vgangal/fanstories/valid.uq_source.single_sentence.scpn.table.plist"  \
#    --output_gen_file_name="/data2/vgangal/fanstories/valid.uq_source.single_sentence.scpn.table.plist.finetuned.1e-6.nucleus.p0.9.T1.500.50.3Epochs" \
#    --model_name_or_path="/data2/vgangal/fanstories/ckpts/1e-6_5epochs.seed42.20pc/checkpoint-81780" \

#File inputs [Do not overinterpret the specific filenames here too much. The names are only for illustration]
INPUT_PROMPT_FILE=$1  # e.g "/data2/vgangal/fanstories/valid.uq_source.single_sentence.scpn.table.gold" #Input prompt file, one prompt per line
GEN_FILE=$2 #e.g"/data2/vgangal/fanstories/valid.uq_source.single_sentence.scpn.table.gold.finetuned.1e-6.nucleus.p0.8.T1.00.100.3Epochs" #Output continuation file,continuations separated by <CAND_SEP>
MODEL_DIR=$3 #e.g "/data2/vgangal/fanstories/ckpts/1e-6_5epochs.seed42.20pc/checkpoint-81780" \      #Directly containing the model files, e.g the .bin file, the vocab etc etc.
#Hyperparameter settings
RETURN_SEQUENCES=100 #How many output continuations to return per prompt?
NUCLEUS_BUDGET=0.9   #Budget for nucleus sampling. The top choices whose mass covers this fraction of the output distribution are the only ones sampled from.
LENGTH_LIMIT=500     #This is to truncate catastrophically long sequences which are generated. Note that domain finetuning by itself should roughly ensure that typical continuations obey soft length upperbounds
SEED=24
MODEL_TYPE=gpt2
# Other sizes of the same arch: gpt2-medium, gpt2-large, gpt2-xl. Note however that these are quite big, especially once you want to finetune them before
# Other arch types: ctrl, openai-gpt, transfo-xl-wt103, xlnet-base-cased etc. Note that this should be consistent with the checkpoint you are supplying under MODEL_FILE

echo "Running output generation for ${GEN_FILE}"
CUDA_VISIBLE_DEVICES=0 python examples/run_generation_bounded.py \
   --model_type=${MODEL_TYPE} \
    --length=${LENGTH_LIMIT} \
    --multi_prompt \
    --stop_token="<|endoftext|>" \
    --num_return_sequences=${RETURN_SEQUENCES} \
    --p=${NUCLEUS_BUDGET} \
    --seed=${SEED} \
    --multi_prompt_file_name=${INPUT_PROMPT_FILE}  \
    --output_gen_file_name=${GEN_FILE} \
    --model_name_or_path=${MODEL_DIR} \

