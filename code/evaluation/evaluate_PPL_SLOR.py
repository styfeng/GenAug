import argparse
import json
import glob
from pprint import pprint
import os
import coloredlogs, logging
import scipy
import math
import numpy as np
import math
from math import log
import torch
import time
from transformers import *
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# paths to evaluation model
lm_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(os.path.abspath('..'),'models/gpt2_full/checkpoint-4000000'))
lm_model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.abspath('..'),'models/gpt2_full/checkpoint-4000000'))
lm_model.eval()


def load_data(prompt_file, continuation_file, unigram_file):
    print("Reading lines...")
    prompts = []
    prompt_f = open(prompt_file, 'r')
    prompt_lines = prompt_f.readlines()
    for prompt in prompt_lines:
        prompts.append(prompt.strip('\n').strip('\ufeff')) 
    continuations = []
    f = open(continuation_file, 'r')
    lines = f.readlines()
    for cs in lines:
        conts = cs.strip('\n').strip('\ufeff').split(" <CAND_SEP> ")
        continuations.append(conts)   
    unigram_dict = {}
    f2 = open(unigram_file, 'r')
    uni_lines = f2.readlines()
    for line in uni_lines:
        elements = line.split('\t')
        unigram_dict[elements[0].strip()] = int(elements[1].strip())
    total_freq = 0
    for k,v in unigram_dict.items():
        total_freq += v
    print("Total_freq: ", total_freq)
    for k,v in unigram_dict.items():
        unigram_dict[k] = v/(total_freq+1)
    counter = 0
    for k,v, in unigram_dict.items():
        if counter < 10:
            print(k, ' | ', v)
        counter += 1
    unknown_value = 0.0001/(total_freq+1)
    print("Unknown value: ", unknown_value)
    assert len(prompts) == len(continuations)
    logger.info('Loaded: %d', len(prompts))
    return prompts, continuations, unigram_dict, unknown_value


def SLOR_score(sentence, unigram_dict, unknown_value):
    tokenized = lm_tokenizer.tokenize(sentence)
    num_tokens = len(tokenized)
    input_ids = torch.tensor(lm_tokenizer.encode(sentence)).unsqueeze(0) 
    with torch.no_grad():
        outputs = lm_model(input_ids, labels=input_ids)
    loss = outputs[0].item()
    ppl = math.exp(loss)
    loss_sum = 0
    for token in tokenized:
        if token in unigram_dict.keys():
            loss_sum += math.log(unigram_dict[token])
        else:
            loss_sum += math.log(unknown_value)
    SLOR = (-loss*num_tokens - loss_sum)/num_tokens
    return [ppl, SLOR]


def evaluate_SLOR(prompts, continuations, unigram_dict, unknown_value):
    final_results = {}
    ppl_results = []
    SLOR_results = []
    for prompt, continuation in tqdm(zip(prompts,continuations)):
        SLOR = []
        for c in continuation[:20]:
            if c != '<blank>' and c != '< e >':
                if c.strip()[0] != "'":
                    sentence = prompt.strip() + ' ' + c.strip()
                else:
                    sentence = prompt.strip() + c.strip()
                SLOR.append(SLOR_score(sentence, unigram_dict, unknown_value))
        ppl_result = np.average([x[0] for x in SLOR])
        SLOR_result = np.average([x[1] for x in SLOR])
        ppl_results.append(ppl_result)
        SLOR_results.append(SLOR_result)
    final_results['PPL'] = np.average(ppl_results)
    final_results['SLOR'] = np.average(SLOR_results)
    return final_results, ppl_results, SLOR_results


prompt_file = sys.argv[1]
continuation_file = sys.argv[2]
unigram_file = sys.argv[3]
overall_results = {}
prompts, continuations, unigram_dict, unknown_value = load_data(prompt_file, continuation_file, unigram_file)
overall_results['SLOR'], ppl_results, SLOR_results = evaluate_SLOR(prompts, continuations, unigram_dict, unknown_value)

pprint(overall_results)
out_filename = continuation_file + '_SLOR'
print("Writing SLOR results to file: ", out_filename)
with open(out_filename, "w") as fout:
    pprint(overall_results, stream=fout)
print("SLOR results written to file: ", out_filename)

out_filename_ppl = continuation_file + '_SLOR_list_ppl'
out_filename_SLOR = continuation_file + '_SLOR_list_SLOR'
print("Writing individual SLOR results to files")
with open(out_filename_ppl, "w") as fout1:
    fout1.write('\n'.join([str(p) for p in ppl_results]))
with open(out_filename_SLOR, "w") as fout2:
    fout2.write('\n'.join([str(s) for s in SLOR_results]))
print("Individual SLOR results written to files")