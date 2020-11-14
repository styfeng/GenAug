import sys
import argparse
import json
import glob
from pprint import pprint
import os
import scipy
import math
import numpy as np
import math
from math import log
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import OrderedDict
import torch
import time
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk import tokenize
from nltk import word_tokenize
from statistics import mean
import bert_score
from bert_score import BERTScorer


def load_data(prompt_file, continuation_file):
    print("Reading lines...")
    prompts = []
    prompt_f = open(prompt_file, 'r')
    prompt_lines = prompt_f.readlines()
    for prompt in prompt_lines:
        prompts.append(prompt.strip('\n').strip('\ufeff')) 
    continuations = []
    cont_f = open(continuation_file, 'r')
    cont_lines = cont_f.readlines()
    for cs in cont_lines:
        conts = cs.strip('\n').strip('\ufeff').split(" <CAND_SEP> ")
        continuations.append(conts)   
    assert len(prompts) == len(continuations)
    print('Loaded: {}'.format(len(prompts)))
    return prompts, continuations


def create_scorer():
    # Create scorer object for passing to get_bert_score
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type='roberta-base')
    return scorer


def get_bert_score(hyp,ref,scorer):
    # hyp: hypothesis ref: reference scorer: Already created BERT Score object
    # Returns F1: BERT-Score F1 between hypothesis and reference
    # Note: Some settings need to be done while creating the scorer object e.g whether to normalize by baseline or not, or which BERT model to use
    hyp = hyp.strip()
    ref = ref.strip()
    P, R, F1 = scorer.score([hyp,],[ref,])
    F1 = float(F1.data.cpu().numpy())
    return F1


def evaluate_BPRO(prompts, continuations, scorer):
    all_results = []
    for prompt, continuation in tqdm(zip(prompts,continuations)):
        scores = []
        for cont in continuation:
            bertscore = get_bert_score(cont,prompt,scorer)
            scores.append(bertscore)
        avg_score = np.average(scores)    
        all_results.append(avg_score)
    final_result = np.average(all_results)
    return final_result, all_results 


prompt_file = sys.argv[1]
continuation_file = sys.argv[2]
overall_results = {}
prompts, continuations = load_data(prompt_file, continuation_file)
scorer = create_scorer()
overall_results['BPRO'], all_results = evaluate_BPRO(prompts, continuations, scorer)

#write overall average results to file
pprint(overall_results)
out_filename = continuation_file + '_BPRO'
print("Writing final BPRO result to file: ", out_filename)
with open(out_filename, "w") as fout:
    pprint(overall_results, stream=fout)
print("Final BPRO result written to file: ", out_filename)

#write individual results to file (for statistical significance purposes later)
out_filename_lst = continuation_file + '_BPRO_list'
print("Writing individual BPRO results to file")
with open(out_filename_lst, "w") as fout_lst:
    fout_lst.write('\n'.join([str(b) for b in all_results]))   
print("Individual BPRO results written to file")