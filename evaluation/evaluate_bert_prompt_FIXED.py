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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk import tokenize
from nltk import word_tokenize

# sentence embedding
embedder = SentenceTransformer('bert-base-nli-mean-tokens')


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


def evaluate_bert_prompt(prompts, continuations, aggregate='max'):
    all_results = []
    for prompt, continuation in tqdm(zip(prompts,continuations)):
        prompt_emb = embedder.encode([prompt])
        cont_emb = embedder.encode(continuation)
        distances = []
        for cemb in cont_emb[:20]:
            distance = scipy.spatial.distance.cdist(prompt_emb[0].reshape(1,-1), cemb.reshape(1,-1), "cosine")[0]
            distances.append(distance)
        avg_distance = np.average(distances)    
        all_results.append(avg_distance)
    final_result = np.average(all_results)
    return final_result, all_results 


prompt_file = sys.argv[1]
continuation_file = sys.argv[2]
overall_results = {}
prompts, continuations = load_data(prompt_file, continuation_file)
overall_results['BERT_prompt_FIXED'], all_results = evaluate_bert_prompt(prompts, continuations)

#write overall average results to file
pprint(overall_results)
out_filename = continuation_file + '_bert_prompt_FIXED'
print("Writing final BERT_prompt result to file: ", out_filename)
with open(out_filename, "w") as fout:
    pprint(overall_results, stream=fout)
print("Final BERT_prompt result written to file: ", out_filename)

#write individual results to file (for statistical significance purposes later)
out_filename_lst = continuation_file + '_bert_prompt_FIXED_list'
print("Writing individual BERT_prompt results to file")
with open(out_filename_lst, "w") as fout_lst:
    fout_lst.write('\n'.join([str(b) for b in all_results]))   
print("Individual BERT_prompt results written to file")