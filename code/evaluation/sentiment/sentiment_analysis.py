import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import math
from tqdm import tqdm
from pprint import pprint


def load_data(prompt_file, continuation_file):
    print("Reading lines...")
    prompts = []
    prompt_f = open(prompt_file, 'r')
    prompt_lines = prompt_f.readlines()
    for prompt in prompt_lines:
        prompts.append(float(prompt.strip('\n').strip('\ufeff'))) 
    continuations = []
    cont_f = open(continuation_file, 'r')
    cont_lines = cont_f.readlines()
    for cs in cont_lines:
        conts = cs.strip('\n').strip('\ufeff').split("\t")
        continuations.append([float(c) for c in conts])   
    assert len(prompts) == len(continuations)
    print('Loaded: %d', len(prompts))
    return prompts, continuations

    
def evaluate_sent(prompts, continuations):
    cont_stds = []
    prompt_cont_diffs = []
    for prompt, continuation in tqdm(zip(prompts, continuations)):
        filtered_cont_scores = [c for c in continuation if c != 0.15861375629901886] #0.15861375629901886 corresponds to score for '<blank>', change as appropriate
        inner_prompt_cont_diffs = []
        if len(filtered_cont_scores) != 0:
            cont_score_std = np.std(filtered_cont_scores)
            cont_stds.append(cont_score_std)
            for cs in filtered_cont_scores:
                inner_prompt_cont_diffs.append(abs(cs - prompt))
            prompt_cont_diffs.append(np.average(inner_prompt_cont_diffs))
     
    avg_cont_std = np.average(cont_stds)
    print("Avg cont std: ", avg_cont_std) 
    avg_prompt_cont_diff = np.average(prompt_cont_diffs) 
    print("Avg prompt cont diff: ", avg_prompt_cont_diff)
  
    return avg_cont_std, avg_prompt_cont_diff, cont_stds, prompt_cont_diffs


prompt_file = sys.argv[1]
continuation_file = sys.argv[2]

overall_results = {}
prompts, continuations = load_data(prompt_file, continuation_file)
avg_cont_std, avg_prompt_cont_diff, cont_stds, prompt_cont_diffs = evaluate_sent(prompts, continuations)
overall_results['avg_cont_std'] = avg_cont_std
overall_results['avg_prompt_cont_diff'] = avg_prompt_cont_diff

pprint(overall_results)
out_filename = continuation_file + '_analysis'
print("Writing metrics to file: ", out_filename)
with open(out_filename, "w") as fout:
    pprint(overall_results, stream=fout)
print("Metrics written to file: ", out_filename)

out_filename_std = continuation_file + '_analysis_list_std'
out_filename_diff = continuation_file + '_analysis_list_diff'
print("Writing individual results to files")
with open(out_filename_std, "w") as fout1:
    fout1.write('\n'.join([str(s) for s in cont_stds]))
with open(out_filename_diff, "w") as fout2:
    fout2.write('\n'.join([str(d) for d in prompt_cont_diffs]))
print("Individual results written to files")