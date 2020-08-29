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
from transformers import *
from metric_utils import *
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk import word_tokenize


lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
lm_model.eval()

# sentence embedding
embedder = SentenceTransformer('bert-base-nli-mean-tokens')


def ppl_score(sentence):
    #print("Sentence: ", sentence)
    input_ids = torch.tensor(lm_tokenizer.encode(sentence)).unsqueeze(0) 
    with torch.no_grad():
        outputs = lm_model(input_ids, labels=input_ids)
    return math.exp(outputs[0].item())


def TTR_score(sentence):
    word_lst = word_tokenize(sentence)
    clean_word_lst = []

    for word in word_lst:
        clean_word_lst.append(word)

    unique_word_lst = set(clean_word_lst)
    TTR = len(unique_word_lst) / len(clean_word_lst)
    #print("Sentence: ", sentence, " / TTR: ", TTR)
    return TTR
    

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


def evaluate_self_bleu(continuations):
    all_results = []
    for continuation in tqdm(continuations):
        result = get_self_metric_corpus_parallel(continuation)
        all_results.append(result)
    final_result = np.average(all_results)
    return final_result, all_results


def evaluate_ngram_fraction(continuations):
    all_results = []
    for continuation in tqdm(continuations):
        result = get_unique_trigrams([c for c in continuation])
        all_results.append(result)
    final_result = np.average(all_results)
    return final_result, all_results


def evaluate_typetoken_ratio(continuations):
    all_results = []
    for continuation in continuations:
        result = np.average([TTR_score(c) for c in continuation if c != '<blank>' and c != '< e >'])
        all_results.append(result)                   
    final_result = np.average(all_results)
    return final_result, all_results


def evaluate_lm_perplexity(prompts,continuations):
    all_results = []
    for prompt, continuation in tqdm(zip(prompts,continuations)):
        ppl = []
        for c in continuation[:20]:
            if c != '<blank>' and c != '< e >':
                if c.strip()[0] != "'":
                    sentence = prompt.strip() + ' ' + c.strip()
                else:
                    sentence = prompt.strip() + c.strip()
                ppl.append(ppl_score(sentence))
        all_results.append(np.average(ppl))
    final_result = np.average(all_results)
    return final_result, all_results    


def evaluate_rare_words(continuations,unigramDist):
    all_results = []
    for continuation in tqdm(continuations):
        mean_log_unigram_prob_gold = 0.0
        l_gold = 0
        for candidate in continuation:
            for c in word_tokenize(candidate):
                l_gold += 1
                if c in unigramDist:
                    mean_log_unigram_prob_gold += unigramDist[c] 
                else:
                    mean_log_unigram_prob_gold += -20.0

        mean_log_unigram_prob_gold/=l_gold
        all_results.append(mean_log_unigram_prob_gold)
    final_result = np.average(all_results)
    return final_result, all_results 


def evaluate_bert_prompt(prompts, continuations, aggregate='max'):
    all_results = []
    for prompt, continuation in tqdm(zip(prompts,continuations)):
        prompt_emb = embedder.encode([prompt])
        cont_emb = embedder.encode(continuation)
        distances = np.average([scipy.spatial.distance.cdist(pemb.reshape(1,-1), cemb.reshape(1,-1), "cosine")[0] for pemb,cemb in zip(prompt_emb, cont_emb)])
        all_results.append(distances)
    final_result = np.average(all_results)
    return final_result, all_results 


def evaluate(metric, prompts, continuations):
    results = {}
    all_results = []
    
    if metric == 'self_bleu':
        results['self_bleu'], all_results = evaluate_self_bleu(continuations)
    
    elif metric == 'ngram_fraction':
        results['ngram_fraction'], all_results = evaluate_ngram_fraction(continuations)

    elif metric == 'typetoken_ratio':
        results['typetoken_ratio'], all_results = evaluate_typetoken_ratio(continuations)
    
    elif metric == 'perplexity':
        results['perplexity'], all_results = evaluate_lm_perplexity(prompts,continuations)
 
    elif metric == 'rare_words':
        results['rare_words'], all_results = evaluate_rare_words(continuations,json.load(open(args.unigramDist)))        

    elif metric == "bert_prompt":
        results['bert_prompt'], all_results = evaluate_bert_prompt(prompts,continuations,aggregate='max')
    
    else:
        print('Metric not found:', metric)
        sys.exit(1)
    
    return results, all_results


def main(args):
    # load dataset
    prompts, continuations = load_data(
        args.prompt_file,
        args.continuation_file)

    # set target metrics
    target_metrics = []
    if len(args.metrics) == 1 and args.metrics[0] == 'all':
        target_metrics = [
            'self_bleu', 'ngram_fraction',
            'typetoken_ratio','rare_words',
            'bert_prompt','perplexity'
        ]
    else:
        target_metrics = args.metrics
    print('Target metrics: {}'.format(target_metrics))

    # evaluate (each metric)
    for tm in target_metrics:
        print("Evaluating: {}".format(tm))
        results, all_results = evaluate(tm, prompts, continuations)
        print(results)
        
        # write average metric results to file
        out_filename = args.continuation_file + '_' + tm
        print("Writing results for {} to file: {}".format(tm, out_filename))
        with open(out_filename, "w") as fout:
            fout.write('\n'.join([str(x) for x in results]))
        print("Results for {} written to file: {}".format(tm, out_filename))
        
        # write individual metric results (for every input prompt) to file
        out_filename_all = out_filename + '_list'
        print("Writing individual results for {} to file: {}".format(tm, out_filename_all))
        with open(out_filename_all, "w") as fout_all:
            fout_all.write('\n'.join([str(x) for x in all_results]))
        print("Individual results for {} written to file: {}".format(tm, out_filename_all))    
        
    # summarize all metric results
    pprint(results)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", default=None, type=str, required=True)
    parser.add_argument("--continuation_file", default=None, type=str, required=True)
    parser.add_argument('--metrics', nargs='+', type=str)
    parser.add_argument("--unigramDist", default=None, type=str, required=False) #json file containing unigram frequency distribution for rare words metric
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    pprint(args.__dict__)
    main(args)
