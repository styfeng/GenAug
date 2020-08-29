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
import spacy
#Note: run pip install spacy (v. 2.2.4) and python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm") 
complete_POS_lst = ['POS', 'PUNCT', 'SYM', 'ADJ', 'CCONJ', 'NUM', 'DET', 'ADV', 'ADP', 'X', 'VERB', 'NOUN', 'PROPN', 'PART', 'INTJ', 'SPACE', 'PRON','AUX','CONJ','SCONJ','LITERAL','entities']

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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


def get_POS(continuation):
    POS_dict = OrderedDict()
    for key in complete_POS_lst:
        POS_dict[key] = [0,0]
    words = continuation.split()
    for x in words:
        for char in x:
            if char.isnumeric():
                POS_dict['LITERAL'][0] += 1
                POS_dict['LITERAL'][1] += 1
                break
    tokenized_cont = nlp(continuation)
    total_len = len(tokenized_cont)
    for ent in tokenized_cont.ents:
        POS_dict['entities'][0] += 1
        POS_dict['entities'][1] += len(ent)
    for token in tokenized_cont:
        if token.pos_ in POS_dict.keys():
            POS_dict[token.pos_][0] += 1
            POS_dict[token.pos_][1] += 1
        if token.tag_ in POS_dict.keys():
            POS_dict[token.tag_][0] += 1
            POS_dict[token.tag_][1] += 1            
    for key, value in POS_dict.items():
        POS_dict[key][1] = value[1]/total_len
    return POS_dict    


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
    logger.info('Loaded: %d', len(prompts))
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


def evaluate_blank_outputs(continuations):
    all_results = []
    for continuation in tqdm(continuations):
        result = (continuation.count('<blank>') + continuation.count('< e >'))/len(continuation)
        all_results.append(result)
    final_result = np.average(all_results)
    return final_result, all_results


def evaluate_lengths(continuations):
    all_results = []
    for continuation in tqdm(continuations):
        result = [len(word_tokenize(c)) if c != '<blank>' and c != '< e >' else 0 for c in continuation]
        all_results.append(np.average(result))        
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


def evaluate_POS(continuations):
    all_results = []
    for continuation in tqdm(continuations):
        POS = [get_POS(c) for c in continuation if c != '<blank>' and c != '< e >']
        if len(POS) != 0:
            result = OrderedDict()
            for key in complete_POS_lst:
                key_sum_0 = sum(c[key][0] for c in POS)
                key_sum_1 = sum(c[key][1] for c in POS)
                result[key] = [key_sum_0/len(POS), key_sum_1/len(POS)]
            all_results.append(result)
    
    final_results = OrderedDict()
    for key in complete_POS_lst:
        key_sum_0 = sum(c[key][0] for c in all_results)
        key_sum_1 = sum(c[key][1] for c in all_results)
        final_results[key] = [key_sum_0/len(all_results), key_sum_1/len(all_results)]         
        
    return final_results
    

def evaluate(metric, prompts, continuations):
    results = {}
    all_results = []
    
    if metric == 'self_bleu':
        results['self_bleu'], all_results = evaluate_self_bleu(continuations)
    
    elif metric == 'ngram_fraction':
        results['ngram_fraction'], all_results = evaluate_ngram_fraction(continuations)

    elif metric == 'typetoken_ratio':
        results['typetoken_ratio'], all_results = evaluate_typetoken_ratio(continuations)
    
    elif metric == 'blank_outputs':
        results['blank_outputs'], all_results = evaluate_blank_outputs(continuations)
    
    elif metric == 'lengths':
        results['lengths'], all_results = evaluate_lengths(continuations)

    elif metric == 'perplexity':
        results['perplexity'], all_results = evaluate_lm_perplexity(prompts,continuations)
 
    elif metric == 'rare_words':
        results['rare_words'], all_results = evaluate_rare_words(continuations,json.load(open(args.unigramDist)))        

    elif metric == "bert_prompt":
        results['bert_prompt'], all_results = evaluate_bert_prompt(prompts,continuations,aggregate='max')
    
    elif metric == 'POS':
        results['POS'] = evaluate_POS(continuations)
        
    else:
        print('Not implemented yet:', metric)
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
            'typetoken_ratio','blank_outputs','lengths',
            'POS','rare_words','bert_prompt','perplexity'
        ]
    else:
        target_metrics = args.metrics
    logger.info('Target metrics: {}'.format(target_metrics))

    # evaluate
    for tm in target_metrics:
        logger.info("Evaluating: %s",tm)
        results, all_results = evaluate(tm, prompts, continuations)
        print(results)

    # summarize output results
    pprint(results)
    out_filename = args.continuation_file + '_' + '_'.join(target_metrics)
    logger.info("Writing metrics to file: ", out_filename)
    with open(out_filename, "w") as fout:
        pprint(results, stream=fout)
    logger.info("Metrics written to file: ", out_filename)
    
    out_filename_all = out_filename + '_list'
    print("Writing individual results to file")
    with open(out_filename_all, "w") as fout1:
        fout1.write('\n'.join([str(x) for x in all_results]))
    print("Individual results written to file")    


def _parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--prompt_file", default=None, type=str, required=True,help="TBW")
    parser.add_argument("--continuation_file", default=None, type=str, required=True,help="TBW")
    parser.add_argument('--metrics', nargs='+', type=str)
    parser.add_argument("--unigramDist", default=None, type=str, required=False) #json file containing frequency distribution for rare words
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    pprint(args.__dict__)
    main(args)
