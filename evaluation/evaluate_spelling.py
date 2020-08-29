import sys
import argparse
import json
import os
import io
from pprint import pprint
import nltk
import numpy as np
import math
import pkg_resources
from symspellpy import SymSpell, Verbosity
import time
from tqdm import tqdm
from string import punctuation
import re

print("loading symspell dictionary and precalculating edits...")
sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=10)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def load_data(continuation_file):
    print("Reading lines...")
    continuations = []
    f = open(continuation_file, 'r')
    lines = f.readlines()
    for line in lines:
        conts = line.split("<CAND_SEP>")
        for cont in conts:
            if cont != "<blank>":
                continuations.append(cont)
    return continuations


def spellcheck(text):
    misspell_count = 0
    my_punctuation = punctuation.replace("'", "")
    clean_text = text.translate(str.maketrans('', '', my_punctuation))
    clean_text = re.sub("[^a-zA-Z'']+", ' ', clean_text)
    clean_text = re.sub(' +', ' ', clean_text)
    clean_text = clean_text.strip()
    words = clean_text.split()
    len_text = len(words)
    distance_sum = 0
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=5, include_unknown=True)
        for suggestion in suggestions:
            if suggestion.distance > 0:
                distance_sum += suggestion.distance
                misspell_count += 1
    return misspell_count, distance_sum


def evaluate_spelling(continuations):
    misspell_lst = []
    distance_lst = []
    for continuation in tqdm(continuations):
        misspell_count, distance_sum = spellcheck(continuation)
        misspell_lst.append(misspell_count)
        distance_lst.append(distance_sum)
    if len(misspell_lst) != 0 and len(distance_lst) != 0:
        final_result = [sum(misspell_lst)/len(misspell_lst), sum(distance_lst)/len(distance_lst)]
    return final_result, misspell_lst, distance_lst


continuation_file = sys.argv[1]
overall_results = {}
continuations = load_data(continuation_file)
overall_results['spellcheck'], misspell_lst, distance_lst = evaluate_spelling(continuations)

pprint(overall_results)
out_filename = continuation_file + '_spellcheck'
print("Writing spellcheck results to file: ", out_filename)
with open(out_filename, "w") as fout:
    pprint(overall_results, stream=fout)
print("Spellcheck results written to file: ", out_filename)

out_filename_misspell = continuation_file + '_spellcheck_list_misspell'
out_filename_distance = continuation_file + '_spellcheck_list_distance'
print("Writing individual spellcheck results to files")
with open(out_filename_misspell, "w") as fout_misspell:
    fout_misspell.write('\n'.join([str(x) for x in misspell_lst]))
with open(out_filename_distance, "w") as fout_distance:
    fout_distance.write('\n'.join([str(x) for x in distance_lst]))
print("Individual spellcheck results written to files")