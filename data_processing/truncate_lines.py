# -*- coding: utf-8 -*-
import io, json, os, collections, pprint, time
import re
from string import punctuation
import unicodedata
import random
import numpy as np
import json
from collections import defaultdict


def truncate_lines(input_file, output_file, proportion):
    print("Reading lines from file")
    f1 = open(input_file, "r")
    input_data = f1.readlines()
    print("Number of input lines: ", len(input_data))
    f1.close()
    
    truncated_lines = []
    input_lens = []
    output_lens = []
    for line in input_data:
        words = line.split()
        input_lens.append(len(words))
        truncated_words = words[:int(round(len(words)*proportion))]
        output_lens.append(len(truncated_words))
        truncated_line = " ".join(truncated_words)
        truncated_lines.append(truncated_line)
    print("Number of truncated lines: ", len(truncated_lines))
    print("Average length of input lines: ", np.average(input_lens))
    print("Average length of output lines: ", np.average(output_lens))
    
    print("Writing truncated lines to file...")
    f2 = open(output_file, "w", encoding='utf-8')
    f2.write('\n'.join(truncated_lines))
    f2.close()
    print("Truncated lines written to file!")

truncate_lines("yelp_test.txt","yelp_test4.txt",0.75)