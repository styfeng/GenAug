import json
from nltk import word_tokenize
import sys
import math

input_file_lines = open(sys.argv[1]).readlines()

output_json = open(sys.argv[2],"w")

dictionary = {}

for line in input_file_lines:
    words = word_tokenize(line.strip().lower())
    for word in words:
        if word not in dictionary: dictionary[word]=0.0
        dictionary[word] += 1.0

Z = sum(dictionary.values()) + 1e-6 
#Second term is a smoothing value. For unknown words at test time, 1e-6*1e-5/(Z+1e-6) will be their assigned unigram prob [1e5 is assumed to be unknown unknown vocab size] 

for word in dictionary:
    dictionary[word] = math.log(dictionary[word]) - math.log(Z)


json.dump(dictionary,output_json)
output_json.close()
