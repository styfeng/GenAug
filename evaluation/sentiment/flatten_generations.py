import json
import io
import sys


prompt_file = open(sys.argv[1],"r")
cont_file = open(sys.argv[2],"r")
out_file = open(sys.argv[3],"w")

combined_lines = []

for prompt, cont in zip(prompt_file, cont_file):
    continuations = cont.split(" <CAND_SEP> ")
    for c in continuations:
        out_file.write(prompt.strip() + ' ' + c.strip('\n')+"\n")

out_file.close()