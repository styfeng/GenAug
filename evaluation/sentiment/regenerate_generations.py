import sys
import json

#Convert json to txt (each batch of 100 sentiment scores is on each line separated by a tab)

generation_preds = json.load(open(sys.argv[1]))
number_of_lines = sys.argv[2] # number of prompts (e.g. 500 or 2000)

new_generation_preds = []
counter = 0
while counter <= int(number_of_lines):
    new_generation_preds.append([str(x) for x in generation_preds[counter*100:(counter+1)*100]]) # if appropriate, replace 100 with number of continuations generated per prompt
    counter += 1

f = open(sys.argv[3],"w")
f.write('\n'.join(['\t'.join(x) for x in new_generation_preds]))
f.close()

print("Lines written to file")