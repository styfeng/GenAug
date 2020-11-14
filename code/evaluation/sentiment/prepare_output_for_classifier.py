import sys
from io import open
import csv

input_file_name = sys.argv[1]
class_file_name = sys.argv[2]

input_lines = [line.strip().split("\t") for line in open(input_file_name).readlines()]

out_header = ["index","genre","filename","year","old_index","source1","source2","sentence1","sentence2","score"]
class_file = open(class_file_name,'w')
writer = csv.DictWriter(class_file, fieldnames=out_header,delimiter="\t")
writer.writeheader()
class_file_elems = []


for index,example in enumerate(input_lines):
    class_file_elem = {}
    class_file_elem["index"] = index
    class_file_elem["genre"] = "NONE"
    class_file_elem["filename"] = "NONE"
    class_file_elem["year"] = "NONE"
    class_file_elem["old_index"] = "NONE"
    class_file_elem["source1"] = "NONE"
    class_file_elem["source2"] = "NONE"
    class_file_elem["sentence1"] = example[0]    
    class_file_elem["sentence2"] = "[SEP]"
    try:
        class_file_elem["score"] = example[1]
    except IndexError:
        #If there's no label in the file just give 0.5. We're just gonna run inference on this anyway, this labels' just a placeholder
        class_file_elem["score"] = "0.5"
    class_file_elems.append(class_file_elem)

for elem in class_file_elems:
    writer.writerow(elem)
class_file.close()

