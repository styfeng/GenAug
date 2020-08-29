import sys
import io
import json

#Convert json to txt

gold_preds = json.load(open(sys.argv[1]))
gold_preds_new = [str(x) for x in gold_preds]

f = open(sys.argv[2],"w")
f.write('\n'.join(gold_preds_new))
f.close()

print("Lines written to file")