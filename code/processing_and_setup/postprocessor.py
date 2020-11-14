import sys
import re

def strip_excessive_exclamation(line):
    rx = re.compile(r'(!)\1{4,}$')
    match = rx.search(line)
    if match == None:
        return line
    else:
        #print(line)
        #print( line[:match.span()[0]])
        return line[:match.span()[0]]

def strip_optional_ending(line):
    rx = re.compile(r'( < e >)?$')
    match = rx.search(line)
    if match == None:
        return line
    else:
        return line[:match.span()[0]]



def postpro_file(in_file_name,out_file_name,multiple_sequences_per_line=True):

    in_file_lines = open(in_file_name,encoding='utf-8').readlines()

    out_file = open(out_file_name,"w",encoding='utf-8')
    for line in in_file_lines:
        line = line.strip()
        if multiple_sequences_per_line:
            continuations = line.split("<CAND_SEP>")
            new_continuations = []
            for continuation in continuations:
                new_continuation = strip_excessive_exclamation(strip_optional_ending(continuation.strip()))
                if len(new_continuation.strip()) == 0: new_continuation = "<blank>"
                new_continuations.append(new_continuation)
            line = " <CAND_SEP> ".join(new_continuations)
        else:
            line = strip_excessive_exclamation(line)

        out_file.write(line+"\n")
    out_file.close()
    print("Finished writing lines to file")
    

if __name__ == "__main__":
    #x = "ohh!!!!!!!hxxx!!!!!!! < e >"
    #x = strip_optional_ending(x)
    #print(x)
    #x = strip_excessive_exclamation(x)
    #print(x)
    
    postpro_file(sys.argv[1],sys.argv[1]+".postpro")

