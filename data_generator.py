import json
import settings as s

in_filename = "data/gcc-10M.trace"
out_filename = "data/gcc-10M_seq.trace"

def main():
    '''
    Parse instrctions into code sequence split by branch instruction

    Input:
        original instruction dataset
        e.g. gcc-1K.trace

    Output:
       a json file which has following format,
       the first layer is a list and the second layer is a dict which has two items:
       data: code sequence
       label: branch (1) or not (0).
       
       The attributes of a instruction is split by whitespace
       
        [
            {
                data  : list of string
                label : 1 or 0
            }
        ]
    '''
    fin  = open(in_filename, mode='r')
    fout = open(out_filename, mode='w')

    data = []
    seq = []
    for line in fin:
        seq.append(line.strip())
        inst = line.split()
        if inst[s.BRANCH] != '-':
            label = int(inst[s.BRANCH] == 'T')
            data.append({"data" : seq, "label" : label}) 
            seq = []
    
    json.dump(data, fout, indent=1)

if __name__ == "__main__":
    main()
