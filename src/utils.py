import torch
import subprocess

def predict(logit):
    return logit > 0

def accuracy(logit, label):
    corrects = ((logit > 0) == label.type(torch.uint8)).sum()
    return float(corrects) / len(label)

def summarize_info(loss, acc):
    return {
        'loss': float(loss),
        'acc': float(acc)
    }

def summarize_score(MAP, MRR):
    return {
        'MAP': float(MAP),
        'MRR': float(MRR)
    }

def calc_score(ans):
    with open('../evaltool/ans.txt', 'w') as f:
        [f.write(str(x) + "\n") for x in ans]

    p = subprocess.Popen("validation.bat", stdin=subprocess.DEVNULL)
    p.communicate()

    lines = []
    with open('../evaltool/result.txt', 'r') as f:
        for line in f:
            lines += [line]
    MAP = lines[0].split('\t') 
    MRR = lines[1].split('\t')
    return float(MAP[1]), float(MRR[1])