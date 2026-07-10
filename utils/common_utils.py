import os
import sys
import random
import torch
import numpy as np

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Info] Random seed set to: {seed}")

class Logger(object):
    
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
       
        self.terminal.write(message)
        
        self.log.write(message)
        
        self.log.flush()

    def flush(self):
        
        self.terminal.flush()
        self.log.flush()