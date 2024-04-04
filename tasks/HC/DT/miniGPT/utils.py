'''
Author: Dylan Li dylan.h.li@outlook.com
Date: 2024-03-26 15:23:17
LastEditors: Dylan Li dylan.h.li@outlook.com
LastEditTime: 2024-03-30 22:45:30
FilePath: /HC3D_simulator/tasks/HC/DT/utils.py
Description: 

Copyright (c) 2024 by Heng Li, All Rights Reserved. 
'''
"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def sample_from_logits(logits, temperature=1.0, sample=False, top_k=None):
    """
    Given a sequence of logits, predict the next token in the sequence,
    feeding the predictions back into the model each time. This function
    assumes that the logits are already produced by the model and are
    passed directly to it.
    """
    # Assuming logits are of shape (b, t, v) where b is batch size, t is sequence length, and v is vocabulary size
    # We only need the last logits for the next token prediction
    logits = logits[:, -1, :] / temperature
    
    # Optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    # Apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample from the distribution or take the most likely
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)
    
    # Return the index of the sampled token
    return ix