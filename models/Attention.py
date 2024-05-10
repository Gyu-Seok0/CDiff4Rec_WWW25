import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    
    def __init__(self, item_size, topk, dim_size = 1000):
        super(Attention, self).__init__()

        self.W_Q = nn.Linear(item_size, dim_size)
        self.W_K = nn.Linear(item_size, dim_size)
        #self.W_V = nn.Linear(item_size, item_size)
        self.denominator = np.sqrt(item_size)
        self.softmax = nn.Softmax(dim = 1)
        self.topk = topk
        
        self.init_weights()
            
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.normal_(param, mean=0, std=0.001)
    
    def forward(self, x1, x2 = None, label_map = None, return_score = False):
        
        if x2 == None:
            x2 = x1
        
        Q = self.W_Q(x1)
        K = self.W_K(x2)
        #V = self.W_V(x2) # dtype=torch.float16,
        
        out = (Q @ K.T) / self.denominator
        
        if label_map is not None:
            out = torch.exp(out)
            out = out * label_map
            interactions = torch.sum(label_map, dim = 1, keepdim = True)
            Attention_score = out / (torch.sum(out, dim = 1, keepdim = True) + 1e-8)
            Attention_score = Attention_score * ((interactions / self.topk) + 1e-8)
        else:
            Attention_score = self.softmax(out) # torch.float64
                
        output = torch.spmm(Attention_score, x2.double())
        
        if return_score:
            return output, Attention_score
        else:
            return output