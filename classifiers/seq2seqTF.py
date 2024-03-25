import torch
from torch.nn import Transformer, Embedding
import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim=200, output_dim=20, nhead=10, nhid=200, nlayers=2, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = Embedding(input_dim, nhid)
        self.transformer = Transformer(d_model=nhid, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dim_feedforward=nhid, dropout=dropout)
        self.generator = nn.Linear(nhid, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src).permute(1, 0, 2)  # Shape: [S, N, E]
        tgt = self.embedding(tgt).permute(1, 0, 2)  # Shape: [T, N, E]
        output = self.transformer(src, tgt)
        return self.generator(output)
