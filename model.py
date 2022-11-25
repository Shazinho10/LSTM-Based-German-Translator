import spacy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import Multi30k
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Encoder(nn.Module):
  def __init__(self, embedding_dim, input_dim, hidden_dim, num_layers):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
    
  def forward(self, x):
    embedding = self.embedding(x)
    output, (hidden, cell) = self.lstm(embedding)
    return hidden.to(device), cell.to(device)

class Decoder(nn.Module):
  def __init__(self, embedding_dim, input_dim, hidden_dim, output_dim, num_layers):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
    self.dense = nn.Linear(hidden_dim, output_dim)

  def forward(self,x, hidden, cell):
    embedding = self.embedding(x)
    output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
    fc = F.relu(self.dense(output))
    return fc.to(device), hidden.to(device), cell.to(device)

class seq2seq(nn.Module):
  print('the parameters needed are embedding_dim, enc_input_dim, dec_input_dim,hidden_dim, output_dim, num_layers')
  def __init__(self, embedding_dim, enc_input_dim, dec_input_dim, 
               hidden_dim, output_dim, num_layers):
    
    super().__init__()

    self.num_layers = num_layers
    self.output_dim = output_dim
    self.embedding_dim = embedding_dim
    
    self.encoder = Encoder(embedding_dim, enc_input_dim, hidden_dim, num_layers)
    self.encoder = self.encoder.to(device)

    self.decoder = Decoder(embedding_dim, dec_input_dim, hidden_dim, 
                           output_dim, num_layers)

    self.decoder = self.decoder.to(device)
  
  def forward(self, src, trg):
    target_len = trg.shape[0]
    batch_size = trg.shape[1]
    output_dim = self.output_dim
    outputs = torch.zeros(target_len, batch_size, output_dim).to(device)

    hidden, cell = self.encoder(src)

    x = trg[0]
    for t in range(1, target_len):
      x = x.unsqueeze(0)
      out, hidden, cell = self.decoder(x, hidden, cell)
      outputs[t] = out
      x = trg[t]
    return outputs

