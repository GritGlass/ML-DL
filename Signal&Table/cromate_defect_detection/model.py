import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h_repeated)
        return out
    

class SequenceDataset(Dataset):
    def __init__(self,file_paths,features,mode,slide_window=1, seq_len=3):
        self.seq_len = seq_len
        self.samples = []

        if mode=='train':
            for path in file_paths:
                df = pd.read_csv(path)
                df_normal= df[df['class']==0]
                df_normal.reset_index(drop=True,inplace=True)
                label=0
                data = df_normal[features].values.astype('float32')

                for i in range(len(data) - seq_len + slide_window):
                    seq = data[i:i+seq_len]
                    self.samples.append((seq, label))

        elif mode=='test':
            for path in file_paths:
                df = pd.read_csv(path)
                label=df['class'].values.astype('int')
                data = df[features].values.astype('float32')

                for i in range(len(data) - seq_len + slide_window):
                    seq = data[i:i+seq_len]
                    seq_label=label[i:i+seq_len]
                    if np.count_nonzero(seq_label)>int(len(seq_label)/2):
                        cls=1
                    else:
                        cls=0
                    self.samples.append((seq, cls))
        else:
            print("Invalid mode. Use 'train' or 'test'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

