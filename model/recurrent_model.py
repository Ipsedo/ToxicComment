import torch as th
import torch.nn as nn


class RecurrentModel(nn.Module):
    def __init__(self, vocab_size, batch_size, seq_length, emb_size=16, hidden_size=32):
        super(RecurrentModel, self).__init__()

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.maxpool = nn.MaxPool1d(seq_length)

        self.pred = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size * 4, 6),
                                  nn.Sigmoid())

        self.fst_h = th.randn(1, batch_size, hidden_size)
        self.fst_c = th.randn(1, batch_size, hidden_size)

    def forward(self, x):
        out = self.emb(x)

        batch_size = x.size(0)
        h, c = self.fst_h, self.fst_c
        if x.is_cuda:
            h, c = h.cuda(), c.cuda()

        out, _ = self.lstm(out, (h[:, :batch_size, :], c[:, :batch_size, :]))

        out = out.permute(0, 2, 1)
        out = self.maxpool(out)
        out = out.squeeze(2)

        out = self.pred(out)
        return out