import torch.nn as nn
import torch as th
from math import sqrt


class ConvModel(nn.Module):
    def __init__(self, vocab_size, sent_max_len, pad_idx):
        super(ConvModel, self).__init__()

        emb_size = int(sqrt(vocab_size))
        out_channel_conv1 = emb_size + 32
        out_channel_conv2 = emb_size + 64

        out_conv1_size = sent_max_len - 5 + 1
        out_conv2_size = out_conv1_size - 3 + 1

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(emb_size, out_channel_conv1, 5),
            nn.ReLU(),
            nn.Conv1d(out_channel_conv1, out_channel_conv2, 3),
            nn.ReLU(),
            nn.MaxPool1d(out_conv2_size)
        )

        nb_label = 6
        self.lin_layers = nn.Sequential(
            nn.Linear(out_channel_conv2, out_channel_conv2 * 4),
            nn.BatchNorm1d(out_channel_conv2 * 4),
            nn.Linear(out_channel_conv2 * 4, nb_label),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.emb(X)
        out = out.permute(0, 2, 1)
        out = self.conv_layers(out).squeeze(2)
        out = self.lin_layers(out)
        return out


class ConvModel2(nn.Module):
    def __init__(self, vocab_size, sent_max_len, pad_idx, emb_size=512):
        super(ConvModel2, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        out_channel_conv2 = 256

        self.seq_conv1 = nn.Sequential(
            nn.Conv1d(emb_size, out_channel_conv2, 3),
            nn.MaxPool1d(sent_max_len - 3 + 1),
            nn.ReLU()
        )

        self.seq_conv2 = nn.Sequential(
            nn.Conv1d(emb_size, out_channel_conv2, 5),
            nn.MaxPool1d(sent_max_len - 5 + 1),
            nn.ReLU()
        )

        self.seq_conv3 = nn.Sequential(
            nn.Conv1d(emb_size, out_channel_conv2, 7),
            nn.MaxPool1d(sent_max_len - 7 + 1),
            nn.ReLU()
        )

        nb_label = 6
        self.seq_lin = nn.Sequential(
            nn.Linear(out_channel_conv2 * 3, out_channel_conv2 * 6),
            nn.BatchNorm1d(out_channel_conv2 * 6),
            nn.Linear(out_channel_conv2 * 6, nb_label),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)

        out1 = self.seq_conv1(out)
        out2 = self.seq_conv2(out)
        out3 = self.seq_conv3(out)

        out = th.cat((out1, out2, out3), dim=1).squeeze(2)

        out = self.seq_lin(out)

        return out


class ConvModel3(nn.Module):
    def __init__(self, vocab_size, sent_max_len, pad_idx):
        super(ConvModel3, self).__init__()

        emb_size = int(sqrt(vocab_size))
        out_channel_conv1 = emb_size - int(emb_size * 1 / 5)
        out_channel_conv2 = out_channel_conv1 - 16
        out_channel_conv3 = out_channel_conv2 - 32

        out_conv1_size = sent_max_len - 7 + 1
        out_conv2_size = out_conv1_size - 5 + 1
        out_conv3_size = out_conv2_size - 3 + 1

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        self.seq1 = nn.Sequential(
            nn.Conv1d(emb_size, out_channel_conv1, 7),
            nn.ReLU(),
            nn.Conv1d(out_channel_conv1, out_channel_conv2, 5),
            nn.ReLU(),
            nn.Conv1d(out_channel_conv2, out_channel_conv3, 3),
            nn.ReLU(),
            nn.MaxPool1d(out_conv3_size)
        )

        nb_label = 6
        self.seq2 = nn.Sequential(
            nn.Linear(out_channel_conv3, out_channel_conv3 * 4),
            nn.BatchNorm1d(out_channel_conv3 * 4),
            nn.Linear(out_channel_conv3 * 4, nb_label),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.emb(X)
        out = out.permute(0, 2, 1)
        out = self.seq1(out).squeeze(2)
        out = self.seq2(out)
        return out
