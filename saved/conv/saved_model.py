import torch.nn as nn
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
