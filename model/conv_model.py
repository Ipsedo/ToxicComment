import torch.nn as nn
import torch as th


class ConvModel(nn.Module):
    def __init__(self, vocab_size, sent_max_len, pad_idx, emb_size=512):
        super(ConvModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        out_channel_conv2 = 128
        out_channel_conv1 = 256
        out_conv1_size = sent_max_len - 5 + 1
        out_conv2_size = out_conv1_size - 3 + 1
        self.seq1 = nn.Sequential(
            nn.Conv1d(emb_size, out_channel_conv1, 5),
            nn.ReLU(),
            nn.Conv1d(out_channel_conv1, out_channel_conv2, 3),
            nn.ReLU(),
            nn.MaxPool1d(out_conv2_size)
        )

        nb_label = 6
        self.seq2 = nn.Sequential(
            nn.Linear(out_channel_conv2, out_channel_conv2 * 4),
            nn.BatchNorm1d(out_channel_conv2 * 4),
            nn.Linear(out_channel_conv2 * 4, nb_label),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.emb(X)
        out = out.permute(0, 2, 1)
        out = self.seq1(out).squeeze(2)
        # somme sur tout les mots -> out.size() == (batch, out_channel_conv1)
        # out = out.mean(dim=2)
        out = self.seq2(out)
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

        nb_label = 6
        self.seq_lin = nn.Sequential(
            nn.Linear(out_channel_conv2 * 2, out_channel_conv2 * 4),
            nn.BatchNorm1d(out_channel_conv2 * 4),
            nn.Linear(out_channel_conv2 * 4, nb_label),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(0, 2, 1)

        out1 = self.seq_conv1(out)
        out2 = self.seq_conv2(out)

        out = th.cat((out1, out2), dim=1).squeeze(2)

        out = self.seq_lin(out)

        return out
