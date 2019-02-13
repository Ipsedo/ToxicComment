import torch.nn as nn


class ConvModel(nn.Module):

    def __init__(self, vocab_size, sent_max_len, emb_size=16):
        super(ConvModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)

        out_channel_conv1 = 32
        out_conv1_size = sent_max_len - 5 + 1
        out_conv2_size = out_conv1_size - 3 + 1
        self.seq1 = nn.Sequential(
            nn.Conv1d(emb_size, 24, 5),
            nn.ReLU(),
            nn.Conv1d(24, out_channel_conv1, 3),
            nn.ReLU(),
            nn.MaxPool1d(out_conv2_size)
        )

        nb_label = 6
        self.seq2 = nn.Sequential(
            nn.Linear(out_channel_conv1, out_channel_conv1 * 4),
            nn.BatchNorm1d(out_channel_conv1 * 4),
            nn.Linear(out_channel_conv1 * 4, nb_label),
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
