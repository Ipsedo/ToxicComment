import torch.nn as nn


class ConvModel(nn.Module):

    def __init__(self, vocab_size, emb_size=50):
        super(ConvModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)

        out_channel_conv1 = 25
        self.seq1 = nn.Sequential(
            nn.Conv1d(emb_size, 25, 5),
            nn.ReLU(),
            nn.Conv1d(25, out_channel_conv1, 3),
            nn.ReLU()
        )
        # TODO rajouter couche

        nb_label = 6
        self.seq2 = nn.Sequential(
            nn.Linear(out_channel_conv1, out_channel_conv1 * 4),
            nn.ReLU(),
            nn.Linear(out_channel_conv1 * 4, nb_label),
            nn.Sigmoid()
        )

    def forward(self, X):
        # TODO verifier enchainement shape
        out = self.emb(X)
        out = out.permute(0, 2, 1)
        out = self.seq1(out)
        # somme sur tout les mots -> out.size() == (batch, out_channel_conv1)
        out = out.mean(dim=2)
        out = self.seq2(out)
        return out
