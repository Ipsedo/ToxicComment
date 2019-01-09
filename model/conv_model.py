import torch.nn as nn


class ConvModel(nn.Module):

    def __init__(self, vocab_size, emb_size=50):
        super(ConvModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)

        out_channel_conv1 = 25
        kernel_size = 5
        self.conv1 = nn.Conv1d(emb_size, out_channel_conv1, kernel_size)
        self.rel1 = nn.ReLU()

        # TODO rajouter couche

        nb_label = 6
        self.lin1 = nn.Linear(out_channel_conv1, nb_label)
        self.sig1 = nn.Sigmoid()

    def forward(self, X):
        # TODO verifier enchainement shape
        out = self.emb(X)
        out = out.permute(1, 0).unsqueeze(0)
        out = self.conv1(out)
        out = self.rel1(out)
        # somme sur tout les mots -> out.size() == (batch, out_channel_conv1)
        out = out.sum(dim=2)
        out = self.lin1(out)
        out = self.sig1(out)
        return out
