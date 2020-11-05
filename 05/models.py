import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)

	# Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return output

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        ######

        # Section 5.0 YOUR CODE HERE

        ######


    def forward(self, x, lengths=None):
        ######

        # Section 5.0 YOUR CODE HERE

        ######

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        ######

        # Section 6.0 YOUR CODE HERE

        ######

    def forward(self, x, lengths=None):

        ######

        # Section 6.0 YOUR CODE HERE

        ######
