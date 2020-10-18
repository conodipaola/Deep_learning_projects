import torch.nn as nn
import torch

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()
        
        # hard assignment of the number of layers to 2
        self.n_layers = 2
        self.hidden_dim = hidden_dim
        
        # hard-coded dropout value to 0.5
        self.drop_prob = 0.5

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.n_layers, dropout = self.drop_prob)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        # new line where dimension of the tokenise vector and batch size are obtained from the input data
        pad, batch_size = reviews.size()
        
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        
        # new line where contiguous wrapping is applied because of number of layers more than 1
        lstm_out = lstm_out.contiguous().view(pad, batch_size, self.hidden_dim)
        
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        
        # hidden state added as returned value
        return self.sig(out.squeeze())