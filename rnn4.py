import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 4
NUM_LAYERS = 1
grad_clipping = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,embeddings):

        super(EncoderRNN, self).__init__()
        self.embedding_dimension = embeddings.shape[1]
        self.num_layers = NUM_LAYERS
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.embedding_dimension).from_pretrained(embeddings,freeze=False)
        self.gru = nn.GRU(self.embedding_dimension, self.embedding_dimension,num_layers = NUM_LAYERS)
        self.embedding_to_RGB_m = nn.Linear(self.embedding_dimension,hidden_size) # variances and mus
        self.embedding_to_RGB_v = nn.Linear(self.embedding_dimension,hidden_size*2) # covariance matrix is symmetric

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        if hidden.requires_grad:
            hidden.register_hook(lambda x: x.clamp(min=-grad_clipping, max=grad_clipping))
        output_m = torch.squeeze(self.embedding_to_RGB_m(output))
        f = torch.squeeze(self.embedding_to_RGB_v(output))
        #cov matrix
        #f = torch.Tensor([ [f[0], 0 , 0], [f[1],f[2] ,0], [f[3],f[4],f[5]] ])
        #Random lower triangular => covariance when multiplied f * f'
        return self.sigmoid(output_m).view(-1,1), f, hidden

    def initHidden(self):
        return torch.zeros(NUM_LAYERS, 1, self.embedding_dimension, device=device) #return torch.zeros(NUM_LAYERS, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embeddings, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dimension = embeddings.shape[1]
        self.RGB_to_embedding = nn.Linear(self.hidden_size,self.embedding_dimension)
        self.embedding = nn.Embedding(output_size, self.embedding_dimension).from_pretrained(embeddings,freeze=False) #=
        self.gru = nn.GRU(self.embedding_dimension, self.embedding_dimension,num_layers=NUM_LAYERS)
        self.out = nn.Linear(self.embedding_dimension, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        hidden = (torch.squeeze(hidden)).unsqueeze(1)
        if NUM_LAYERS == 1 :
            hidden = hidden.view([1,300])
            hidden = hidden.unsqueeze(0)
        output, hidden = self.gru(output, hidden)
        if hidden.requires_grad:
            hidden.register_hook(lambda x: x.clamp(min=-grad_clipping, max=grad_clipping))
        output = self.softmax(self.out(output[0]))
        #print(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class RGB_to_Hidden(nn.Module):
    def __init__(self,rgb_n_dims, hidden_n_dims):
        super(RGB_to_Hidden, self).__init__()
        self.rgb_n_dims = rgb_n_dims
        self.hidden_n_dims = hidden_n_dims
        self.single_layer = nn.Linear(self.rgb_n_dims,self.hidden_n_dims)

    def forward(self, input):
        output = self.single_layer(input)
        #output = torch.stack([output for i in range(NUM_LAYERS)]).to(device)
        if NUM_LAYERS != 1 : output = (torch.squeeze(output)).unsqueeze(1)
        return output
