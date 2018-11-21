import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 4
NUM_LAYERS = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,embeddings):

        super(EncoderRNN, self).__init__()
        self.embedding_dimension = embeddings.shape[1]
        self.num_layers = NUM_LAYERS
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.embedding_dimension).from_pretrained(embeddings,freeze=False)
        self.gru = nn.GRU(self.embedding_dimension, self.embedding_dimension,num_layers = NUM_LAYERS)
        self.embeedding_to_RGB = nn.Linear(self.embedding_dimension,hidden_size) #could be larger
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.embeedding_to_RGB(output)
        output = self.sigmoid(output)
        return output, hidden

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
        output = self.softmax(self.out(output[0]))
        #print(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length+1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        #hidden = (torch.squeeze(hidden)).unsqueeze(0)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

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