import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 4
NUM_LAYERS = 2
batch_size = 4
pad_idx = 2 #idx of the padding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,embeddings):

        super(EncoderRNN, self).__init__()
        self.embedding_dimension = embeddings.shape[1]
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, self.embedding_dimension,padding_idx=pad_idx).from_pretrained(embeddings,freeze=False)
        self.gru = nn.GRU(self.embedding_dimension, self.embedding_dimension,num_layers = NUM_LAYERS)
        self.embeedding_to_RGB = nn.Linear(self.embedding_dimension,hidden_size) #could be larger

    # def forward(self, input, hidden):
    #
    #     embedded = self.embedding(input).view(1, batch_size, self.embedding_dimension)
    #     output = embedded
    #     output = torch.nn.utils.rnn.pack_padded_sequence(output, batch_size) #size always MAX LENGTH always since padded
    #     output, hidden = self.gru(output, hidden)
    #     output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
    #     output = self.embeedding_to_RGB(output)
    #     return output, hidden
    def forward(self, input_seqs, input_lengths, hidden=None):

        embedded = self.embedding(input_seqs)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = self.embeedding_to_RGB(outputs)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        #print(outputs.shape)
        outputs = F.sigmoid(outputs)
        return outputs, hidden

    def initHidden(self):
        return torch.zeros(self.embedding_dimension, batch_size, self.embedding_dimension, device=device) #return torch.zeros(NUM_LAYERS, 1, self.hidden_size, device=device)

#RGB SPACE = 3 = HIDDEN SIZE
#RGB VS HIDDEN
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embeddings, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dimension = embeddings.shape[1]
        self.output_size = output_size #self.embedding_dimension


        self.rgb_to_embedding = nn.Linear(self.hidden_size,self.embedding_dimension)

        #TODO a list of fully connected
        self.MLP = nn.ModuleList()
        self.MLP += [nn.Linear(self.hidden_size,self.embedding_dimension)]
        self.MLP += [nn.Linear(self.embedding_dimension,self.embedding_dimension) for _ in range(NUM_LAYERS)]
        #self.MLP =  [[nn.Linear(self.hidden_size,self.embedding_dimension)] +
        #                          [nn.Linear(self.embedding_dimension,self.embedding_dimension) for _ in range(NUM_LAYERS)]]

        self.embedding = nn.Embedding(output_size, self.embedding_dimension,padding_idx=pad_idx).from_pretrained(embeddings,freeze=False)

        self.gru = nn.GRU(self.embedding_dimension, self.embedding_dimension,num_layers=NUM_LAYERS)
        self.out = nn.Linear(self.embedding_dimension, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        output = self.embedding(input).view(1, batch_size, self.embedding_dimension)
        #output is 1 x 64 x 300
        output = F.relu(output)
        

        #print(RGB_to_EMB.is_cuda)
        hidden = self.rgb_to_embedding(hidden)
        #torch.squeeze clone?

        hidden_layers = torch.stack([hidden.clone() for i in range(NUM_LAYERS)]).to(device)
        hidden_layers = torch.squeeze(hidden_layers)
        #hidden_laxers = hidden_layers.to(device) #print(hidden_layers[0].shape)
        # NUM_LAYERS * batch_size * embedding_dimensions
        #print(hidden.size())
        #output = torch.nn.utils.rnn.pack_padded_sequence(output, batch_size)
        output, hidden = self.gru(output, hidden_layers) #self.gru(output, hidden)
        #output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) #, batch_first=True

        output = self.softmax(self.out(output[0]))

        return output, hidden

    def initHidden(self):
        return torch.zeros(NUM_LAYERS, batch_size, self.hidden_size, device=device)



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,embeddings, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_dimension = embeddings.shape[1] #3 dim RGB
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        #self.embedding = nn.Embedding(self.output_size, self.hidden_size) #300 -> 300
        self.embedding = nn.Embedding(output_size, self.embedding_dimension).from_pretrained(embeddings,freeze=False)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length) #300 + 3 ?
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,num_layers=NUM_LAYERS)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

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


