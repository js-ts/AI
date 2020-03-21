import torch
import torch.nn as nn 
import torch.nn.functional as F 

import random

# SOS
# EOS

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        input_size: input_language.n_words
        '''
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embdeding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embdeding(input).view(1, 1, -1)
        output = embedded

        output, hidden = self.gru(output, hidden)

        return output, hidden
    
    def initHidden(self, ):
        return torch.zeros(1, 1, self.hidden_size)



class DecoderRNNSimple(nn.Module):
    def __init__(self, hidden_size, output_size):
        '''
        output_size: output_language.n_words
        '''

        super(DecoderRNNSimple, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self, ):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNNAttn(nn.Module):
    def __init__(self, hidden_size, output_size, max_lenght=10, dropout_p=0.1, ):
        '''
        Here the maximum length is 10 words (that includes ending punctuation) 
        '''
        super(DecoderRNNAttn, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_lenght = max_lenght

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_lenght)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = self.attn(torch.cat((embedded[0], hidden[0]), 1))
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights

    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


SOS_TOKEN = 0
EOS_TOKEN = 1
teacher_forcing_ratio = 0.5 

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_output[i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]])
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # feed target as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i] # teacher
    else:
        for i in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[i])

            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_TOKEN:
                break
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()


# encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
# criterion = nn.NLLLoss()
