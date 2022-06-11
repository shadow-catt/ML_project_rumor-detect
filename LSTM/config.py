
import torch


num_words =50000  # Maximum number of words, select the word with the highest usage rate before using new_words
maxlength =58     #The get_maxlength() method in the utils package is used to get the maximum length of tokens. Take
                    # the mean of tokens and add two standard deviations of tokens.
                    # Assuming that the distribution of tokens length is a normal distribution, max_tokens can cover about 95% of the samples

embedding_dim = 300 #  the dimension of word vectors
lstm_hidden_size=128  #LSTM hidden layer size
lstm_num_layers=2     # the layer of LSTM
lstm_bidirectional=True # Two-way LSTM
lr=1e-3    # the learning rate

dropout = 0.2  #The probability of a random zero
batch_size = 128
early_stop_cnt = 10   #If the accuracy of validation set is not improved, the epoch can be waited at most
epoch= 150

device = torch.device("cpu")