import pandas as pd
import matplotlib.pyplot as plt
import  torch
import jieba
import bz2
import  os,re
import gensim
import pkuseg
import numpy as np
from gensim.models import KeyedVectors
import config as config

# the pretrain word embedding model
def bz2Decompress():
    if os.path.exists("./embeddings/sgns.weibo.bigram") == False:
        with open ("./embeddings/sgns.weibo.bigram", 'wb') as new_file, open("./embeddings/sgns.weibo.bigram.bz2", 'rb') as file:
            decompressor = bz2.BZ2Decompressor()
            for data in iter (lambda: file.read (100 * 1024), b''):
                new_file.write (decompressor.decompress (data))

#load the data
def get_df():
    weibo = pd.read_csv('./data/ced_dataset.txt', sep='\t', names=['label', 'content'], encoding='utf-8')
    weibo = weibo.dropna ()  # delete the NA value
    return  weibo['label'].values.tolist(),weibo['content'].values.tolist()

def jieba_cut(contents):
    contents_S=[]
    for line in contents:
        current_segment=jieba.lcut(line) # the list of the tokenization
        contents_S.append (current_segment)
    return contents_S

def pkuseg_cut(contents,model_name="web"):
    contents_S = []
    for line in contents:
        line = re.sub ("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", line)
        current_segment = jieba.lcut (line)  # the list of the tokenization words
        contents_S.append(current_segment)
    return contents_S

# load the stopwords
def get_stopwords():
    stopwords = pd.read_csv ("./stopwords/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'],
                             encoding='utf-8')
    return  set(stopwords['stopword'].values.tolist())

# delete the stopwords
def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words


# use gensim to load the pretraining word embedding model
def get_word2vec():
    word2vec=KeyedVectors.load_word2vec_format('./embeddings/sgns.weibo.bigram',binary=False,unicode_errors="ignore")
    return word2vec

## Returns preprocessed (labels, index)= >; ([1] [234] 1234)
def key_to_index(contents,word2vec,num_words):
    '''
    :param contents:
    :param word2vec: Pre-trained word vector model, word vectors are arranged in descending order according to frequency of use
    :param num_words: Maximum number of words, select the word with the highest usage rate before using new_words
    :return:
    '''
    train_tokens=[]
    contents_S = pkuseg_cut(contents)
    stopword = get_stopwords()
    contents_clean, all_words = drop_stopwords(contents_S, stopword)
    for line_clean in contents_clean:
        for i, key in enumerate(line_clean):
            try:
                index=word2vec.key_to_index[key]
                if index<num_words:
                    line_clean[i]=word2vec.key_to_index[key]
                else:
                    line_clean[i] =0  # Words exceeding the preceding num_words are replaced with 0
            except KeyError:  # If the word is not in the dictionary, 0 is printed
                line_clean[i]=0
        train_tokens.append(line_clean)
    return train_tokens

def labels_contents():
    labels, contents = get_df()
    contents_S = jieba_cut(contents)
    stopword = get_stopwords()
    contents_clean, all_words = drop_stopwords(contents_S, stopword)
    contents_clean=[" ".join (x) for x in contents_clean]
    return labels, contents_clean

def get_maxlength(train_tokens):
    num_tokens = [len (tokens) for tokens in train_tokens]
    num_tokens = np.array (num_tokens)
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int (max_tokens)
    return max_tokens

def padding_truncating(train_tokens,maxlen):
    for i,token in enumerate(train_tokens):
        if len(token)>maxlen:
            train_tokens[i]=token[len(token)-maxlen:]
        elif len(token)<maxlen:
            train_tokens[i]=[0]*(maxlen-len(token))+token
    return train_tokens

# word embedding
def get_embedding(word2vec,num_words=50000,embedding_dim=300):
    embedding_matrix = np.zeros((num_words, embedding_dim))  # a matrix of [num_words, embedding_dim]
    for i in range (num_words):
        embedding_matrix[i, :] = word2vec[i]  # word vector for the first 50000 index words
    embedding_matrix = embedding_matrix.astype('float32')
    return torch.from_numpy(embedding_matrix)

word2vec=get_word2vec()

embedding=get_embedding(word2vec,num_words=config.num_words,embedding_dim=config.embedding_dim)

## draw the curve of learning
def plot_learning_curve(train_loss,test_loss, title=''):
    plt.figure(figsize=(20, 8))
    plt.plot(train_loss, c='tab:red', label='train')
    plt.plot(test_loss, c='tab:blue', label='test')
    plt.ylim(0.0, 1.)
    plt.xlabel('Training epoch')
    plt.ylabel(title)
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_learning_curve([1,2,3,4],[2,3,1,2])
