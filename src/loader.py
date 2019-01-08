import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import config
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec

path = '../data/'
model = word2vec.Word2Vec.load(path + 'word2vec.model')
empty_word = np.zeros(config.features).astype(np.float32)

class MyDataset(Dataset):
    def __init__(self, usage):
        in_f = open(path + '%s_q.pkl' % usage, 'rb')
        self.data_q = pickle.load(in_f)
        in_f.close()
        in_f = open(path + '%s_a.pkl' % usage, 'rb')
        self.data_a = pickle.load(in_f)
        in_f.close()
        self.label = np.load(path + '%s_label.npy' % usage)
        self.length = len(self.label)
        
    def __getitem__(self, index):
        line = self.data_q[index]
        line_q_vector = []
        words = line.split(' ')
        for word in words:
            word = word.strip()
            if model.wv.__contains__(word):
                line_q_vector.append(model.wv.__getitem__(word))
            else:
                line_q_vector.append(empty_word)
        line_q_vector = np.array(line_q_vector)
        
        line = self.data_a[index]
        line_a_vector = []
        words = line.split(' ')
        for word in words:
            word = word.strip()
            if model.wv.__contains__(word):
                line_a_vector.append(model.wv.__getitem__(word))
            else:
                line_a_vector.append(empty_word)
        line_a_vector = np.array(line_a_vector)
        
        return torch.Tensor(line_q_vector), torch.Tensor(line_a_vector), self.label[index]
    
    def __len__(self):
        return self.length

def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    if pad > len(vec):
        pad_size[0] = pad - len(vec)
        return torch.cat([vec, torch.zeros(pad_size)])
    else:
        return vec[:pad:]

def collate_fn(batch):
    # find longest sequence
    max_len1 = max([len(s[0]) for s in batch] + [5])
    xs1 = torch.stack([pad_tensor(s[0], pad=max_len1) for s in batch], dim=0)
    max_len2 = max([len(s[1]) for s in batch] + [5])
    xs2 = torch.stack([pad_tensor(s[1], pad=max_len2) for s in batch], dim=0)
    ys = torch.Tensor([s[2] for s in batch])
    return xs1, xs2, ys
    
def get_loaders():
    train_set = MyDataset('train')
    val_set = MyDataset('val')
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, 
        shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(val_set, batch_size=config.batch_size, 
        shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(val_set, batch_size=config.batch_size, collate_fn=collate_fn)
    return train_loader, validation_loader, test_loader