import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ProjModule(nn.Module):

    def __init__(self):
        super(ProjModule, self).__init__()

        self.dropout = nn.Dropout(config.dropout_p)
        self.proj1 = nn.Sequential(
            nn.Linear(config.features, config.mem_dim),
            nn.Sigmoid()
        )
        self.proj2 = nn.Sequential(
            nn.Linear(config.features, config.mem_dim),
            nn.Tanh()
        )

    # batch_size * word_num * feature -> batch_size * word_num * mem_dim
    def forward(self, x):
        dropped_x = self.dropout(x)
        proj1 = self.proj1(dropped_x)
        proj2 = self.proj2(dropped_x)
        proj = proj1.mul(proj2)
        return proj


class AttnModule(nn.Module):

    def __init__(self):
        super(AttnModule, self).__init__()

    # batch_size * word_num * mem_dim -> batch_size * word_num * mem_dim
    def forward(self, in_q, in_a):
        e_qa = torch.bmm(in_q, in_a.transpose(1, 2))
        w_a = F.softmax(e_qa, dim=1) # different from original repo
        w_q = F.softmax(e_qa, dim=2)
        h_a = torch.bmm(w_a.transpose(1, 2), in_q)
        h_q = torch.bmm(w_q, in_a)
        t_a = in_a.mul(h_a)
        t_q = in_q.mul(h_q)
        return t_a, t_q


class ConvModule(nn.Module):

    def __init__(self):
        super(ConvModule, self).__init__()

        self.convs = nn.ModuleList([self.get_conv(l) for l in config.kernel_sizes])
        self.fc = nn.Linear(2 * len(config.kernel_sizes) * config.cov_dim, 1)
        self.index = range(len(config.kernel_sizes))

    def get_conv(self, kernel_size):
        return nn.Sequential(
            nn.Conv1d(config.mem_dim, config.cov_dim, kernel_size),
            nn.ReLU(inplace=True)
        )

    # batch_size * word_num * mem_dim -> batch_size * 1
    def forward(self, t_q, t_a):
        t_q = t_q.transpose(1, 2)
        q_out = [self.convs[i](t_q) for i in self.index]
        q_max = [F.max_pool1d(q_out[i], q_out[i].size(2)).squeeze(2) for i in self.index]

        t_a = t_a.transpose(1, 2)
        a_out = [self.convs[i](t_a) for i in self.index]
        a_max = [F.max_pool1d(a_out[i], a_out[i].size(2)).squeeze(2) for i in self.index]

        # batch_size * cov_dim
        conv_out = torch.cat(q_max + a_max, 1)
        return  self.fc(conv_out)


class CompareAggregate(nn.Module):

    def __init__(self, path=None):
        super(CompareAggregate, self).__init__()

        # use word2vec instead of embedding
        self.represent = ProjModule()
        self.compare = AttnModule()
        self.aggregate = ConvModule()

        if path != None:
            self.load_weight(path)

    # outputs logit without sigmoid
    def forward(self, in_q, in_a):
        # word representation
        vector_q = self.represent(in_q)
        vector_a = self.represent(in_a)

        # attention & comparison
        t_a, t_q = self.compare(vector_q, vector_a)

        # aggregation
        logit = self.aggregate(t_a, t_q)
        return logit
    
    def predict(self, logit):
        return F.sigmoid(logit)

    def load_weight(self, path):
        if config.use_cuda:
            state_dict = torch.load(path + self.get_name())
        else:
            state_dict = torch.load(path + self.get_name(), map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict, strict=False)

    def save_weight(self, path):
        torch.save(self.state_dict(), path + self.get_name())

    def get_name(self):
        return 'CompareAggregate'
