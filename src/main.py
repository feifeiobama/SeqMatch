from tensorboardX import SummaryWriter
import config
import torch
import utils
import loader
import model
from tqdm import tqdm

save_path = '../checkpoint/'

class Summary(object):
    def __init__(self, epoch_length):
        self.epoch_length = epoch_length
        self.writer = SummaryWriter()
        
    def add_info(self, epoch, step, info:dict, usage):
        x = epoch * self.epoch_length + step
        self.writer.add_scalars('loss', {usage: info['loss']}, x)
        self.writer.add_scalars('acc', {usage: info['acc']}, x)
        
    def add_score(self, epoch, step, info:dict):
        x = epoch * self.epoch_length + step
        self.writer.add_scalars('MAP', {"val": info['MAP']}, x)
        self.writer.add_scalars('MRR', {"val": info['MRR']}, x)
        
    def __del__(self):
        self.writer.close()
        
        
class PlayGround(object):
    def __init__(self, path=None):
        self.net = model.CompareAggregate()
        if config.use_cuda:
            self.net = self.net.cuda()
        if path != None:
            self.net.load_weight(path)
        self.train_loader, self.validation_loader, self.test_loader = loader.get_loaders()
        self.summary = Summary(len(self.train_loader))
        self.optim = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.epoch = self.step = 0
        self.last_map = 0
        print('Inititalize done')
        
    def calculate_loss(self, logit, label):
        loss_func = torch.nn.BCEWithLogitsLoss()
        return loss_func(logit, label)
    
    def train(self, in_q, in_a, label):
        self.net.train()
        self.optim.zero_grad()
        logit = self.net(in_q, in_a).flatten()
        loss = self.calculate_loss(logit, label)
        loss.backward()
        self.optim.step()
        acc = utils.accuracy(logit, label)
        info = utils.summarize_info(loss, acc)
        self.summary.add_info(self.epoch, self.step, info, 'train')
        
    def validate_info(self):
        loss = acc = 0
        self.net.eval()
        for cnt, (in_q, in_a, label) in enumerate(self.validation_loader):
            if cnt == config.val_num:
                break
            in_q, in_a, label = in_q.detach(), in_a.detach(), label.detach()
            if config.use_cuda:
                in_q, in_a, label = in_q.cuda(), in_a.cuda(), label.cuda()
            logit = self.net(in_q, in_a).flatten()

            loss += self.calculate_loss(logit, label) / config.val_num
            acc += utils.accuracy(logit, label) / config.val_num
        info = utils.summarize_info(loss, acc)
        self.summary.add_info(self.epoch, self.step, info, 'validation')
        
    # return if MAP is highest
    def validate_score(self):
        self.net.eval()
        ans = []
        for in_q, in_a, _ in self.test_loader:
            in_q, in_a = in_q.detach(), in_a.detach()
            if config.use_cuda:
                in_q, in_a = in_q.cuda(), in_a.cuda()
            logit = self.net(in_q, in_a).flatten()
            ans += self.net.predict(logit).tolist()
            
        curr_map, curr_mrr = utils.calc_score(ans)
        score = utils.summarize_score(curr_map, curr_mrr)
        self.summary.add_score(self.epoch, self.step, score)
        if curr_map >= self.last_map:
            self.last_map = curr_map
            return True
        else:
            return False
        
    def play(self):
        print('Let us begin')
        for self.epoch in range(config.epochs):
            for self.step, (in_q, in_a, label) in tqdm(enumerate(self.train_loader)):
                if config.use_cuda:
                    in_q, in_a, label = in_q.cuda(), in_a.cuda(), label.cuda()
                self.train(in_q, in_a, label)
                if (self.step + 1) % config.val_step == 0:
                    self.validate_info()
                if (self.step + 1) % config.save_step == 0 and self.validate_score():
                    self.net.save_weight(save_path)  
        

playground = PlayGround()
playground.play()

# playground = PlayGround(save_path)
# playground.validate_score()