import multiprocessing

# used everywhere
features = 300
min_count = 2 # I think key word should occur at least twice in the text
num_workers = multiprocessing.cpu_count()
use_cuda = True

# loader.py
batch_size = 256
max_length = 57

# main.py
epochs = 10
lr = 1e-3
save_step = 1000
val_step = 100
val_num = 10

# model.py
dropout_p = 0.04
mem_dim = 150
cov_dim = 150
kernel_sizes = [1, 2, 3, 4, 5]