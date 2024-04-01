import torch

def H(n):
    curr = n
    n_h = 0
    while curr != 1:
        if curr % 2 == 0:
            curr = curr / 2
        else:
            curr = 3*curr + 1
        n_h += 1
    return n_h

def gen_dataset(n_max, file_path='data/hn_dataset.pt'):
    data = []
    for n in range(1, n_max + 1):
        hn = H(n)
        data.append((n, hn))
    torch.save(data, file_path)

n_max = 100000
gen_dataset(n_max, 'data/hn_dataset.pt')