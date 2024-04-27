import torch
import math

# simple_poly is 2x^2 + 3x - 10
# div_binom is (2x^2 + 3x - 10) / math.binom(x, 2) or 0 if denom is 0
# normed_hn is normed_h_n(x) = hn(1000000x)
# normed_sin is normed_sin(x) = sin(1000000x)

FN_NAME = 'normed_sin'

def sin(x):
    return math.sin(x)

def hn(n):
    curr = n
    n_h = 0
    while curr != 1:
        if curr % 2 == 0:
            curr = curr / 2
        else:
            curr = 3*curr + 1
        n_h += 1
    return n_h

def div_binom(n):
    if math.comb(n, 2) == 0:
        return 0
    else:
        return ((2 * n)**2 + 3*n - 10) / math.comb(n, 2)
    
def gen_dataset_norm(n_max, file_path=f'data/{FN_NAME}_dataset.pt'):
    data = []
    for n in range(1, n_max + 1):
        fn = sin(n)
        data.append((n/1000000, fn))
    torch.save(data, file_path)

def gen_dataset(n_max, file_path=f'data/{FN_NAME}_dataset.pt'):
    data = []
    for n in range(1, n_max + 1):
        fn = div_binom(n)
        data.append((n, fn))
    torch.save(data, file_path)

n_max = 1000000
gen_dataset_norm(n_max, f'data/{FN_NAME}_dataset.pt')
#gen_dataset(n_max, f'data/{FN_NAME}_dataset.pt')