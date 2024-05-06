import torch
import math
import numpy as np


FN_NAME = 'wonky_hn'

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
    
def continuous_pointil(x):
    if (-2.424 <= x <= -2.24) or (2.24 <= x <= 2.424):
        return 1
    return 0

def log_gamma(x):
    return math.log(math.gamma(x))

def weier(x):
    #sum_{n}^{infty}(a^n * cos(b^n\pi x))
    count = 0
    for i in range(1, 101):
        count += (0.5 ** i) * math.cos(4 ** i * math.pi * x)
    return count

def sawtooth(x):
    count = 0
    for i in range(1, 101):
        count += ((-1)**(i + 1) / i) * math.sin(i * x)
    return count

def bessel(x):
    count = 0
    for i in range(1, 101):
        count += ((-1)**i) / (math.gamma(i)*math.gamma(i+3)) * (x/2)**(2*i +2)
    return count

def dedekind_eta(tau): #continuous_dede
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24) * np.product([1 - q**n for n in range(1, 100)])
    return eta.real

def sin2(x):
    return math.sin((x)**2)

def sin184(x):
    return math.sin((pow(x, 1.84)).real)

def sin16(x):
    return math.sin(pow(x,1.6))

def wonky_hn(x):
    t1 = (4 * x + 1) / 4
    t2 = ((2 * x + 1) / 4) * math.cos(math.pi * x)
    return t1 - t2

def gen_dataset_continuous(start, end, n_max, file_path=f'data/{FN_NAME}_dataset.pt'):
    data = []
    inputs = []
    step_size = (end - start) / n_max
    i = start
    while i < end:
        inputs.append(i)
        i += step_size
    for x in inputs:
        fx = wonky_hn(x)
        data.append((x, fx))
    torch.save(data, file_path)

n_max = 1000000
start = 0
end = 100
gen_dataset_continuous(start, end, n_max, f'data/{FN_NAME}_dataset.pt')
#gen_dataset(n_max, f'data/{FN_NAME}_dataset.pt')