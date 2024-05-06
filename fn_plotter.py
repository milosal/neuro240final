import numpy as np
import matplotlib.pyplot as plt
import math

def dedekind_eta(tau):
    q = np.exp(2j * np.pi * tau)
    eta = q**(1/24) * np.product([1 - q**n for n in range(1, 100)])
    return eta.real

def log_gamma(x):
    return math.log(math.gamma(x))

def sin184(x):
    return math.sin((pow(x, 1.84)).real)

def wonky_hn(x):
    t1 = (4 * x + 1) / 4
    t2 = ((2 * x + 1) / 4) * math.cos(math.pi * x)
    return t1 - t2

inputs = np.linspace(0.2, 100.0, 100000)
print(inputs)
values = np.array([wonky_hn(input) for input in inputs])
print(values)
plt.figure(figsize=(10, 6))
plt.plot(inputs.real, values, label="wonky(x)")
plt.title("wonky(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()