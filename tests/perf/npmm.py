import time, numpy as np
N = 10

M1 = np.random.rand(N, N)
M2 = np.random.rand(N, N)


T1 = time.time()

for i in range(1000000):
    M3 = M1 @ M2

T2 = time.time()

print(T2 - T1)
