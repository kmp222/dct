import math

from scipy.fft import dctn
import numpy as np
from timeit import default_timer as timer
import timeit
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import time
from dct import my_dct2_2d


def parse_c_times(path):
    file1 = open(path, 'r')
    lines = file1.readlines()
    times = []
    for line in lines:
        (size, time) = line.strip().split(' ')
        times.append(float(time))
    return times
# m = np.array(
#     [
#         [231, 32, 233, 161, 24, 71, 140, 245],
#         [247, 40, 248, 245, 124, 204, 36, 107],
#         [234, 202, 245, 167, 9, 217, 239, 173],
#         [193, 190, 100, 167, 43, 180, 8, 70],
#         [11, 24, 210, 177, 81, 243, 8, 112],
#         [97, 195, 203, 47, 125, 114, 165, 181],
#         [193, 70, 174, 167, 41, 30, 127, 245],
#         [87, 149, 57, 192, 65, 129, 178, 228],
#     ]
# )
# m = dctn(m, type=2, norm='ortho', workers=-1)
# print(m)

sizes = [i for i in range(10,100, 10)]
new_sizes = [i for i in range(100, 1001, 50)]
sizes.extend(new_sizes)


my_times = parse_c_times('my_dct_performance.txt')
libc_times = parse_c_times('library_performance.txt')
scipy_times = parse_c_times('scipy.txt')


#my_times = [my_dct_bench(size) for size in sizes]
time_complexity2 = [(i*i*math.log2(i))/10**9 for i in sizes]
time_complexity3 = [(i**3)/10**7 for i in sizes]

plt.plot(sizes, scipy_times, label='scipy dct', color='#0ADA00')
plt.plot(sizes, libc_times, label='fftw dct', color='#33C1FF')
plt.plot(sizes, my_times, label='our dct', color='#FF7433')
plt.plot(sizes, time_complexity2, label=r'$n^2*log_2(n)$', linestyle='dashed', color='blue')
plt.plot(sizes, time_complexity3, label=r'$n^3$', linestyle='dashed', color='red')
plt.xlabel('n')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.legend()
plt.savefig('times.pdf', format='pdf', bbox_inches='tight')
plt.close()
