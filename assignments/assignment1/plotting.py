import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
with open("stats/uniform_sampling_batch", 'r') as f:
    big_area = json.load(f)

with open("stats/uniform_sampling_batch_spec_range", 'r') as f:
    small_area = json.load(f)


ran = [(i+1) * 100 for i in range(10)]

plt.plot(ran, big_area[1], 'b-', label='area A')
plt.plot(ran, small_area[1], 'r-', label='area B')
plt.title("comparing std bigger and smaller sampling region using batched sampling")
plt.xlabel("number of points used to estimate area")
plt.ylabel("standard deviation of batches")
plt.legend()
plt.show()

plt.plot(ran, big_area[0], 'b-', label='area A')
plt.plot(ran, small_area[0], 'r-', label='area B')
plt.xlabel("number of points used to estimate area")
plt.ylabel("mean of batches")
plt.title("comparing std bigger and smaller sampling region using batched sampling")
plt.legend()
plt.show()
"""

"""
with open("sampling_comparison_latin.json", 'r') as f:
    latin = json.load(f)

with open("sampling_comparison_ortho.json", 'r') as f:
    orthogonal = json.load(f)

with open("sampling_comparison_uniform.json", 'r') as f:
    uniform = json.load(f)

ran = [500, 1000, 2000, 4000, 8000]

plt.plot(ran, latin[1], 'b-', label='latin')
plt.plot(ran, orthogonal[1], 'r-', label='orthogonal latin')
plt.plot(ran, uniform[1], 'y-', label='uniform')
plt.title("comparing standard deviation three sampling methods")
plt.xlabel("number of points used to estimate area")
plt.ylabel("estimated standard dev area")
plt.legend()
plt.show()
"""

with open('stats/chain_length_vs_num_points.np', 'rb') as f:
    data = np.load(f)

num_points_li = [10, 50, 100, 500, 1000, 5000, 10000]
chain_len_li = [10, 50, 100, 500, 1000, 5000, 10000, 50000]

numrows = 2
numcols = 2

for i, number_points in enumerate(num_points_li[:4]):
    plot_data = list(zip(*data[i]))    
    mean_data_points, std_data_points = np.array(plot_data[0]), np.array(plot_data[1]) 
    plt.subplot(numrows, numcols, i+1)
    plt.plot(chain_len_li, plot_data[0], label='mean estimate')
    plt.fill_between(chain_len_li, mean_data_points-std_data_points, mean_data_points+std_data_points, label='standard deviation', alpha=0.2)
    plt.xlabel("steps used to determine point in Mandelbrot set")
    plt.ylabel("area Mandelbrot set")
    plt.legend()
    plt.title("number of points {}".format(number_points))
    plt.xscale('log')
    plt.tight_layout()

plt.show()



