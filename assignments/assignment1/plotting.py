import json
import matplotlib.pyplot as plt
import seaborn as sns

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
