import numpy as np

rng = np.random.default_rng(3)

from matplotlib.pyplot import subplots
x = rng.standard_normal(100)
y = rng.standard_normal(100)

fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker='+')
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y")
ax.set_xlim([-1,1])
fig.set_size_inches(12,3)

total = 0
for value, weight in zip([2,3,19],
[0.2,0.3,0.5]):
    total += weight * value
print('Weighted average is: {0}'.format(total))