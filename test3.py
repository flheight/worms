import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import time

X, y = make_moons(n_samples=10000, noise=0.1, random_state=42)

np.random.seed()

from worms2 import Worms

net2 = Worms(k=2)
net2.data = X


start = time.time()
net2.learn(10000, init_nodes=2, lam=1.5, mu=1.3, lr=.05)
end = time.time()

print(f"Elapsed time : {end - start}")


plt.scatter(X[:, 0], X[:, 1], color="gray")

for i in range(2):
    plt.plot(net2.clusters[i][:, 0], net2.clusters[i][:, 1], linewidth=5)

plt.show()
