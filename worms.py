import numpy as np
from sklearn.cluster import KMeans

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def __init_clusters(self):
        kmeans = KMeans(self.out_dim).fit(self.data)
        var = .01 * np.eye(self.data.shape[1])
        self.clusters = [np.random.multivariate_normal(center, var, 30) for center in kmeans.cluster_centers_]

    def learn(self, epochs, lam=.005, lr=.5):
        self.__init_clusters()

        sample = self.data[np.random.choice(np.arange(self.data.shape[0]), 256)]
        density = [np.mean(np.exp(-np.sum((sample[:, np.newaxis, :] - chain)**2 / (2 * .2**2), axis=2)), axis=0) for chain in self.clusters]

        for _ in range(epochs):
            x = self.data[np.random.randint(self.data.shape[0])]

            segments = [chain[1:] - chain[:-1] for chain in self.clusters]
            diff = [x - chain[:-1] for chain in self.clusters]

            projections = [np.einsum('ij,ij->i', seg, df) / np.einsum('ij,ij->i', seg, seg) for seg, df in zip(segments, diff)]
            projections = [np.clip(proj[:, np.newaxis], 0, 1) for proj in projections]
            dists = [np.einsum('ij,ij->i', df - proj * seg, df - proj * seg) for df, proj, seg in zip(diff, projections, segments)]

            winner_worm = np.argmin([np.min(dist) for dist in dists])
            winner_idx = np.argmin(dists[winner_worm])
            weight = projections[winner_worm][winner_idx, 0]
            weights = np.array([[weight], [1 - weight]])

            sample = self.data[np.random.choice(np.arange(self.data.shape[0]), 256)]

            diffs = sample[:, np.newaxis, :] - self.clusters[winner_worm][winner_idx:winner_idx + 2]
            density[winner_worm][winner_idx:winner_idx + 2] = np.mean(np.exp(-np.einsum('ijk,ijk->ij', diffs, diffs) / (2 * .2**2)), axis=0)

            self.clusters[winner_worm][winner_idx:winner_idx + 2] += (x - self.clusters[winner_worm][winner_idx:winner_idx + 2]) * lr * weights
            self.clusters[winner_worm][:-1] += lam * lr * segments[winner_worm] / (1 + density[winner_worm][:-1, np.newaxis])
            self.clusters[winner_worm][1:] -= lam * lr * segments[winner_worm] / (1 + density[winner_worm][1:, np.newaxis])
