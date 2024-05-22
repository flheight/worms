import numpy as np
from sklearn.cluster import KMeans

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def __init_clusters(self, max_nodes, eps=.005):
        kmeans = KMeans(self.out_dim).fit(self.data)
        var = eps * np.eye(self.data.shape[1])
        self.clusters = [np.random.multivariate_normal(center, var, max_nodes) for center in kmeans.cluster_centers_]

    def learn(self, epochs, max_nodes=50, lam=.005, lr=.5):
        self.__init_clusters(max_nodes)

        density = [np.zeros((chain.shape[0] - 1, 1)) for chain in self.clusters]

        for epoch in range(1, epochs + 1):
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

            density[winner_worm][winner_idx] += 1

            cdensity1 = np.cumsum(density[winner_worm]).reshape(-1, 1)
            cdensity2 = np.cumsum(density[winner_worm][::-1]).reshape(-1, 1)

            d = density[winner_worm] + cdensity1[-1]

            self.clusters[winner_worm][winner_idx:winner_idx + 2] += (x - self.clusters[winner_worm][winner_idx:winner_idx + 2]) * lr * weights
            self.clusters[winner_worm][:-1] += lam * lr * segments[winner_worm] * cdensity2 / d
            self.clusters[winner_worm][1:] -= lam * lr * segments[winner_worm] * cdensity1 / d
