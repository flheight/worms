import numpy as np
from sklearn.cluster import KMeans

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def learn(self, epochs, init_nodes, lam, lr, batch_size=32):
        kmeans = KMeans(self.out_dim).fit(self.data)
        var = 0.01 * np.eye(self.data.shape[1])
        self.clusters = [np.random.multivariate_normal(center, var, init_nodes) for center in kmeans.cluster_centers_]

        for epoch in range(epochs):
            indices = np.random.randint(self.data.shape[0], size=batch_size)
            x_batch = self.data[indices]

            diffs = [x_batch - worm[:, np.newaxis] for worm in self.clusters]
            dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]

            min_dists = np.array([np.min(dt, axis=0) for dt in dists])
            winner_worm_idxs = np.argmin(min_dists, axis=0)
            winner_idxs = np.array([np.argmin(dists[winner_worm_idxs[i]][:, i]) for i in range(x_batch.shape[0])])

            for winner_worm_idx, winner_worm_count in zip(*np.unique(winner_worm_idxs, return_counts=True)):
                segments = self.clusters[winner_worm_idx][1:] - self.clusters[winner_worm_idx][:-1]

                density = np.zeros(self.clusters[winner_worm_idx].shape[0] - 1)
                unique_idxs, unique_idxs_counts = np.unique(winner_idxs[winner_worm_idxs == winner_worm_idx], return_counts=True)
                left_mask = unique_idxs > 0
                right_mask = unique_idxs < self.clusters[winner_worm_idx].shape[0] - 1
                density[unique_idxs[left_mask] - 1] += unique_idxs_counts[left_mask]
                density[unique_idxs[right_mask]] += unique_idxs_counts[right_mask]
                cumdensity1 = np.cumsum(density)
                cumdensity2 = np.cumsum(density[::-1])
                d = batch_size * (density + cumdensity1[-1])
                segments *= lam * lr * winner_worm_count / d.reshape(-1, 1)
                self.clusters[winner_worm_idx][:-1] += segments * cumdensity2.reshape(-1, 1)
                self.clusters[winner_worm_idx][1:] -= segments * cumdensity1.reshape(-1, 1)

            for i in range(batch_size):
                winner_worm = winner_worm_idxs[i]
                winner_idx = winner_idxs[i]
                self.clusters[winner_worm][winner_idx] += diffs[winner_worm][winner_idx, i] * lr / batch_size
