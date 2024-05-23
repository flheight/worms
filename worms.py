import numpy as np
from sklearn.cluster import KMeans

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def __init_clusters(self, init_nodes):
        kmeans = KMeans(self.out_dim).fit(self.data)
        var = 1e-4 * np.eye(self.data.shape[1])
        self.clusters = [np.random.multivariate_normal(center, var, init_nodes) for center in kmeans.cluster_centers_]

    def __mh(self, worm_idx, direction, n_samples, alpha1=0.95, alpha2=0.95):
        init_cost = self.loss(self.clusters, n_samples)
        self.cost.append(init_cost)

        snapshot = self.clusters.copy()
        var = 1e-4 * np.eye(self.data.shape[1])

        if direction == 0:
            new = np.random.multivariate_normal(self.clusters[worm_idx][0], var, 1)
            snapshot[worm_idx] = np.vstack((new, snapshot[worm_idx]))
        else:
            new = np.random.multivariate_normal(self.clusters[worm_idx][-1], var, 1)
            snapshot[worm_idx] = np.vstack((snapshot[worm_idx], new))

        last_cost = self.loss(snapshot, n_samples)
        if last_cost / init_cost < alpha1:
            return snapshot

        if self.clusters[worm_idx].shape[0] <= 2:
            return self.clusters

        snapshot = self.clusters.copy()

        if direction == 0:
            snapshot[worm_idx] = snapshot[worm_idx][1:]
        else:
            snapshot[worm_idx] = snapshot[worm_idx][:-1]

        last_cost = self.loss(snapshot, n_samples)
        if last_cost / init_cost < alpha2:
            return snapshot

        return self.clusters

    def learn(self, epochs, init_nodes, lam, lr, batch_size, mh=100):
        self.lam = lam
        self.__init_clusters(init_nodes)
        self.cost = []

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
                segments *= self.lam * lr * winner_worm_count / batch_size
                self.clusters[winner_worm_idx][:-1] += segments
                self.clusters[winner_worm_idx][1:] -= segments

            for i in range(batch_size):
                winner_worm = winner_worm_idxs[i]
                winner_idx = winner_idxs[i]
                self.clusters[winner_worm][winner_idx] += diffs[winner_worm][winner_idx, i] * lr / batch_size

            if epoch % mh == 0:
                for k in range(self.out_dim):
                    self.clusters = self.__mh(k, 0, n_samples=batch_size)
                    self.clusters = self.__mh(k, 1, n_samples=batch_size)

    def loss(self, clusters, n_samples):
        x_batch = self.data[np.random.randint(self.data.shape[0], size=n_samples)]
        diffs = [x_batch - worm[:, np.newaxis] for worm in self.clusters]
        dists = [np.einsum('ijk,ijk->ij', df, df) for df in diffs]
        min_dists = np.array([np.min(dt, axis=0) for dt in dists])
        winner_worm_idxs = np.argmin(min_dists, axis=0)
        winner_idxs = np.array([np.argmin(dists[winner_worm_idxs[i]][:, i]) for i in range(x_batch.shape[0])])

        mse = np.mean([dists[winner_worm_idxs[i]][winner_idx, i] for i, winner_idx in enumerate(winner_idxs)])

        prior_error = 0
        for winner_worm_idx, winner_worm_count in zip(*np.unique(winner_worm_idxs, return_counts=True)):
            segments = self.clusters[winner_worm_idx][1:] - self.clusters[winner_worm_idx][:-1]
            prior_error += self.lam * np.mean(np.einsum('ij,ij->i', segments, segments) / n_samples)

        return mse + prior_error
