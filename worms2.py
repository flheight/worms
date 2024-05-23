import numpy as np
from sklearn.cluster import KMeans
import torch

class Worms:
    def __init__(self, k):
        self.out_dim = k

    def load_data(self, data):
        self.data = data

    def __init_clusters(self, init_nodes):
        kmeans = KMeans(self.out_dim).fit(self.data)
        var = 1e-4 * np.eye(self.data.shape[1])
        self.clusters = [np.random.multivariate_normal(center, var, init_nodes) for center in kmeans.cluster_centers_]

    def __mh(self, worm_idx, direction, alpha1=0.95, alpha2=0.95):
        init_cost = self.loss(self.clusters)

        snapshot = self.clusters.copy()
        var = 1e-4 * np.eye(self.data.shape[1])

        if direction == 0:
            new = np.random.multivariate_normal(self.clusters[worm_idx][0], var, 1)
            snapshot[worm_idx] = np.vstack((new, snapshot[worm_idx]))
        else:
            new = np.random.multivariate_normal(self.clusters[worm_idx][-1], var, 1)
            snapshot[worm_idx] = np.vstack((snapshot[worm_idx], new))

        last_cost = self.loss(snapshot)
        if last_cost / init_cost < alpha1:
            return snapshot

        if self.clusters[worm_idx].shape[0] <= 2:
            return self.clusters

        snapshot = self.clusters.copy()

        if direction == 0:
            snapshot[worm_idx] = snapshot[worm_idx][1:]
        else:
            snapshot[worm_idx] = snapshot[worm_idx][:-1]

        last_cost = self.loss(snapshot)
        if last_cost / init_cost < alpha2:
            return snapshot

        return self.clusters

    def learn(self, epochs, init_nodes, lam, mu, lr, mh=100):
        self.lam = lam
        self.mu = mu
        self.__init_clusters(init_nodes)

        for epoch in range(epochs):
            x = self.data[np.random.randint(self.data.shape[0])]

            diff = [x - worm for worm in self.clusters]
            dist = [np.einsum('ij,ij->i', df, df) for df in diff]

            winner_worm_idx = np.argmin([np.min(dist) for dist in dist])
            winner_idx = np.argmin(dist[winner_worm_idx])

            segments = self.clusters[winner_worm_idx][1:] - self.clusters[winner_worm_idx][:-1]

            segments *= self.lam * lr
            self.clusters[winner_worm_idx][:-1] += segments
            self.clusters[winner_worm_idx][1:] -= segments

            segments *= self.mu / (2 * self.lam)
            self.clusters[winner_worm_idx][:-2] -= segments[1:]
            self.clusters[winner_worm_idx][:-1] += segments
            self.clusters[winner_worm_idx][1:] -= segments
            self.clusters[winner_worm_idx][2:] += segments[:-1]

            self.clusters[winner_worm_idx][winner_idx] += lr * diff[winner_worm_idx][winner_idx]

            if epoch % mh == 0:
                for k in range(self.out_dim):
                    self.clusters = self.__mh(k, 0)
                    self.clusters = self.__mh(k, 1)

    def loss(self, clusters):
        x = self.data[np.random.randint(self.data.shape[0])]

        diff = [x - worm for worm in self.clusters]
        dist = [np.einsum('ij,ij->i', df, df) for df in diff]

        winner_worm_idx = np.argmin([np.min(dist) for dist in dist])
        winner_idx = np.argmin(dist[winner_worm_idx])

        mse = dist[winner_worm_idx][winner_idx]

        segments = clusters[winner_worm_idx][1:] - clusters[winner_worm_idx][:-1]

        dist_error = self.lam * np.sum(np.einsum('ij,ij->i', segments, segments))

        smoothness_error = -self.mu * np.sum(np.einsum('ij,ij->i', segments[1:], segments[:-1]))

        return mse + dist_error + smoothness_error
