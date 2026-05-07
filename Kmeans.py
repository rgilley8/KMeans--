# ================================================================================
#  K-Means Clustering
#
# Name: Grant Gilley
# Date:
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
import pandas as pd
# Here I load in the data that I am analyzing
df = pd.read_csv(".venv\\Data\\train.csv")
# This removes any rows with missing values
df = df.dropna()
# This whittles dow the data to just the numerical features that I can use for clustering
X = df[[
    "Popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_in min/ms"
]].values
# This standardizes the feature scales
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ==============================================================================
# SETUP
# ==============================================================================

# set a seed
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


# ==============================================================================
# PART 1 --- BUILDING BLOCKS
# ==============================================================================
# K-Means is just four small pieces. Master these and the rest is free.
# ==============================================================================


def euclidean_distance(a, b):
    """Compute the Euclidean distance between two 1-D points.

    d(a, b) = sqrt( sum_j (a_j - b_j)^2 )"""
    # same metric from KNN
    # K-Means & KNN share quite a lot of machinery
    # np.linalg.norm(a - b)
    return np.sqrt(np.sum((a - b) ** 2))


def initialize_centroids(X, k, rng):
    """Choose k starting centroids by picking k rows of X at random,
    WITHOUT replacement."""
    indices = rng.choice(X.shape[0], size=k, replace=False)

    # if you're ever worried about mutating an array
    # just make a copy
    return X[indices].copy()


def assign_clusters(X, centroids):
    """For every point in X, find the index of the nearest centroid."""
    # calculate our squared distances
    # None is a placeholder (1)
    # X --> shape (N, 1, p)
    # - one slot open for "which centroid"
    # centroids --> shape(1, k, p)
    # - one slot open for "which point"

    # subtracting pairs them up by stretching
    # each array along its open slot
    # (N, 1, p) - (1, k, p) --> (N, k, p)
    # entry [i, j, :] of our result is exactly
    # X[i] - centroids[j]

    # axis = 0 --> "which point" --> length N
    # axis = 1 --> "which centroid" --> length k
    # axis = 2 --> "which feature" --> length p

    # axis = 2 geometrically
    # - square the box elementwise, then sum along
    # axis = 2 (feature axis). That collapses each
    # p-length residual vector into a single scalar
    # (squared length of the vector).
    # Pythagorean Theorem, run in parallel for all
    # N*k (point, centroid) pairs.
    sq_dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    # doing axis = 1 here gives, for each row, the column
    # that has the smallest distance
    # i.e., the index of the nearest centroid
    return np.argmin(sq_dists, axis=1)


def update_centroids(X, labels, k, old_centroids):
    """Recompute each centroid as the MEAN of the points currently assigned
    to it."""
    new_centroids = old_centroids.copy()

    for j in range(k):
        # boolean mask for points currently
        # assigned to cluster j
        members = X[labels == j]
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
    return new_centroids


def compute_inertia(X, labels, centroids):
    """Within-cluster sum of squares --- aka WCSS, aka inertia, aka J.

    J = sum_j sum_{x_i in S_j} ||x_i - mu_j||^2"""
    # this is what K-Means is optimizing
    # both our assignment and our update step
    # can be proven to decrease (or maintain) this J.
    # (N, p) -- residual per point
    diffs = X - centroids[labels]
    return np.sum(diffs**2)


# ==============================================================================
# PART 2 --- FULL ALGORITHM
# ==============================================================================
# Now we assemble the pieces into Lloyd's algorithm.
# ==============================================================================


def kmeans(X, k, rng, max_iter=100, tol=1e-6):
    """Lloyd's algorithm for K-Means clustering."""
    # establish centroids
    centroids = initialize_centroids(X, k, rng)
    # This saves me an initial copy of the centroid locations
    init_centroids = centroids.copy()

    # This will track inertia across iterations of the algorithm
    # Using this I can compare the starting inertia of each initialization method
    # and I can plot how quickly the inertia drops for each method
    inertia_tracker = []

    n_iter = 0  # safeguard

    for n_iter in range(1, max_iter + 1):
        # assign step
        labels = assign_clusters(X, centroids)
        # Now I compute the inertia for this iteration and add it to the tracker
        inertia_tracker.append(compute_inertia(X, labels, centroids))
        # update step
        # key is to keep our old centroids
        # to check for convergence
        old_centroids = centroids.copy()
        centroids = update_centroids(X, labels, k, old_centroids)
        # convergence check
        # Frobenius norm = sqrt of
        # the sum of squared elementwise
        # differences
        shift = np.linalg.norm(centroids - old_centroids)
        # check tolerance
        if shift < tol:
            break

    # final label re-assignment
    labels = assign_clusters(X, centroids)
    inertia = compute_inertia(X, labels, centroids)
    return centroids, labels, inertia, n_iter, inertia_tracker, init_centroids


def plot_clusters(X, labels, centroids, title="K-Means result"):
    """Convenience wrapper for the scatter-plot-with-centroids pattern."""
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="magma", s=25)
    plt.scatter(
        # alternatively could do *centroids.T
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        label="Centroids",
    )
    plt.legend()
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig("kmeans_results_clusters.png")
    plt.show()


# ==============================================================================
# Adding my own Metrics & Visualizations
# ==============================================================================

def plot_convergence(inertia_tracker):
    # This will plot the inertia across each iteration to provide a visual of its
    # convergence. This can be used to compare the convergence behavior of different
    # intialization methods
    plt.figure(figsize=(7, 5))
    plt.plot(list(range(1, len(inertia_tracker)+1)), inertia_tracker)
    plt.xlabel("Iteration")
    plt.ylabel("Inertia")
    plt.title("Inertia Convergence Across Iterations")
    plt.xticks(range(1, len(inertia_tracker) + 1))
    plt.tight_layout()
    plt.savefig("kmeans_convergence.png")
    plt.show()

def plot_init_sensitivity():
    """This Displays the sensitivity of K-Means to random seeding by running the algorthim
    20 times with different seeds and plotting the final interias for each run."""
    inertias_20 = []
    for seed in range(20):
        _, _, inertia, _, _, _ = kmeans(X, 11, np.random.default_rng(seed))
        inertias_20.append(inertia)
    print(f"Max inertia: {max(inertias_20):.2f}")
    print(f"Min inertia: {min(inertias_20):.2f}")
    print(f"Mean inertia: {np.mean(inertias_20):.2f}")

    plt.figure(figsize=(7, 4))
    plt.hist(inertias_20, bins=10, edgecolor="black")
    plt.xlabel("Final Inertia")
    plt.ylabel("Count")
    plt.title("Inertia Across 20 Random Initializations")
    plt.tight_layout()
    plt.savefig("kmeans_init_sensitivity.png")
    plt.show()


def plot_init_centroids(X, init_centroids, title="Initial Centroid Locations"):
    # This will plot the initial centroid locations
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=25, alpha=0.6)
    plt.scatter(
        # alternatively could do *centroids.T
        init_centroids[:, 0],
        init_centroids[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        label="InitialCentroids",
    )
    plt.legend()
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig("kmeans_initial_centroids.png")
    plt.show()

def run_metrics():
    """In this function, I run all the metrics needed for comparison."""
    centroids, labels, inertia, n_iter, inertia_tracker, init_centroids = kmeans(X, rng=rng, k=11)
    print(f"Final Inertia: {inertia:.2f}")
    print(f"Iterations: {n_iter}")
    plot_init_centroids(X, init_centroids)
    plot_clusters(X, labels, centroids, title="K-Means on blobs (k = 11)")
    plot_convergence(inertia_tracker)
    plot_init_sensitivity()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_metrics()

    print("\n" + "=" * 60)
    print("Done!")
