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

# ==============================================================================
# SETUP
# ==============================================================================

# set a seed
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# make blobs
X, y_true = make_blobs(
    n_samples=300, centers=4, cluster_std=0.8, random_state=RNG_SEED  # k
)

# plot our data
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=25)
plt.title("Ground Truth (Blobs)")
plt.show()

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

    # This will track inertia across iterations of the algorithm
    # Using this I can compare the starting inertia of each initialization method
    # and I can plot how quickly the inertia drops for each method
    inertia_tracker = []

    n_iter = 0  # safeguard

    for n_iter in range(1, max_iter + 1):
        # assign step
        labels = assign_clusters(X, centroids)
        # update step
        # key is to keep our old centroids
        # to check for convergence
        old_centroids = centroids.copy()
        centroids = update_centroids(X, labels, k, old_centroids)
        # Now I compute the inertia for this iteration and add it to the tracker
        inertia_tracker.append(compute_inertia(X, labels, centroids))
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
    return centroids, labels, inertia, n_iter, inertia_tracker


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
    plt.show()


def demo_part2():
    """Part 2.1 + 2.2 --- run K-Means with k=4 on the well-behaved blobs."""
    centroids, labels, inertia, n_iter, inertia_tracker = kmeans(X, rng=rng, k=4)
    print([f'{x:.2f}' for x in inertia_tracker])
    print(f"Final Inertia: {inertia:.2f}")
    print(f"Iterations: {n_iter}")
    plot_clusters(X, labels, centroids, title="K-Means on blobs (k = 4)")
    plot_convergence(inertia_tracker)


# ==============================================================================
# PART 3 --- ELBOW METHOD
# ==============================================================================


def demo_part3():
    """Part 3 --- elbow plot from k=1 to k=10."""
    K_range = range(1, 11)
    inertias = []
    for k in K_range:
        best = np.inf
        for seed in range(10):
            # run 10 diff random seeds
            # per k and take the best inertia
            _, _, inertia, _ = kmeans(X, k, rng=np.random.default_rng(seed))
            if inertia < best:
                best = inertia
        inertias.append(best)

    plt.figure(figsize=(7, 5))
    plt.plot(list(K_range), inertias)
    plt.axvline(4, color="red", linestyle="--", alpha=0.5, label="True k = 4")
    plt.xlabel("K")
    plt.ylabel("Best inertia (of 10 seeds)")
    plt.title("Elbow Method")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# PART 4 --- CONFRONTING THE PITFALLS
# ==============================================================================
# Each subsection makes one lecture pitfall painfully concrete.
# ==============================================================================


def demo_part4_1_init_sensitivity():
    """4.1 --- different random seeds produce different final inertias."""
    inertias_20 = []
    for seed in range(20):
        _, _, inertia, _, _ = kmeans(X, 4, np.random.default_rng(seed))
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
    plt.show()


def demo_part4_2_feature_scale():
    """4.2 --- what happens when one feature dominates the distance metric."""
    # multiply column 1 by 1000
    X_bad = X.copy()

    X_bad[:, 1] *= 1000

    _, labels_bad, _, _ = kmeans(X_bad, 4, rng=rng)

    # standardize the data
    # we will z-score our X_bad (broken data)
    # z-scoring is scale-invariant so X_bad &
    # X_norm will land in the same spot
    X_std = (X_bad - X_bad.mean(axis=0)) / X_bad.std(axis=0)

    _, labels, _, _ = kmeans(X_std, 4, rng=rng)

    # create 2 plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(X_bad[:, 0], X_bad[:, 1], c=labels_bad, cmap="magma", s=25)
    axes[0].set_title("Unscaled (col 1 x 1000)")
    axes[1].scatter(X_std[:, 0], X_std[:, 1], c=labels, cmap="Blues", s=25)
    axes[1].set_title("Standardized (Fixed)")
    plt.tight_layout()
    plt.show()


def demo_part4_3_outlier():
    """4.3 --- a single outlier pulls a centroid off course."""
    # add one point way away at (50,50)
    X_out = np.vstack([X, [[50, 50]]])

    centroids_out, labels_out, _, _ = kmeans(X_out, 4, rng=rng)

    # show the distribution of points across the 4 clusters
    counts = np.bincount(labels_out)
    print(f"Points per cluster: {counts.tolist()}")
    print(f"Total: {counts.sum()}")

    plot_clusters(
        X_out, labels_out, centroids_out, title="K-Means with one outlier at (50, 50)"
    )


def demo_part4_4_moons():
    """4.4 --- K-Means' spherical assumption fails on non-convex clusters."""
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=RNG_SEED)

    centroids_m, labels_m, _, _ = kmeans(X_moons, k=2, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="PuBu", s=25)
    axes[0].set_title("Ground Truth (two moons)")

    axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_m, cmap="RdBu", s=25)
    axes[1].scatter(*centroids_m.T, c="red", marker="X", s=200, edgecolors="black")
    axes[1].set_title("K-Means Result (Moons)")
    plt.tight_layout()
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
    plt.show()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    demo_part2()
    # demo_part3()
    # demo_part4_1_init_sensitivity()
    # demo_part4_2_feature_scale()
    # demo_part4_3_outlier()
    # demo_part4_4_moons()

    print("\n" + "=" * 60)
    print("Done!")
