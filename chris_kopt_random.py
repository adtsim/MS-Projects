#FINAL RANDOM CHRIS V/S K-OPT 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import string
import time
from itertools import combinations

# Constants for k-opt
NUM_ITERATIONS = 100
K_OPT_K = 2

def calculate_distances(points):
    return squareform(pdist(points, metric='euclidean'))

def create_complete_graph(point_count, distances):
    G = nx.Graph()
    for i, j in combinations(range(point_count), 2):
        G.add_edge(i, j, weight=distances[i][j])
    return G

def mst_dfs_path(G, start=0):
    T = nx.minimum_spanning_tree(G)
    path = list(nx.dfs_preorder_nodes(T, source=start))
    path.append(start)
    return path

def calculate_path_distance(path, distances):
    return sum(distances[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))

def calculate_total_graph_distance(distances):
    return np.sum(distances) / 2

def run_k_opt_algorithm(points, distances, start_point, k=2):
    # Placeholder for k-opt algorithm
    num_points = len(points)
    path = np.random.permutation(num_points).tolist()
    if path[0] != start_point:
        path.remove(start_point)
        path.insert(0, start_point)
    path.append(path[0])  # Making it a round trip
    return path, calculate_path_distance(path, distances)

def run_christofides_algorithm(points, distances, start_point):
    complete_graph = create_complete_graph(len(points), distances)
    path_christofides = mst_dfs_path(complete_graph, start=start_point)
    total_distance_christofides = calculate_path_distance(path_christofides, distances)
    return path_christofides, total_distance_christofides

def plot(points, path, distances, title="", show_complete_graph=False, show_all_weights=False):

    plot_start_time = time.time()

    plt.figure(figsize=(10, 10))
    x = points[:, 0]
    y = points[:, 1]

    if show_complete_graph:
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    plt.plot(*zip(*points[[i, j]]), color='gray', linestyle='dashed', zorder=1)
                    if show_all_weights:
                        mid_point = (points[i] + points[j]) / 2
                        plt.text(mid_point[0], mid_point[1], f"{distances[i][j]:.2f}",
                                 fontsize=8, color='black', ha='center', va='center', zorder=2)

    for i in range(len(path) - 1):
        plt.plot(*zip(*points[[path[i], path[i + 1]]]), color='blue', zorder=2)
        if show_all_weights:
            weight = distances[path[i], path[i + 1]]
            plt.text((x[path[i]] + x[path[i + 1]]) / 2, (y[path[i]] + y[path[i + 1]]) / 2, f"{weight:.2f}",
                     fontsize=8, color='black', weight='bold', ha='center', va='center', zorder=6)

    plt.scatter(x, y, s=100, color='red', zorder=5)

    if path:
        start_point = points[path[0]]
        plt.scatter(start_point[0], start_point[1], s=100, color='green', zorder=5)

    for i, (x_i, y_i) in enumerate(zip(x, y)):
        label = str(i) if i < 26 else string.ascii_uppercase[i // 26 - 1] + string.ascii_uppercase[i % 26]
        plt.annotate(label, (x_i, y_i), color='white', weight='bold',
                     fontsize=8, ha='center', va='center', zorder=6)

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plot_end_time = time.time() - plot_start_time
    print("Plotting time: ", plot_end_time)


def main():

    start_program_time = time.time()

    point_count = int(input("Enter the number of points: "))
    np.random.seed(0)
    points = np.random.rand(point_count, 2)
    distances = calculate_distances(points)

    total_original_distance = calculate_total_graph_distance(distances)
    print(f"Total distance of the original graph: {total_original_distance:.2f}")

    start_point = np.random.randint(point_count)
    print(f"Using {start_point} as the starting point for both algorithms.")

    start_time_christofides = time.time()
    path_christofides, total_distance_christofides = run_christofides_algorithm(points, distances, start_point)
    christofides_time = time.time() - start_time_christofides
    print("Best Path (Christofides):", path_christofides)
    print("Best Path Distance (Christofides):", total_distance_christofides)
    plot(points, path_christofides, distances, title="Christofides Algorithm", show_complete_graph=True, show_all_weights=True)


    start_time_k_opt = time.time()
    path_k_opt, total_distance_k_opt = run_k_opt_algorithm(points, distances, start_point, K_OPT_K)
    k_opt_time = time.time() - start_time_k_opt
    print("Best Path (k-opt):", path_k_opt)
    print("Best Path Distance (k-opt):", total_distance_k_opt)
    plot(points, path_k_opt, distances, title="k-opt Algorithm", show_complete_graph=True, show_all_weights=True)

    program_runtime = time.time() - start_program_time
    print(f"Runtime for Christofides Algorithm: {christofides_time:.8f} seconds")
    print(f"Runtime for k-opt Algorithm: {k_opt_time:.8f} seconds")
    print(f"Total program runtime: {program_runtime:.2f} seconds")

if __name__ == "__main__":
    main()
