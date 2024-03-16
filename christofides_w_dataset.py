import folium
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations, chain
from math import radians, sin, cos, asin, sqrt
import time

file_path = 'us_cities.csv'  # Adjust this to the path of your CSV file

def create_complete_graph(point_count, coords):
    G = nx.Graph()
    for i, j in combinations(range(point_count), 2):
        distance = haversine(coords[i], coords[j])
        G.add_edge(i, j, weight=distance)
    return G

def find_odd_degree_nodes(T):
    return [v for v, d in T.degree() if d % 2 == 1]

def form_minimum_weight_matching(G, odd_degree_nodes):
    bipartite_graph = nx.Graph()
    max_weight = max(G[u][v]['weight'] for u, v in combinations(G, 2)) + 1
    for i, u in enumerate(odd_degree_nodes):
        for j, v in enumerate(odd_degree_nodes):
            if j > i:
                weight = G[u][v]['weight']
                bipartite_graph.add_edge(u, v, weight=max_weight - weight)
    matching = nx.algorithms.matching.max_weight_matching(bipartite_graph, maxcardinality=True)
    return matching

def christofides_algorithm(coords):
    start_time = time.time()
    G = create_complete_graph(len(coords), coords)
    T = nx.minimum_spanning_tree(G)
    odd_degree_nodes = find_odd_degree_nodes(T)
    matching = form_minimum_weight_matching(G, odd_degree_nodes)
    multi_graph = nx.MultiGraph(T)
    multi_graph.add_edges_from(matching)
    eulerian_circuit = list(nx.eulerian_circuit(multi_graph))
    path = list(dict.fromkeys(chain.from_iterable(eulerian_circuit)))
    path.append(path[0])  # make it a circuit
    runtime = time.time() - start_time
    return path, runtime

def read_input(file_path, n):
    df = pd.read_csv(file_path, nrows=n)
    cities = df['City'].tolist()
    coords = df[['Latitude', 'Longitude']].to_numpy()
    return cities, coords

def haversine(coord1, coord2):
    """Calculate the great-circle distance between two points on the Earth."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def jitter_coords(coords, jitter_amount=0.02):  # Increased jitter_amount
    """Apply a small random jitter to coordinates to prevent overlap."""
    jittered_coords = []
    for coord in coords:
        lat_jitter = coord[0] + random.uniform(-jitter_amount, jitter_amount)
        lon_jitter = coord[1] + random.uniform(-jitter_amount, jitter_amount)
        jittered_coords.append((lat_jitter, lon_jitter))
    return jittered_coords

def plot_optimal_path_with_folium(cities, coords, path):
    # Create map object centered around the average coordinates
    avg_lat = np.mean([coords[idx][0] for idx in path])
    avg_lon = np.mean([coords[idx][1] for idx in path])
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5)

    # Add points for each city
    for i, city in enumerate(cities):
        folium.CircleMarker(
            location=[coords[i][0], coords[i][1]],
            radius=3,
            color='blue' if i != path[0] else 'red',
            fill=True,
            fill_color='blue' if i != path[0] else 'red',
            fill_opacity=1,
            popup=city
        ).add_to(m)

    # Add lines to represent the optimal path
    for i in range(len(path) - 1):
        start_coord = coords[path[i]]
        end_coord = coords[path[i + 1]]
        folium.PolyLine([(start_coord[0], start_coord[1]), (end_coord[0], end_coord[1])], color='blue', weight=2.5, opacity=1).add_to(m)

    return m


def angle(start_coord, end_coord):
    """Calculate the angle between two coordinates."""
    lat1, lon1 = start_coord
    lat2, lon2 = end_coord
    delta_lon = lon2 - lon1
    x = np.sin(np.radians(delta_lon)) * np.cos(np.radians(lat2))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(delta_lon))
    angle_rad = np.arctan2(x, y)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def main():
    start_total_time = time.time()
    n = int(input("Enter the number of cities: "))
    cities, coords = read_input(file_path, n)
    
    # Calculate runtime of Christofides algorithm
    optimal_path_indices, christofides_runtime = christofides_algorithm(coords)
    optimal_path = [cities[idx] for idx in optimal_path_indices]

    # Calculate total distance of the complete graph
    total_complete_distance = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            total_complete_distance += haversine(coords[i], coords[j])

    # Calculate total distance of the optimal graph
    total_optimal_distance = 0
    for i in range(len(optimal_path_indices) - 1):
        total_optimal_distance += haversine(coords[optimal_path_indices[i]], coords[optimal_path_indices[i + 1]])
    # Add the distance between the last city and the first city to close the loop
    total_optimal_distance += haversine(coords[optimal_path_indices[-1]], coords[optimal_path_indices[0]])

    # Plot all cities as a connected graph
    G = create_complete_graph(len(coords), coords)
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='blue')
    labels = {i: city for i, city in enumerate(cities)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    plt.title('All Cities Connected')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

    #Folium map representation
    folium_map = plot_optimal_path_with_folium(cities, coords, optimal_path_indices)
    display(folium_map)

    total_runtime = time.time() - start_total_time
    print("The optimal path is:")
    print(" -> ".join(optimal_path))
    print(f"Total runtime of the code: {total_runtime:.4f} seconds")
    print(f"Runtime of Christofides algorithm: {christofides_runtime:.4f} seconds")
    print(f"Total distance of the complete graph: {total_complete_distance:.2f} km")
    print(f"Total distance of the optimal graph: {total_optimal_distance:.2f} km")

if __name__ == "__main__":
    main()
