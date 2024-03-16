import csv
import networkx as nx
import matplotlib.pyplot as plt
import time
from math import radians, sin, cos, sqrt, asin

def read_cities_from_csv(filename, num_rows):
    """Read city coordinates from a CSV file."""
    cities = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if present
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            city = (row[0], float(row[1]), float(row[2]))
            cities.append(city)
    return cities


def haversine(coord1, coord2):
    """Calculate the great-circle distance between two points on the Earth."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Modify the functions to use haversine instead of Euclidean distance

def plot_all_cities_connected(cities, title):
    """Plot all cities as a connected graph."""
    G = nx.Graph()
    total_distance = 0  # Initialize total distance
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            city1 = cities[i]
            city2 = cities[j]
            distance = haversine((city1[1], city1[2]), (city2[1], city2[2]))  # Use haversine method
            total_distance += distance  # Add distance to total distance
            G.add_edge(city1[0], city2[0], weight=distance)

    plt.figure(figsize=(10, 6))
    pos = {city[0]: (city[1], city[2]) for city in cities}
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='blue')

    # Color the starting point with a different color
    starting_point = cities[0]
    plt.plot(starting_point[1], starting_point[2], 'ro')

    # Add edge labels representing distances
    edge_labels = nx.get_edge_attributes(G, 'weight')
    rounded_edge_labels = {edge: round(weight, 2) for edge, weight in edge_labels.items()}
    for edge, weight in rounded_edge_labels.items():
        position = (pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        plt.text(position[0], position[1], str(weight), fontsize=10, color='red', ha='center', va='center')

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

    print("Total Distance of Complete Graph:", round(total_distance, 2))  # Print total distance of complete graph

def plot_optimal_tour(cities, optimal_tour):
    """Plot the optimal tour."""
    plt.figure(figsize=(10, 6))
    total_distance = 0  # Initialize total distance
    for i in range(len(optimal_tour) - 1):
        city1 = cities[optimal_tour[i]]
        city2 = cities[optimal_tour[i+1]]
        distance = haversine((city1[1], city1[2]), (city2[1], city2[2]))  # Use haversine method
        total_distance += distance  # Add distance to total distance
        plt.plot([city1[1], city2[1]], [city1[2], city2[2]], 'b-')
        plt.text((city1[1] + city2[1]) / 2, (city1[2] + city2[2]) / 2, str(round(distance, 2)), fontsize=10, ha='center', va='center')
        # Add arrow indicating direction of traversal
        plt.arrow(city1[1], city1[2], city2[1] - city1[1], city2[2] - city1[2], shape='full', lw=0, length_includes_head=True, head_width=0.03, color='b')

    city1 = cities[optimal_tour[-1]]
    city2 = cities[optimal_tour[0]]
    distance = haversine((city1[1], city1[2]), (city2[1], city2[2]))  # Use haversine method
    total_distance += distance  # Add distance to total distance
    plt.plot([city1[1], city2[1]], [city1[2], city2[2]], 'b-')
    plt.text((city1[1] + city2[1]) / 2, (city1[2] + city2[2]) / 2, str(round(distance, 2)), fontsize=10, ha='center', va='center')
    # Add arrow indicating direction of traversal
    plt.arrow(city1[1], city1[2], city2[1] - city1[1], city2[2] - city1[2], shape='full', lw=0, length_includes_head=True, head_width=0.03, color='b')

    for city in cities:
        plt.plot(city[1], city[2], 'bo')
        plt.text(city[1], city[2], city[0], fontsize=12, ha='right')

    # Color the starting point with a different color
    starting_point = cities[optimal_tour[0]]
    plt.plot(starting_point[1], starting_point[2], 'ro')
    plt.text(starting_point[1], starting_point[2], starting_point[0], fontsize=12, ha='right', color='red')

    plt.title('Optimal Tour')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

    print("Total Distance of Optimal Tour:", round(total_distance, 2))  # Print total distance of optimal tour



def calculate_tour_length(cities, tour):
    """Calculate the total length of the tour."""
    total_length = 0
    for i in range(len(tour)):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i + 1) % len(tour)]]
        distance = haversine((city1[1], city1[2]), (city2[1], city2[2]))  # Use haversine method
        total_length += distance
    return total_length


def k_opt_tsp(cities, k=2):
    """K-opt algorithm for TSP."""
    num_cities = len(cities)
    optimal_tour = list(range(num_cities))
    min_length = calculate_tour_length(cities, optimal_tour)

    improved = True
    start_time = time.time()  # Start measuring runtime
    while improved:
        improved = False
        for i in range(num_cities):
            for j in range(i + 2, num_cities):
                if j - i == k:
                    new_tour = list(optimal_tour)
                    new_tour[i:j] = reversed(new_tour[i:j])
                    length = calculate_tour_length(cities, new_tour)
                    if length < min_length:
                        optimal_tour = new_tour
                        min_length = length
                        improved = True
    end_time = time.time()  # Stop measuring runtime
    runtime = end_time - start_time
    return optimal_tour, min_length, runtime  # Return both the optimal tour, its total distance, and the runtime

# Example usage:
if __name__ == "__main__":
    # Provide the filename of your CSV file
    csv_filename = 'us_cities.csv'

    # Get the number of rows to read from the user
    n = int(input("Enter the number of rows to read: "))

    # Read city coordinates from the first 'n' rows of the CSV file
    cities = read_cities_from_csv(csv_filename, n)

    # Plot all cities as a connected graph
    plot_all_cities_connected(cities, 'All Cities Connected')

    # Get the optimal tour using k-opt algorithm (k=2 by default)
    start_time_total = time.time()  # Start measuring total runtime
    optimal_tour, total_distance, runtime = k_opt_tsp(cities)
    end_time_total = time.time()  # Stop measuring total runtime
    print("Optimal Tour:", [city[0] for city in cities])
    print("Runtime of k-opt algorithm:", runtime, "seconds")
    print("Total runtime of the program:", end_time_total - start_time_total, "seconds")

    # Plot the optimal tour
    plot_optimal_tour(cities, optimal_tour)
