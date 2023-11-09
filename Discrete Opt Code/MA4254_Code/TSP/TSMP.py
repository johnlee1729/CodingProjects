import random
import math
import matplotlib.pyplot as plt

def generate_random_cities(n):
    return [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(n)]

def euclidean_distance(point1, point2):
    # Euclidean distance calculation
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def total_distance(solution, distances):
    #total distance calculation
    total = 0
    for i in range(len(solution) - 1):
        total += distances[solution[i]][solution[i+1]]
    total += distances[solution[-1]][solution[0]]
    return total

def generate_initial_solution(num_cities):
    return list(range(num_cities))

def generate_perturbed_tour(tour):
    n = len(tour)
    i = random.randint(0, n - 2)  # Randomly select i
    j = random.randint(i + 1, n - 1)  # Randomly select j such that i < j

    perturbed_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

    return perturbed_tour

def simulated_annealing(distances, max_iterations, initial_temperature, cooling_rate):
    num_cities = len(distances)
    current_solution = generate_initial_solution(num_cities)
    current_distance = total_distance(current_solution, distances)
    best_solution = current_solution
    best_distance = current_distance
    temperature = initial_temperature

    for iteration in range(max_iterations):
        new_solution = generate_perturbed_tour(current_solution)
        new_distance = total_distance(new_solution, distances)

        if new_distance < current_distance:
            current_solution = new_solution
            current_distance = new_distance
            if new_distance < best_distance:
                best_solution = new_solution
                best_distance = new_distance
        else:
            if random.random() < math.exp((current_distance - new_distance) / temperature):
                current_solution = new_solution
                current_distance = new_distance

        temperature *= cooling_rate

    return best_solution, best_distance

# Generate random cities
cities = generate_random_cities(50)

# Generate a distance matrix based on Euclidean distances
distances = [[euclidean_distance(cities[i], cities[j]) for j in range(50)] for i in range(50)]

# Define the parameters
max_iterations = 10000
initial_temperature = 10000.0
cooling_rate = 0.995

# Run simulated annealing
best_solution, best_distance = simulated_annealing(distances, max_iterations, initial_temperature, cooling_rate)

print(f"Best solution: {best_solution}")
print(f"Total distance: {best_distance}")

def plot_solution(cities, solution):
    x = [city[0] for city in cities]
    y = [city[1] for city in cities]
    
    plt.plot(x, y, 'ro')  # Plot cities as dots
    
    # Add lines between cities in the solution
    for i in range(len(solution) - 1):
        plt.plot([x[solution[i]], x[solution[i+1]]], [y[solution[i]], y[solution[i+1]]], 'k-')
    plt.plot([x[solution[-1]], x[solution[0]]], [y[solution[-1]], y[solution[0]]], 'k-')  # Connect last city to first

    plt.show()

plot_solution(cities, best_solution)