import random

def generate_distance_matrix(n):
    distance_matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = random.randint(1, 20)  # Adjust the range as needed

    return distance_matrix

def calculate_total_distance(tour, distance_matrix):
    total_distance = 0
    n = len(tour)

    for i in range(n):
        total_distance += distance_matrix[tour[i]-1][tour[(i+1)%n]-1]

    return total_distance

# Generate a random 10x10 distance matrix
random_distance_matrix = generate_distance_matrix(10)

# Print the generated distance matrix
print("Generated Distance Matrix:")
for row in random_distance_matrix:
    print(row)

# Example tour
tour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Calculate total distance of the tour
total_distance = calculate_total_distance(tour, random_distance_matrix)
print(f"\nTotal distance of the tour: {total_distance}")
