import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n = 15  # Number of facilities
m = 25  # Number of customers

# Generate random facility locations
facilities_x = np.random.rand(n)
facilities_y = np.random.rand(n)

# Generate random customer locations
customers_x = np.random.rand(m)
customers_y = np.random.rand(m)

# Initialize distance matrix
distance_matrix = np.zeros((m, n))

# Compute distances and fill the distance matrix
for i in range(m):
    for j in range(n):
        dx = customers_x[i] - facilities_x[j]
        dy = customers_y[i] - facilities_y[j]
        distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)

# Print the distance matrix
print("Distance Matrix:")
print(distance_matrix)

plt.scatter(customers_x, customers_y, label="Clients", color="blue", marker="o")
plt.scatter(facilities_x, facilities_y, label="Facility", color="white", marker="*", s=100, edgecolors="red", linewidth=2)
plt.legend(loc="upper right")
plt.show()
