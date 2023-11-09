import numpy as np
import matplotlib.pyplot as plt
from pulp import LpVariable, lpSum, LpBinary, LpMinimize, LpProblem, value

# Number of clients
m = 25
# Number of facility locations
n = 15

# Clients' locations
x_c, y_c = np.random.rand(m), np.random.rand(m)

# Facilities' potential locations
x_f, y_f = np.random.rand(n), np.random.rand(n)

# Fixed costs
f = np.ones(n)

# Distance
c = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        c[i, j] = np.linalg.norm([x_c[i] - x_f[j], y_c[i] - y_f[j]])

# Create LP problem
model = LpProblem(name="FLP", sense=LpMinimize)

# Define decision variables
y = {j: LpVariable(name=f"y_{j}", cat=LpBinary) for j in range(1, n+1)}
x = {(i, j): LpVariable(name=f"x_{i}_{j}", cat=LpBinary) for i in range(1, m+1) for j in range(1, n+1)}

# Objective function
model += lpSum(f[j-1] * y[j] for j in range(1, n+1)) + lpSum(c[i-1, j-1] * x[i, j] for i in range(1, m+1) for j in range(1, n+1))

# Constraints
for i in range(1, m+1):
    model += lpSum(x[i, j] for j in range(1, n+1)) == 1

for i in range(1, m+1):
    for j in range(1, n+1):
        model += x[i, j] <= y[j]

# Solve the problem
model.solve()

print("Optimal value:", value(model.objective))

# Extract solution
x_is_selected = {(i, j): value(x[i, j]) for i in range(1, m+1) for j in range(1, n+1)}
y_is_selected = {j: value(y[j]) for j in range(1, n+1)}

# Visualize the solution

# Plot clients and facilities
plt.scatter(x_c, y_c, label="Clients", color="blue", marker="o")
plt.scatter(x_f, y_f, label="Facility", color="white", marker="*", s=100, edgecolors="red", linewidth=2)

# Plot connections and selected facilities
for i in range(1, m+1):
    for j in range(1, n+1):
        if x_is_selected[(i, j)]:
            plt.plot([x_c[i-1], x_f[j-1]], [y_c[i-1], y_f[j-1]], color="black")

for j in range(1, n+1):
    if y_is_selected[j]:
        plt.scatter(x_f[j-1], y_f[j-1], color="red", s=100, edgecolors="red", linewidth=2)

# Set legend position to top right
plt.legend(loc="upper right")

plt.show()

