import numpy as np
from pulp import LpVariable, lpSum, LpBinary, LpMinimize, LpProblem, value
from tabulate import tabulate


#Since AFLP is equvilent to FLP I have omited that code. This code is for the relaxed AFLP
# Define a function to solve the Relaxed AFLP problem
def solve_raflp():
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
    model = LpProblem(name="RAFLP", sense=LpMinimize)

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

    return value(model.objective)

# Run the simulation 100 times
results = []

for _ in range(100):
    result = solve_raflp()
    results.append(result)

# Print all the answers in a table
table = [["Run", "Result"]]
for i, result in enumerate(results, 1):
    table.append([i, result])

# Calculate and print the average result
average_result = np.mean(results)
table.append(["Average", average_result])

# Print the table
print(tabulate(table, headers="firstrow", tablefmt="grid"))