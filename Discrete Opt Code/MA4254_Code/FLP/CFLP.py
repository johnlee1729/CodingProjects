import numpy as np
import pulp
import pandas as pd

# WARNING: THIS CODE TAKE A LONG TIME TO RUN (AROUND 5 MINS)

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["Run", "Optimal Value (MILP)", "Optimal Value (Relaxed LP)"])

# Create a counter to keep track of the number of times the solutions are the same
same_solution_count = 0

# Run the code 100 times
for run in range(100):
    # Generate random data for this run
    np.random.seed(run)  # Set seed for reproducibility
    m = 25
    n = 15
    x_c, y_c = np.random.rand(m), np.random.rand(m)
    x_f, y_f = np.random.rand(n), np.random.rand(n)
    f = np.ones(n)
    c = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            c[i, j] = np.linalg.norm([x_c[i] - x_f[j], y_c[i] - y_f[j]])
    a = np.ones(m)
    r = 2

    # MILP
    model = pulp.LpProblem("Facility_Location_Problem", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(n)), cat='Binary')
    y = pulp.LpVariable.dicts("y", range(n), cat='Binary')
    for i in range(m):
        model += pulp.lpSum(x[i, j] for j in range(n)) == 1
    for j in range(n):
        model += pulp.lpSum(x[i, j] for i in range(m)) <= r * y[j]
    model += pulp.lpSum(f[j] * y[j] + c[i, j] * x[i, j] for i in range(m) for j in range(n))
    model.solve()
    optimal_value_milp = pulp.value(model.objective)

    # Relaxed LP
    model_relaxed = pulp.LpProblem("Relaxed_Facility_Location_Problem", pulp.LpMinimize)
    x_relaxed = pulp.LpVariable.dicts("x_relaxed", ((i, j) for i in range(m) for j in range(n)), lowBound=0, upBound=1, cat='Continuous')
    y_relaxed = pulp.LpVariable.dicts("y_relaxed", range(n), cat='Binary')
    for i in range(m):
        model_relaxed += pulp.lpSum(x_relaxed[i, j] for j in range(n)) == 1
    for j in range(n):
        model_relaxed += pulp.lpSum(x_relaxed[i, j] for i in range(m)) <= r * y_relaxed[j]
    model_relaxed += pulp.lpSum(f[j] * y_relaxed[j] + c[i, j] * x_relaxed[i, j] for i in range(m) for j in range(n))
    model_relaxed.solve()
    optimal_value_relaxed = pulp.value(model_relaxed.objective)

    # Append results to DataFrame
    results_df = results_df.append({"Run": run+1, "Optimal Value (MILP)": optimal_value_milp, 
                                    "Optimal Value (Relaxed LP)": optimal_value_relaxed}, ignore_index=True)

    # Check if the solutions are the same
    if np.isclose(optimal_value_milp, optimal_value_relaxed):
        same_solution_count += 1

# Print the results table
print(results_df)

# Print the number of times the solutions are the same
print(f"Number of times solutions are the same: {same_solution_count}")
