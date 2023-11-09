import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product
import pandas as pd

# Define a function named 'run_simulation' that sets up and solves the facility location problem.
def run_simulation():
    num_customers = 25
    num_facilities = 15

    # Generate random customer and facility locations
    customers = [(np.round(np.random.randn(),2), np.round(np.random.randn(),2)) for i in range(num_customers)]
    facilities = [(np.round(np.random.randn(),2), np.round(np.random.randn(),2)) for i in range(num_facilities)]

    # Generate random setup costs for facilities
    setup_cost = [np.random.rand() for i in range(num_facilities)]
    cost_per_mile = 1.0

    # Define a function to compute distance between two locations
    def compute_distance(loc1, loc2):
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        return np.sqrt(dx**2 + dy**2)

    # Generate a list of all possible customer-facility combinations
    cartesian_prod = list(product(range(num_customers), range(num_facilities)))

    # Calculate shipping cost for each combination
    shipping_cost = {(c,f): cost_per_mile * compute_distance(customers[c], facilities[f]) for c, f in cartesian_prod}

    # Create a MIP model
    m_MIP = gp.Model('facility_location_MIP')

    # Define binary decision variables for facility selection
    select = m_MIP.addVars(num_facilities, vtype=GRB.BINARY, name='Select')

    # Define continuous decision variables for assignment
    assign = m_MIP.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name='Assign')

    # Add constraints to the model
    m_MIP.addConstrs((assign[(c,f)] <= select[f] for c, f in cartesian_prod), name='Setup2ship')
    m_MIP.addConstrs((gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers)), name='Demand')

    # Set the objective function
    m_MIP.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

    # Optimize the model
    m_MIP.optimize()

    # Create an LP relaxation of the MIP model
    m_LP = m_MIP.relax()

    # Optimize the LP model
    m_LP.optimize()

    return m_MIP.ObjVal, m_LP.ObjVal

# Initialize lists to store results
mip_results = []
lp_results = []

# Initialize counter for same results
same_results_count = 0

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['MIP Objective', 'LP Relaxation Objective'])

# Run the simulation 100 times
for _ in range(100):
    mip_obj, lp_obj = run_simulation()
    mip_results.append(mip_obj)
    lp_results.append(lp_obj)

    # Check if the results are the same
    if np.isclose(mip_obj, lp_obj, atol=1e-5):
        same_results_count += 1

    # Append the results to the DataFrame
    results_df = results_df.append({'MIP Objective': mip_obj, 'LP Relaxation Objective': lp_obj}, ignore_index=True)

# Display the DataFrame
print(results_df)

# Compare results
avg_mip_obj = sum(mip_results) / len(mip_results)
avg_lp_obj = sum(lp_results) / len(lp_results)

print(f"Average MIP Objective: {avg_mip_obj}")
print(f"Average LP Relaxation Objective: {avg_lp_obj}")
print(f"Number of times results are the same: {same_results_count}")
