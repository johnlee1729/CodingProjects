import numpy as np
import matplotlib.pyplot as plt

# Define the spatial grid
num_points_x = 200  # Increased number of points in x-direction
num_points_y = 200  # Increased number of points in y-direction
length_x = 2.0
length_y = 2.0
dx = length_x / (num_points_x - 1)
dy = length_y / (num_points_y - 1)

# Initialize the solution field
u = np.zeros((num_points_x, num_points_y))

# Define the source term
x_values = np.linspace(-1, 1, num_points_x)
y_values = np.linspace(-1, 1, num_points_y)
X, Y = np.meshgrid(x_values, y_values)
source = 400 * (X**4 - Y**4) * np.sin(20*X*Y)

# Define the exact solution
def exact_solution(x, y):
    return (x**2 - y**2) * np.sin(20*x*y)

# Apply boundary conditions
u[:, 0] = 0
u[0, :] = 0
u[:, -1] = 20 * x_values * (x_values**2 - 1) * np.cos(20*x_values) - 2 * np.sin(20*x_values)
u[-1, :] = 20 * y_values * (1 - y_values**2) * np.cos(20*y_values) + 2 * np.sin(20*y_values)

# Time-stepping loop
for _ in range(1000):
    u_new = u.copy()
    for i in range(1, num_points_x - 1):
        for j in range(1, num_points_y - 1):
            u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - dx*dy*source[i, j])
    u = u_new.copy()

# Calculate L2 error norm
error = np.sqrt(np.sum((u - exact_solution(X, Y))**2) * dx * dy)

# Print the L2 error norm
print(f"L2 Error Norm: {error}")

# Create a meshgrid for plotting
X, Y = np.meshgrid(x_values, y_values)

# Create a contour plot of u
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, u, cmap='viridis', levels=20)
plt.colorbar(contour, label='u')
plt.title("Finite Difference Method Solution")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Calculate the exact solution
exact_u = exact_solution(X, Y)

# Calculate the numerical error
numerical_error = np.abs(u - exact_u)

# Plot the numerical error
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, numerical_error, cmap='viridis', levels=20)
plt.colorbar(contour, label='Numerical Error')
plt.title("Numerical Error (Difference)")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
