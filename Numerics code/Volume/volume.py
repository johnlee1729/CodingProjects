import numpy as np
import matplotlib.pyplot as plt



# Define the domain and grid
nx = ny = 200
Lx, Ly = 2, 2
dx = Lx / nx
dy = Ly / ny

# Initialize grid variables
x = np.linspace(-1+dx/2, 1-dx/2, nx)
y = np.linspace(-1+dy/2, 1-dy/2, ny)
X, Y = np.meshgrid(x, y)

# Define boundary conditions
def boundary_conditions(u):
    u[:, 0] = 0  # u(x,0) = 0
    u[0, :] = 0  # u(0,y) = 0
    u[-1, :] = 20*x[-1]*(x[-1]**2 - 1)*np.cos(20*x[-1]) + 2*np.sin(20*x[-1])  # u_x(1, y)
    u[:, -1] = 20*y[:] * (1-y[:]**2) * np.cos(20*y[:]) + 2*np.sin(20*y[:])  # u_y(x, 1)

# Define the source term
def source_term(x, y):
    return 400*(x**4 - y**4) * np.sin(20*x*y)

# Define the exact solution
def exact_solution(x, y):
    return (x**2 - y**2) * np.sin(20*x*y)

# Initialize solution field
u = np.zeros((nx, ny))

# Define number of iterations
num_iterations = 1000

# Initialize lists to store error norms
errors = []

# Perform finite volume iterations
for _ in range(num_iterations):
    u_new = np.copy(u)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]) - dx*dy*source_term(x[i], y[j])
    
    boundary_conditions(u_new)
    u = np.copy(u_new)

    # Calculate L2 error norm
    error = np.sqrt(np.sum((u - exact_solution(X, Y))**2) * dx * dy)
    errors.append(error)

# Calculate convergence rates
convergence_rates = [np.log(errors[i]/errors[i+1]) / np.log(2) for i in range(len(errors)-1)]

# Plot the solution
plt.contourf(X, Y, u, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Finite Volume Method Solution')
plt.show()

# Plot the error convergence
plt.figure()
plt.plot(range(1, len(errors)+1), errors, marker='o')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('L2 Error Norm')
plt.title('Convergence of Finite Volume Method')
plt.show()

# Print the convergence rates
print("Convergence Rates:")
for i, rate in enumerate(convergence_rates):
    print(f"Iteration {i+1}: {rate:.4f}")

# Plot the numerical error
numerical_error = np.abs(u - exact_solution(X, Y))

plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, numerical_error, cmap='viridis', levels=20)
plt.colorbar(contour, label='Numerical Error')
plt.title("Numerical Error (Volume)")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
