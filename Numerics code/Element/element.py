import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def finite_element(lb, rb, N, mu, rhs, lbv, rbv):
    dx = (rb - lb) / N
    u = np.zeros((N+1, N+1))
    u[0, :] = lbv
    u[-1, :] = rbv

    mu_int = np.zeros(N)
    for i in range(N):
        mu_int[i] = quad(lambda y: mu(lb + i*dx, y), lb + i*dx, lb + (i+1)*dx)[0]

    rhs_int = np.zeros((N-1, N))
    for i in range(N-1):
        x0 = lb + i*dx
        x1 = x0 + dx
        x2 = x1 + dx
        for j in range(N):
            y0 = lb + j*dx
            y1 = y0 + dx
            y2 = y1 + dx
            rhs_int[i, j] = quad(lambda x: quad(lambda y: rhs(x, y) * (x - x0) / dx, y0, y1)[0] + 
                                quad(lambda y: rhs(x, y) * (x2 - x) / dx, y1, y2)[0], x0, x1)[0] 

    A = np.diag((mu_int[:-1] + mu_int[1:]) / (dx*dx)) - \
        np.diag(mu_int[1:-1] / (dx * dx), -1) - \
        np.diag(mu_int[1:-1] / (dx * dx), 1)

    for j in range(N):
        rhs_int[0, j] += mu_int[0] * lbv / (dx * dx)
        rhs_int[-1, j] += mu_int[-1] * rbv / (dx * dx)

    for j in range(N):
        u[1:-1, j] = np.linalg.solve(A, rhs_int[:, j])

    return u

def mu(x, y):
    return 1  

def rhs(x, y):
    return 400 * (x**4 - y**4) * np.sin(20 * x * y)

def u_x1(y):
    return 20*y*(1-y**2)*np.cos(20*y) + 2*np.sin(20*y)

def u_y0(x):
    return 20*x*(x**2 - 1)*np.cos(20*x) - 2*np.sin(20*x)

def true_solution(x, y):
    return (x**2 - y**2) * np.sin(20*x*y)

lb, rb = -1, 1
N = 200 
lbv, rbv = 0, 0
u_y = u_y0(np.linspace(-1, 1, N+1))
u_x = u_x1(np.linspace(-1, 1, N+1))

u = finite_element(lb, rb, N, mu, rhs, lbv, rbv)

x = np.linspace(lb, rb, N+1)
y = np.linspace(lb, rb, N+1)
X, Y = np.meshgrid(x, y)

# Calculate true solution
u_true = true_solution(X, Y)

# Calculate numerical error
error = np.abs(u - u_true)

plt.contourf(X, Y, error, cmap='viridis')
plt.colorbar()
plt.title('Numerical Error')
plt.xlabel('x')
plt.ylabel('y')
plt.show()