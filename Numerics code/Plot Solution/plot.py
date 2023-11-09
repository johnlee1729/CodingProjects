import numpy as np
import matplotlib.pyplot as plt

def u(x, y):
    return (x**2 - y**2) * np.sin(20 * x * y)

x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)

X, Y = np.meshgrid(x, y)
Z = u(X, Y)

plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar()
plt.title('u(x, y) = (x^2 - y^2) * sin(20xy)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
