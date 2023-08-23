import commlib as cl
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

# x-axis from 2 to 7 with 1000 points that are evenly spaced
x = np.linspace(2, 7, num = 1000)
y = cl.Qfunction(x)

# QUESTION A

# A1, A2, A3
q1 = np.exp(-(x**2) / 2)
q2 = ( 1/4 * np.exp(-(x**2)) ) + ( 1/4 * np.exp(-(x**2) / 2) )
q3 = ( 1/12 * np.exp(-(x**2) / 2) ) + ( 1/4 * np.exp(-(2*x**2) / 3) )
q_all = [q1, q2, q3]

# close all plots  
plt.close('all')

for i, qi in zip( [1, 2, 3], q_all ):
    # figure i, Q(x) and Qi(x)
    plt.figure(i)
    plt.title('Q(x) / Q{}(x)'.format(i))
    plt.xlabel('x')
    plt.ylabel('y = Q(x)')
    plt.yscale("log")
    plt.plot(x, y, linestyle = 'solid', label = 'Q(x)')
    plt.plot(x, qi, linestyle = 'dashed', label = 'Q{}(x)'.format(i))
    plt.legend()

plt.show()

# QUESTION B
for i, qi in zip( [1, 2, 3], q_all ):
    e = abs(qi - y) / abs(y)            # Qi(x), y = Q(x)
    print("e{} = {}".format(i, np.trapz(e, x)))         # trapz function calculates the integral

# QUESTION C

def Qinv(y, tol = 1e-6, max_iter = 100):

    # Initial guess for x based on Q(x) = 0.5 * erfc(x / sqrt(2))
    x = np.sqrt(2) * sp.erfcinv(2 * y)
    
    for _ in range(max_iter):
        q_x = 0.5 * sp.erfc( x / np.sqrt(2))
        error = q_x - y

        if abs(error) < tol:
            return x
        
        # Update x using Newton's method
        x -= error / ( np.sqrt(2) * np.exp(-x ** 2 / 2) / np.pi)
    
    raise RuntimeError("Qinv did not converge")

# Test the Qinv function
y_values = np.linspace(0.01, 0.1, num = 10)

for y in y_values:
    x_approx = Qinv(y)
    print("\nApproximate x for y = {:.2f} is {:.2f}".format(y, x_approx) )

    # Calculate the accuracy by comparing Q(x) and Qinv(Q(x))
    x_val = np.linspace(2, 7, num = 1000)
    q_val = 0.5 * sp.erfc( x_val / np.sqrt(2) )
    x_reconstructed = [Qinv(q) for q in q_val]

    # Calculate the maximum absolute error between x_values and x_reconstructed
    max_error = np.max( np.abs(x_val - x_reconstructed) )
    print("Maximum absolute error: {}".format(max_error) )