import commlib as cl
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

#x-axis from 2 to 7 with 100 points that are evenly spaced
x = np.linspace(2, 7, num = 1000)
y = cl.Qfunction(x)

#QUESTION A

#A1, A2, A3
q1 = np.exp(-(x**2) / 2)
q2 = ( 1/4 * np.exp(-(x**2)) ) + ( 1/4 * np.exp(-(x**2) / 2) )
q3 = ( 1/12 * np.exp(-(x**2) / 2) ) + ( 1/4 * np.exp(-(2*x**2) / 3) )

#close all plots  
plt.close('all')

#figure 1, Q(x) and Q1(x)
plt.figure(1)
plt.title('Q(x) / Q1(x)')
plt.xlabel('x')
plt.ylabel('y = Q(x)')
plt.yscale("log")
plt.plot(x, y, linestyle = 'solid', label = 'Q(x)')
plt.plot(x, q1, linestyle = 'dashed', label = 'Q1(x)')
plt.legend()

#figure 2, Q(x) and Q2(x)
plt.figure(2)
plt.title('Q(x) / Q2(x)')
plt.xlabel('x')
plt.ylabel('y = Q(x)')
plt.yscale("log")
plt.plot(x, y, linestyle = 'solid', label = 'Q(x)')
plt.plot(x, q2, linestyle = 'dashed', label = 'Q2(x)')
plt.legend()

#figure 3, Q(x) and Q3(x)
plt.figure(3)
plt.title('Q(x) / Q3(x)')
plt.xlabel('x')
plt.ylabel('y = Q(x)')
plt.yscale("log")
plt.plot(x, y, linestyle = 'solid', label = 'Q(x)')
plt.plot(x, q3, linestyle = 'dashed', label = 'Q3(x)')
plt.legend()

plt.show()

#QUESTION B

#Q(1), Q(2), Q(3), y = Q(x)
e1 = abs(q1 - y) / abs(y)
e2 = abs(q2 - y) / abs(y)
e3 = abs(q3 - y) / abs(y)

#trapz function calculates the integral
print("e1 = {}".format(np.trapz(e1, x)))
print("e2 = {}".format(np.trapz(e2, x)))
print("e3 = {}".format(np.trapz(e3, x)))

#QUESTION C

def Qinv(y, tol = 1e-6, max_iter = 100):
    # Initial guess for x based on Q(x) = 0.5 * erfc(x / sqrt(2))
    x = np.sqrt(2) * sp.erfcinv(2 * y)
    
    for _ in range(max_iter):
        q_x = 0.5 * sp.erfc(x / np.sqrt(2))
        error = q_x - y

        if abs(error) < tol:
            return x
        
        # Update x using Newton's method
        x -= error / (np.sqrt(2) * np.exp(-x ** 2 / 2) / np.pi)
    
    raise RuntimeError("Qinv did not converge")

# Test the Qinv function
y = 0.01
x_approx = Qinv(y)
print("Approximate x for y = {} is, {:.3f}".format(y, x_approx))

# Calculate the accuracy by comparing Q(x) and Qinv(Q(x))
x_val = np.linspace(2, 7, num = 1000)
q_val = 0.5 * sp.erfc(x_val / np.sqrt(2))
x_reconstructed = [Qinv(q) for q in q_val]

# Calculate the maximum absolute error between x_values and x_reconstructed
max_error = np.max(np.abs(x_val - x_reconstructed))
print("Maximum absolute error: {}".format(max_error))