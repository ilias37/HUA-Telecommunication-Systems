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
        q_x = 0.5 * sp.erfc( x / np.sqrt(2) )
        error = q_x - y

        if abs(error) < tol:
            return x
        
        # Update x using Newton's method
        x -= error / ( np.sqrt(2) * np.exp(-x ** 2 / 2) / np.pi )
    
    raise RuntimeError("Qinv did not converge")

# test the Qinv function
# create an array of SNR values (x values) in dB
snr_values = np.linspace(-10, 20, num=1000)  #in dB
x_values = np.sqrt(10 ** (snr_values / 10))  # Convert to linear scale

# calculate the accuracy for each SNR value
q_values = 0.5 * sp.erfc(x_values / np.sqrt(2))

# calculate the accuracy for each SNR value
x_reconstructed = [Qinv(q) for q in q_values]
error = np.max(np.abs( snr_values - 10 * np.log10(x_reconstructed) ) )

print("Maximum absolute error:", error, "dB")