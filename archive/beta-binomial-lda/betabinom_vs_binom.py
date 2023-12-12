import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, betabinom
import matplotlib.pyplot as plt

# Define the total count and three different rates
n = 10
rates = [0.1, 0.2, 0.5]

# Calculate the probability mass function (PMF) for each rate
pmfs = []
for rate in rates:
    pmf = np.zeros(n+1)
    for k in range(n+1):
        pmf[k] = np.math.comb(n, k) * (rate ** k) * ((1 - rate) ** (n - k))
    pmfs.append(pmf)

# Plot the PMFs
plt.figure(figsize=(5,4))
for i in range(len(rates)):
    plt.plot(pmfs[i], "-o", label='Rate = {}'.format(rates[i]))
plt.legend()
plt.xlabel('Junction count')
plt.ylabel('Probability')
plt.title('Binomial distribution with n = 10')
plt.savefig("binomal_example.pdf")
plt.show()
    

# Set the parameters
n = 10
p = 0.1
a1 = 0.1
b1 = 0.9
a2 = 1
b2 = 9

# Generate the data
x = np.arange(n+1)

# Plot the distributions
plt.plot(x, binom.pmf(x, n, p), "-o", label='Binomial(%i, %.1f)' % (n,p))
plt.plot(x, betabinom.pmf(x, n, a1, b1), "-o", label='Beta-Binomial(%i, %.1f, %.1f)' % (n,a1,b1))
plt.plot(x, betabinom.pmf(x, n, a2, b2), "-o", label='Beta-Binomial(%i, %.1f, %.1f)' % (n,a2,b2))
plt.legend( fontsize=12)
plt.xlabel('Junction counts', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.title('Binomial vs. Beta-Binomial distribution', fontsize=16)
plt.savefig("bb_example.pdf")
plt.show()