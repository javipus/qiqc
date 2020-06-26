"""
Quantum phase estimation algorithm.
"""

def logp(n, delta):
    """Log2 probability of measuring closest integer phase in a register with n qubits"""
    return -2*n + 2*(np.log2(abs(1-np.exp(2j*np.pi*delta*2**n))) - np.log2(abs(1-np.exp(2j*np.pi*delta))))

# How P(a) changes w/ n and delta
fig, ax = plt.subplots()

for n in range(2,6):
    delta = np.logspace(-10, -n-1, base=2)
    p = 2**logpra(n, delta)
    ax.plot(delta, p, label=str(n))
plt.legend(title='n')
plt.xlabel(r'$\delta$')
plt.ylabel('P(a)')
plt.xscale('log')
plt.show()
