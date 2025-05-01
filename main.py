import numpy as np
import matplotlib.pyplot as plt
from benchmark_functions import sphere, rastrigin, rastrigin_ii, ackley, griewank, zakharov
from DE_tools import run_de_in_latent_space

# Parameters
dim = 5000
latent_dim = 1
maxiter = 1000
bounds = [(-10, 10)] * latent_dim

functions = {
    'Sphere': sphere,
    'Rastrigin': rastrigin,
    'Rastrigin II': rastrigin_ii,
    'Ackley': ackley,
    'Griewank': griewank,
    'Zakharov': zakharov
}

results = {}
final_costs = {}

for name, func in functions.items():
    print(f"Running {name}...")
    result, costs = run_de_in_latent_space(func, dim, latent_dim, bounds, maxiter)
    results[name] = costs
    final_costs[name] = result.fun
    print(f"Final Cost for {name}: {result.fun:.4e}")

# Plot
plt.figure(figsize=(12, 7))
for name, costs in results.items():
    plt.plot(costs, label=name)

plt.yscale("log")
plt.xlabel("Generation")
plt.ylabel("Best Cost (log scale)")
plt.title("Differential Evolution in Fourier-Latent Space")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("de_fourier_latent_convergence.png", dpi=300)
plt.show()

# Summary
print("\nFinal Best Costs:")
for name, cost in final_costs.items():
    print(f"{name:15}: {cost:.6e}")
