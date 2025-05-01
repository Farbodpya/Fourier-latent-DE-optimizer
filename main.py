import matplotlib.pyplot as plt
from benchmark_functions import functions
from DE_tools import run_de_in_latent_space

dim = 5000
latent_dim = 1
maxiter = 1000

results = {}
final_costs = {}

for name, func in functions.items():
    print(f"Running {name}...")
    cost, history = run_de_in_latent_space(func, dim, latent_dim, maxiter)
    results[name] = history
    final_costs[name] = cost
    print(f"Final Cost for {name}: {cost:.4e}")

# Plotting
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

# Print summary
print("\nFinal Best Costs:")
for name, cost in final_costs.items():
    print(f"{name:15}: {cost:.6e}")
