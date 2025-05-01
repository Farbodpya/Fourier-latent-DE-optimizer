from scipy.optimize import differential_evolution
from .fourier_projection import decode

def run_de_in_latent_space(func, dim, latent_dim, bounds, maxiter=1000):
    costs_per_generation = []

    def objective_1d(z):
        x = decode(z, dim).flatten()
        return func(x)

    def callback_de(xk, convergence):
        cost = objective_1d(xk)
        costs_per_generation.append(cost)

    result = differential_evolution(
        objective_1d, bounds, strategy='best1bin',
        maxiter=maxiter, callback=callback_de, disp=True
    )

    return result, costs_per_generation
