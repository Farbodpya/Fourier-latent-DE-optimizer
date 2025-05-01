from scipy.optimize import differential_evolution
from fourier_projection import decode

def run_de_in_latent_space(func, dim, latent_dim, maxiter=1000):
    bounds = [(-10, 10)] * latent_dim
    costs_per_generation = []

    def objective_1d(z):
        x = decode(z, dim).flatten()
        return func(x)

    def callback(xk, convergence):
        cost = objective_1d(xk)
        costs_per_generation.append(cost)

    result = differential_evolution(objective_1d, bounds, strategy='best1bin',
                                    maxiter=maxiter, callback=callback, disp=True)

    return result.fun, costs_per_generation
