# PSO_PortfolioOptimization

The following repository contains an implementation of a [Particle Swarm Optimization (PSO)](https://en.wikipedia.org/wiki/Particle_swarm_optimization) algorithm from scratch in Python. 

PSO belongs to the family of meta-heuristic algorithms. It explores a large search space using iterative improvements and evaluates many potential solutions before converging to a near-optimal one. Compared to standard optimization algorithms, PSO incurs higher computational cost but offers the advantage of escaping local optima.

## Contents

- **`pso.ipynb`**: Jupyter notebook with a detailed explanation of the algorithm and demonstrations on:
  - The Ackley function  
  - The Rastrigin function  
  - A portfolio optimization problem  

- **`pso.py`**: Raw Python implementation of the PSO algorithm (Object-Oriented-Programming.  

- **`LICENSE`**: License information (MIT).  


## Portfolio Optimization Task

The notebook includes a worked example on portfolio optimization, where PSO is applied to asset allocation under constraints.  
To evaluate its effectiveness, the performance of PSO is compared against:  

- **SciPy’s standard optimization routines**  
- **A Genetic Algorithm (GA) implementation**  

This highlights the strengths and weaknesses of PSO relative to more traditional and evolutionary optimization approaches.

## 🛠️ Usage

To use the algorithm you can just specify a new objective function as following:

```python

from pso import PSO

def objective_function(x):

# Define your function here

return ...


pso = PSO(obj_func=objective_function, dim=2, num_particles=30, max_iter=100)

best_position, best_value = pso.optimize()
```


## License 
This project is licensed under the MIT License.
Copyright © 2025 Leonardo di Gennaro.

