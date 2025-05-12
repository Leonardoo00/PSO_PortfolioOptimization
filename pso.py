# =============================================================================
#              PARTICLE SWARM OPTIMIZATION (PSO) FROM SCRATCH
# =============================================================================
# Author: Leonardo di Gennaro
# GitHub: https://github.com/Leonardoo00
# LinkedIn: https://www.linkedin.com/in/leonardo-mario-di-gennaro-57827522a/
#
# Description:
# This file contains a full implementation of the Particle Swarm Optimization
# algorithm in Python. The class `PSO` provides an interface for solving
# optimization problems using the PSO metaheuristic.
# =============================================================================
import numpy as np

class PSO(): 
    """
    Particle Swarm Optimization (PSO) algorithm.

    A population-based optimization algorithm inspired by social behaviors in nature.
    This class provides methods for initializing particles, updating their positions
    and velocities, and tracking personal and global bests to converge toward an optimal
    solution over time.

    The PSO instance maintains history and supports flexible fitness functions and
    boundary constraints. Particle state is updated iteratively using inertia,
    cognitive, and social components.
    
    Parameters
    ----------
    See `__init__` method for a detailed description of initialization parameters.

    Methods
    -------
    next_step()
        Perform one iteration of the optimization by updating velocities and positions, 
        evaluating fitness, and updating bests.
    
    update_positions()
        Update the position of each particle based on its current velocity, while 
        respecting the defined bounds of the search space.

    update_pbest()
        Update each particle's personal best (`pbest`) if the current position 
        has a better fitness.

    update_gbest(position, fitness)
        Update the global best (`gbest`) position and fitness if a new better fitness 
        is found among the particles.
    """
    
    def __init__(self, particles, velocities, bounds, fitness_function, w=0.8, c_1=1.5, c_2=1.5, max_iter=50): 
        """
        Initialize the Particle Swarm Optimization (PSO) algorithm.

        Parameters
        ----------
        particles : list of dict
            A list of particles, where each particle is a dictionary with the following structure:
                - 'position' : numpy.ndarray
                    The position of the particle in the search space.
                - 'velocity' : numpy.ndarray
                    The velocity of the particle.
        velocities : list of numpy.ndarray
            Initial velocity vectors for each particle. Must match the structure of `particles`.
        bounds : tuple of numpy.ndarray
            A tuple (lower_bounds, upper_bounds), each a 1D array of shape (dimensions,), used to clip particle positions.
        fitness_function : callable
            A function that evaluates the fitness of a given position. It should take a single numpy.ndarray as input.
        w : float, optional
            Inertia weight controlling the influence of previous velocities. Default is 0.8.
        c_1 : float, optional
            Cognitive (personal) acceleration coefficient. Default is 1.5.
        c_2 : float, optional
            Social (global) acceleration coefficient. Default is 1.5.
        max_iter : int, optional
            Maximum number of iterations for the optimization process. Default is 50.

        Notes
        -----
        This constructor initializes internal parameters and evaluates the initial personal best (`pbest`) for each particle
        as well as the global best (`gbest`) based on the provided fitness function.
        It also sets up history tracking for positions and best scores across iterations.
        """

        self.particles = particles                                   # Initialize particles
        self.velocities = velocities                                 # Initialize velocites 
        self.bounds = bounds                                         # Define bounds of the search space (used to clip raw position)
        self.dimensions = len(velocities[1])                         # Dimensionality of the optimization problem
        self.fitness_function = fitness_function                     # Initialize the fitness function

        self.w = w                                                   # Inertia parameter
        self.c_1 = c_1                                               # Cognitive coefficient
        self.c_2 = c_2                                               # Social coefficient

        self.iter = 0
        self.max_iter = max_iter                                     # Maximum number of iterations

        self.gbest = None                                            # Initialize global best
        self.gbest_fitness = -np.inf

        self.pos_history = []                                        # To store particle positions per iteration
        self.gbest_history = []                                      # To store gbest position per iteration
        self.gbest_fit_history = []                                  # To store gbest fit per iteration

        # Initialize particles
        for p in self.particles:                                  
            p['pbest'] = np.copy(p['position'])
            p['pbest_fitness'] = self.fitness_function(p['position'])
            self.update_gbest(p['pbest'], p['pbest_fitness'])


    def next_step(self): 
        """
        Perform a single iteration (step) of the Particle Swarm Optimization algorithm.

        This method updates the velocity and position of each particle based on its personal best (pbest)
        and the global best (gbest). It also evaluates the new fitness of each particle, updates personal bests 
        if improvements are found, and updates the global best if any particle achieves a better fitness.

        The velocity update follows the standard PSO formula:

            v = w * v + c1 * r1 * (pbest - position) + c2 * r2 * (gbest - position)

        where:
            - w  : inertia weight
            - c1 : cognitive coefficient
            - c2 : social coefficient
            - r1, r2 : random values drawn from a uniform distribution in [0, 1]

        After computing the new velocity and position, the position is clipped within the defined bounds.

        Notes
        -----
        This function also logs the history of:
            - current positions (`pos_history`)
            - current global best position (`gbest_history`)
            - current global best fitness (`gbest_fit_history`)
        """

        # Stop the algo in case the max_iter is reached
        if self.iter >= self.max_iter:
            raise StopIteration("Maximum iterations reached")
        
        # Run the algo
        if self.iter > 0:
            self.update_positions()                                           
            self.update_pbest()               
                
        self.pos_history.append([np.copy(p['position']) for p in self.particles])           # Store current positions before updating
        self.gbest_history.append(np.copy(self.gbest))                                      # Store current gbest position before updating
        self.gbest_fit_history.append(np.copy(self.gbest_fitness))                          # Store current gbest fit before updating

        self.iter += 1                                                                      # Update iteration step

    def update_positions(self):
        """
        Update the positions of all particles in the swarm.

        For each particle, its position is updated using its current velocity as follows:

            position = position + velocity

        After updating, each particle's position is clipped to remain within the specified search bounds.

        Notes
        -----
        - This method assumes that the `particles` list contains dictionaries with 'position' and 'velocity' keys.
        - The bounds are defined by `self.bounds`, a tuple of (lower_bounds, upper_bounds), both of which
        are NumPy arrays of shape (dimensions,).
        """

        # Iterate over the swarm
        for p in self.particles: 

            # Generate random vectors from uniform [0,1]
            r1 = np.random.rand(self.dimensions)
            r2 = np.random.rand(self.dimensions)
            
            # Calculate velocity components following Eq.(2)
            cognitive = self.c_1 * r1 * (p['pbest'] - p['position'])
            social = self.c_2 * r2 * (self.gbest - p['position'])
            
            # Update velocity
            p['velocity'] = self.w * p['velocity'] + cognitive + social
            
            # Update position (raw position for algo mechanism)
            p['position'] += p['velocity']

            # Clip position after update
            p['position'] = np.clip(p['position'], self.bounds[:, 0], self.bounds[:, 1])  


    def update_pbest(self):
        """
        Update the personal best (pbest) position and fitness for each particle.

        For each particle in the swarm, the current position is evaluated using the fitness function.
        If the current fitness is better than the particle's personal best fitness (`pbest_fitness`),
        both the `pbest` position and `pbest_fitness` are updated accordingly.

        Notes
        -----
        - This method assumes that each particle is a dictionary containing:'position', 'pbest', 'pbest_fitness'.
        - The fitness function is provided via `self.fitness_function` and should accept a NumPy array as input.
        """

        # Iterate over the swarm
        for p in self.particles:
            current_fitness = self.fitness_function(p['position'])          # Calculate current fitness
            if current_fitness > p['pbest_fitness']:                        # Check if new pbest is reached
                p['pbest'] = np.copy(p['position'])                         # Update pbest position
                p['pbest_fitness'] = current_fitness                        # Update pbest fitness
                self.update_gbest(p['pbest'], p['pbest_fitness'])           # Check if new gbest is reached


    def update_gbest(self, position, fitness):
        """
        Update the global best (gbest) position and fitness if the candidate is better.

        Parameters
        ----------
        position : numpy.ndarray
            The position vector of the candidate solution to be evaluated as a potential new global best.
        fitness : float
            The fitness value associated with the candidate position.

        Notes
        -----
        If the candidate fitness is greater than the current global best fitness (`gbest_fitness`),
        this method updates the global best position (`gbest`) and fitness (`gbest_fitness`) accordingly.
        """

        if fitness > self.gbest_fitness:                                    # Check if better fitness is reached
            self.gbest = np.copy(position)                                  # Update gbest position
            self.gbest_fitness = fitness                                    # Update gbest fitness

# =============================================================================
#                               TESTING SECTION
# =============================================================================
# The following code demonstrates how to use the PSO class defined above.
# It includes a simple test scenario with a sample fitness function.
# Modify this section to test the PSO algorithm on custom problems.
# =============================================================================

# Vector-compatible Ackley function
def ackley_pso(position):
    x, y = position[0], position[1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    return term1 + term2 + 20 + np.e

# PSO fitness (maximization)
def pso_fitness(position):
    return -ackley_pso(position)  

# Initialize particles
n_particles = 30
dimensions = 2
bounds = np.array([[-5, 5], [-5, 5]]) 

particles = [{
    'position': np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=dimensions),
    'velocity': np.zeros(dimensions)
} for _ in range(n_particles)]

# Initialize velocities (optional, if not part of particles)
velocities = np.zeros((n_particles, dimensions))

# Run PSO
pso = PSO(particles, velocities, bounds, pso_fitness, w=0.7, c_1=1.5, c_2=1.5, max_iter=50)

# Optimization loop
while pso.iter < pso.max_iter:
    pso.next_step()

    # Print results for each iteration
    print(f"Iter n.{pso.iter} | Global Best {pso.gbest_fitness}")
    
    # Check algo final result
    if pso.iter == pso.max_iter:
        print('\n')
        print(f"The final result is: {-pso.gbest_fitness:4f}")