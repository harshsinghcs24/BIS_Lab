import numpy as np
import math

def calculate_beam_weight(height, width, thickness):
    length = 10.0      
    density = 7850     
    cross_section_area = (2 * width * thickness) + ((height - 2 * thickness) * thickness)
    return cross_section_area * length * density

def calculate_bending_stress(height, width, thickness):
    central_load = 50000          
    length = 10.0
    moment_of_inertia = (width * (height**3) / 12.0) - ((width - thickness) * ((height - 2 * thickness)**3) / 12.0)
    bending_moment = (central_load * length) / 4.0
    distance_to_edge = height / 2.0  
    return (bending_moment * distance_to_edge) / moment_of_inertia

def objective_function(dimensions):
    height, width, thickness = dimensions
    
    if thickness >= width/2 or thickness >= height/2 or thickness <= 0.005:
        return 1e10  
        
    weight = calculate_beam_weight(height, width, thickness)
    stress = calculate_bending_stress(height, width, thickness)
    max_allowable_stress = 250e6  
    
    if stress > max_allowable_stress:
        penalty = (stress - max_allowable_stress) * 10.0
        return weight + penalty
        
    return weight

def levy_flight(beta=1.5):
    num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    return u / (abs(v) ** (1 / beta))

def cuckoo_search_beam_optimization():
    NUM_NESTS = 25
    DISCARD_RATE = 0.25           
    MAX_ITERATIONS = 100
    
    bounds = np.array([[0.1, 1.0], [0.1, 0.5], [0.005, 0.05]])
    
    nests = np.zeros((NUM_NESTS, 3))
    for i in range(3):
        nests[:, i] = bounds[i, 0] + (bounds[i, 1] - bounds[i, 0]) * np.random.rand(NUM_NESTS)
        
    fitness = np.array([objective_function(nest) for nest in nests])
    
    best_nest = nests[np.argmin(fitness)].copy()
    best_fitness = np.min(fitness)
    
    for iteration in range(MAX_ITERATIONS):
        for i in range(NUM_NESTS):
            tweak = levy_flight() * 0.01  
            new_nest = nests[i] + tweak * (nests[i] - best_nest) * np.random.normal(0, 1, 3)
            new_nest = np.clip(new_nest, bounds[:, 0], bounds[:, 1])
            new_fitness = objective_function(new_nest)
            
            opponent_idx = np.random.randint(0, NUM_NESTS)
            if new_fitness < fitness[opponent_idx]:
                nests[opponent_idx] = new_nest
                fitness[opponent_idx] = new_fitness

        rankings = np.argsort(fitness)
        nests, fitness = nests[rankings], fitness[rankings]
        
        num_to_abandon = int(NUM_NESTS * DISCARD_RATE)
        start_idx = NUM_NESTS - num_to_abandon
        
        for i in range(start_idx, NUM_NESTS):
            for dim in range(3):
                nests[i, dim] = bounds[dim, 0] + (bounds[dim, 1] - bounds[dim, 0]) * np.random.rand()
            fitness[i] = objective_function(nests[i])
            
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_nest = nests[np.argmin(fitness)].copy()
            
    print("--- OPTIMIZATION COMPLETE ---")
    print(f"Optimal Beam Height    : {best_nest[0]:.4f} meters")
    print(f"Optimal Beam Width     : {best_nest[1]:.4f} meters")
    print(f"Optimal Beam Thickness : {best_nest[2]:.4f} meters")
    print(f"Minimized Beam Weight  : {best_fitness:.2f} kg")

cuckoo_search_beam_optimization()
