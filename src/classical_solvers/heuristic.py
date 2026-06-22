import numpy as np
import time
from src.utils.tour_utils import calculate_tour_length

def solve_nearest_neighbor(coords:np.ndarray,start_city:int=0)->tuple[list,float]:
    n=len(coords)
    if n==1:
        return [0] ,0.0
    dist_matrix = np.linalg.norm(
        coords[:, np.newaxis] - coords[np.newaxis, :], axis=2
    )
    visited=[False]*n
    tour=[]
    current_city=start_city
    visited[current_city]=True
    tour.append(current_city)
    
    for _ in range(n-1):
        nearest_city=-1
        nearest_dist=float("inf")
        for city in range(n):
            if not visited[city] and dist_matrix[current_city][city]<nearest_dist:
                nearest_dist=dist_matrix[current_city][city]
                nearest_city=city
        
        visited[nearest_city]=True
        tour.append(nearest_city)
        current_city=nearest_city
    
    length=calculate_tour_length(coords,tour)
    return tour,length

def solve_two_opt(coords:np.ndarray,initial_tour:list)->tuple[list,float]:
    n = len(initial_tour)
    if n <= 3:
        return initial_tour, calculate_tour_length(coords, initial_tour)
    
    dist_matrix = np.linalg.norm(
        coords[:, np.newaxis] - coords[np.newaxis, :], axis=2
    )
    tour=initial_tour[:]
    improved=True
    while improved:
        improved=False
        for i in range(n-1):
            for j in range(i+2,n):
                if i==0 and j==n-1:
                    continue
                
                city_i=tour[i]
                city_i1=tour[i+1]
                city_j=tour[j]
                city_j1=tour[(j+1)%n]
                current_cost=dist_matrix[city_i][city_i1]+dist_matrix[city_j][city_j1]
                new_cost = dist_matrix[city_i][city_j] + dist_matrix[city_i1][city_j1]
                if new_cost<current_cost-1e-10:
                    tour[i+1:j+1]=tour[i+1:j+1][::-1]
                    improved=True
                    
    length=calculate_tour_length(coords,tour)
    return tour,length

 
def solve_nn_two_opt(coords: np.ndarray, start_city: int = 0) -> tuple[list, float, float, float]:
    
    start = time.perf_counter()
 
    nn_tour, nn_length =solve_nearest_neighbor(coords, start_city)
    final_tour, final_length = solve_two_opt(coords, nn_tour)
    total_time = time.perf_counter() - start
 
    return final_tour, nn_length, final_length, total_time
                    
