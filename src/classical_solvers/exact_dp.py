import numpy as np
import os
from collections import defaultdict
def solve_exact_dp(coords:np.ndarray):
    n=len(coords)
    if n==1:
        return 0.0,[0]
    
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
    memo=defaultdict(int)
    parent_pointers=defaultdict(int)
    def tsp_dp(mask:int,node:int)->float:
        if mask==(1<<n)-1:
            return dist_matrix[node][0]
        if (mask,node) in memo:
            return memo[(mask,node)]
        min_dist=float('inf')
        best_next_city=-1
        for v in range(n):
            if not (mask & (1<<v)):
                current_route_cost=dist_matrix[node][v]+tsp_dp(mask|(1<<v),v)
                if current_route_cost<min_dist:
                    min_dist=current_route_cost
                    best_next_city=v
        
        memo[(mask,node)]=min_dist
        parent_pointers[(mask,node)]=best_next_city
        return min_dist
    
    optimal_length=tsp_dp(1,0)
    tour=[]
    current_mask=1
    current_city=0
    
    while len(tour)<n:
        tour.append(current_city)
        next_city=parent_pointers.get((current_mask,current_city))
        if next_city is None:
            break
        
        current_mask|=(1<<next_city)
        current_city=next_city
        
    return float(optimal_length),tour
                    
    
    