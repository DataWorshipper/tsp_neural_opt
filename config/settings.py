import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "instances")
INSTANCE_FILE = os.path.join(DATA_DIR, "instances.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "stage1_classical")
DP_RESULTS_FILE = os.path.join(RESULTS_DIR, "dp_results.jsonl")
HEURISTIC_RESULTS_FILE = os.path.join(RESULTS_DIR, "heuristic_results.jsonl")
RANDOM_SEED = 42
SIZES_OF_N = [5, 8, 10, 12, 15, 20, 50, 100]  
INSTANCES_PER_SIZE = 10  

if __name__=="__main__":
    print(BASE_DIR)
    print(DATA_DIR)
    print(INSTANCE_FILE)
    print(RESULTS_DIR)
    print(DP_RESULTS_FILE)
    print(HEURISTIC_RESULTS_FILE)
    
    

