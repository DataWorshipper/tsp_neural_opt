import numpy as np
from config.settings import INSTANCE_FILE, SIZES_OF_N, INSTANCES_PER_SIZE, RANDOM_SEED
from src.utils.io_utils import write_jsonl

def generate_all_instances():
    print("Generating shared dataset source-of-truth...")
    np.random.seed(RANDOM_SEED)
    all_instances = []
    instance_counter = 0

    for n in SIZES_OF_N:
        for replicate in range(INSTANCES_PER_SIZE):
            coords = np.random.rand(n, 2).tolist()

            instance_data = {
                "instance_id": f"TSP{n}_{replicate:03d}",
                "n": n,
                "coords": coords
            }

            all_instances.append(instance_data)
            instance_counter += 1

    write_jsonl(INSTANCE_FILE, all_instances, append=False)

    print(
        f"Successfully generated and logged "
        f"{instance_counter} unique instances into {INSTANCE_FILE}!"
    )

if __name__ == "__main__":
    generate_all_instances()