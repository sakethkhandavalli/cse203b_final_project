from dataset_generation import generate_dataset
from SB_ML import ML_algorithm
from SB import SB

# Testing pipeline
dataset = generate_dataset(1000)
for problem in dataset:
    ml_nodes, ml_time = ML_algorithm(problem, False)
    sb_nodes, sb_time = SB(problem, False)
    print(ml_time, sb_time)