# Learning to Branch in Mixed Integer Programming - Implementation

This is an implementation of the "Learning to Branch in Mixed Integer Programming" paper by Elias B. Khalil. The implementation uses a machine learning approach to improve the performance of branch-and-bound algorithms in mixed integer programming.

## Usage

To use the implementation, run the `testing_pipeline.py` script with the following command-line arguments:

```console
python3 testing_pipeline.py
```

The implementation will generate 1000 random MIP problems and run the SB, SB+ML and PC algorithms on them. The Number of nodes and the time taken for each run is stored and printed.

## Files

- `SB_ML.py`: the main script that executes the branch-and-bound algorithm with the SB+ML and PC strategies.
- `SB.py`: the main script that executes the branch-and-bound algorithm with the SB strategy.
- `dynamic_features.py`: this script contains the code to generate the dynamic features for a particular node in the tree
- `static_features.py`: this script contains the code to generate the static features for a particular node in the tree
- `dataset_generation.py`: this script generates the random MIP problems by selecting random coefficients for the objective and constraints. We also randomly add integer constraints for some of the variables. The number of variables and constraints for each problem is also randomized.
- `testing_pipeline.py`: this script contains the code to generate and run the dataset on the SB, SB+ML and PC strategies.
- `svm_rank_classify`: this is the SVM rank executable used to run a trained model and get predictions.
- `svm_rank_learn`: this is the SVM rank executable used to train a model on a given set of features.

## References

- Khalil, E. B. (2016). Learning to Branch in Mixed Integer Programming. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16) (pp. 1280-1286).

- Mixed integer programming: Analyzing 12 years of progress. In J¨unger, M., and Reinelt, G., eds., Facets of Combinatorial Optimization. Springer 2013.\\

- Learning to rank for information retrieval(Foundations and Trends in Information Retrieval 2009 3(3):225–331) - Liu, T.-Y.\\

- A supervised machine learning approach to variable branching in branch-and-bound. (Technical Report, Universit´e de Li`ege 2014) - Alvarez, A. M.; Louveaux, Q.; and Wehenkel, L..
