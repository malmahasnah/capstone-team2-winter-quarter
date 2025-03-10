# A Causal Approach to Understanding Student Depression

Source: for LASSO - https://medium.com/@agrawalsam1997/feature-selection-using-lasso-regression-10f49c973f08

To set up the required conda environment,
1. Create a new anaconda environment with >conda create --name `<name>`
2. Run >pip install -r requirements.txt
3. There are different arguments to be used with run.py to see different parts of our project.
4. To see graphs, use >python run.py graph `<algorithm>`, with `<algorithm>` being fisherz, chisq, or kci.


To run counterfactual computing
1. Run counterfactual.py file
2. Functions take intervention number
3. Only for sleep_duration only input 1-4 (explained by function)

To run the FCI with KCI or "fast KCI" algorithm:
1. Run >pip install -r requirements.txt
2. Run >python3 fast-kci.py

**important note**: due to randomness, the resulting graphs will look slightly different when running these algorithms. Fast KCI and KCI may take 20+ hours to run.

