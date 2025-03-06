# capstone-team2-winter-quarter

Source: for LASSO - https://medium.com/@agrawalsam1997/feature-selection-using-lasso-regression-10f49c973f08

To set up the required conda environment,
1. Create a new anaconda environment with >conda create --name `<name>`
2. Run >pip install -r requirements.txt
3. There are different arguments to be used with run.py to see different parts of our project.
4. To see graphs, use >python run.py graph <algorithm>, with <algorithm> being fisherz or chisq
5. To play with our counterfactual computing function, use >python run.py counterfactual <intervention_var>, with the options being "Financial_Stress", "Academic_Pressure", and "Suicidal_Thoughts". It will prompt you to ask for values, the values we used were 5, 5, and 1 respectively.

To run the FCI with KCI or "fast KCI" algorithm:
1. Run >pip install -r requirements.txt
2. Run >python3 FCI-fastKCI.py

**TODO, adjust instructions for Fast KCI + normal KCI
+also add instructions for the counterfactual computing part.
