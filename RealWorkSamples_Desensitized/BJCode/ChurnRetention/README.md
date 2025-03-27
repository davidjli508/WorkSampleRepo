# This is the Data Science Repository for Brighton Jones Use Cases
Questions?
Contact: David - david.li@brightonjones.com 

## Overview
The repo contains a dedicated Constants/Helper Functions file *MLConstants.py*, located under */MLConstants.py*

The repo contains 3 main types of files used in a Use Case Pipeline. 

1. Large Use Case Files (i.e. *ChurnUseCase.py*) located under */UseCase/__.py*
2. (Snowflake) Data Retrieval/Preliminary Transformations (i.e. *ChurnDT.py*) located under */Data/__.py*
3. Build/Train Modeling Files (i.e. *LogisticRegression.py*) located under */Modelling/__.py*

Also, the repo contains other 2 other types of files

1. Exploratory Data Analysis Files (i.e. *ChurnEDA.py*) located under */EDA/__.py*
2. Jupyter Notebook Sandbox files, helpful for development/exploration and usually suffixed with *.ipynb*.


## Execution Process Info
When a use case desires a model to be invoked for reasons such as:

1. New Predictions for New Subset or Cohort of Data
2. Evaluating Model Metrics as desired

, you typically will only run the Large Use Case File, which handles the whole start-to-end modelling process and calls other files as necessary.

The Large Use Case File follows these steps:

**Import Necessary Data Science Packages** -> **Import Constants File** -> **Ingest/Retrieve Data from Source (Snowflake) + Apply Transformations** -> **Build/Train Model off Dataset**

The files are designed to be as modular as possible for this process. For example if you wanted to run a use case but with a different type of model, you would create the Large Use Case File as usual, but with the Build/Train Model File substituted with the new desired Model File. This minimizes code duplication and inefficient repetition, and the necessity to invest time writing a new start-to-end process from scratch.

## Execution Output/Completion Info
Once the Model Building/Training finshes running, the process concludes by writing
1. Predictions
2. Model Parameters 
3. Model Performance Metrics

to separate tables in Snowflake (detailed above). These tables will have associated keys in order to associate a run across all 3 tables. 

Predictions are used as necessary such as for other workstreams or end users.
Model Parameters give insight on how the model was configured (i.e. coefficients, intercepts)
Model Metrics help diagnose and monitor model performance/accuracy and model drift.

## Package Requirements (i.e. What this is Verified to Work With as of 10/18/23)
Pandas: 2.0.3
Numpy: 1.24.3
Scikit-Learn: 1.2.2
Snowflake-Connector-Python: 3.2.0
Snowflake-Sqlalchemy: 1.5.0
Imbalanced-Learn: 0.10.1
Sqlalchemy: 1.4.49