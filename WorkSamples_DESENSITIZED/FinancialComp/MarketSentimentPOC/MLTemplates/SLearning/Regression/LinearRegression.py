## Code for Linear Regression Modeling

# Code assumes the input dataframe has transformations completed
# If needs to be done on Python side, make new code

## Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import datetime 
from sqlalchemy import create_engine
from snowflake.connector.pandas_tools import pd_writer


## LinearRegressionParams() takes a built Linear Regression model and captures the model paramters
# lin_reg_model is a Linear Regression Model object, assumedly from BuildLinearRegression()
# Returns: List of Linear Regression Model Parameters
def LinearRegressionParams(lin_reg_model):
    # Obtain Feature names in Order
    feature_names = lin_reg_model.feature_names_in_
    # Obtain Linear Regression Model Coefficients in Order
    model_coefficients = lin_reg_model.coef_
    # Obtain Linear Regression Model Intercept
    model_intercept = lin_reg_model.intercept_

    # Consolidate into Pandas Dataframe
    param_data = np.append(model_coefficients, model_intercept)
    param_cols = np.append(feature_names, "model_intercept")
    
    param_df = pd.DataFrame(columns = param_cols)
    param_df.loc[len(param_df)] = param_data.tolist()

    return param_df

## LinearRegressionMetrics() takes a built Linear Regression model and captures the model paramters
# lin_reg_model is a Linear Regression Model object, assumedly from BuildLinearRegression()
# Returns: List of Linear Regression Model Metrics
def LinearRegressionMetrics(lin_reg_model, X_test, y_test):

    # Generate Predictions to validate with
    y_pred = lin_reg_model.predict(X_test)

    # Obtain Linear Regression Metrics
    explained_variance = round(sk.metrics.explained_variance_score(y_test, y_pred),4)
    mean_absolute_error = round(sk.metrics.mean_absolute_error(y_test, y_pred), 4) 
    mse = round(sk.metrics.mean_squared_error(y_test, y_pred), 4)
    median_absolute_error = round(sk.metrics.median_absolute_error(y_test, y_pred), 4)
    r2 = round(sk.metrics.r2_score(y_test, y_pred), 4)

    # Consolidate into Pandas Dataframe
    metrics_data = [explained_variance, mean_absolute_error, mse, median_absolute_error, r2]
    metrics_cols = ["explained_variance", "mean_absolute_error", "mean_squared_error", "median_absolute_error", "r2_score"]

    metrics_df = pd.DataFrame(columns = metrics_cols)
    metrics_df.loc[len(metrics_df)] = metrics_data

    return metrics_df

## BuildLinearRegression() constructs a linear regression model
# learning_dataset is a Pandas dataframe, to build model off of
# target_var is the variable to predict on (needs to be in quotes)
def BuildLinearRegression(learning_dataset, target_var):

    # Build Linear Regression Object
    lin_reg_model = sk.linear_model.LinearRegression()
    
    # Define predictors and target variable
    X_DF = learning_dataset.loc[:, learning_dataset.columns != target_var]
    Y_DF = learning_dataset[target_var]

    # Split into Train and Test Sets
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X_DF, Y_DF, test_size=0.2, random_state=42)
    
    # Start Stopwatch for training time
    training_start_time = datetime.datetime.now()
    # Train the model using the training sets
    lin_reg_model.fit(X_train, y_train)
    # End Stopwatch for training time
    training_end_time = datetime.datetime.now()

    # Calculate training time duration in seconds
    training_time_sec = (training_end_time - training_start_time).total_seconds()

    # Capture the fitted parameters in Pandas dataframe table
    lin_reg_model_params = LinearRegressionParams(lin_reg_model)
    lin_reg_model_params["training_start_time"] = training_start_time
    lin_reg_model_params["training_end_time"] = training_end_time
    lin_reg_model_params["training_time_sec"] = training_time_sec

    # Capture the model performance metrics in Pandas dataframe table
    lin_reg_model_metrics = LinearRegressionMetrics(lin_reg_model, X_test, y_test)
    lin_reg_model_metrics["training_start_time"] = training_start_time
    lin_reg_model_metrics["training_end_time"] = training_end_time
    lin_reg_model_metrics["training_time_sec"] = training_time_sec

    # Returns: a Linear Regression Object, a Pandas Dataframe of Fitted Model Parameters, a Pandas Dataframe of Model Metrics
    # optionally: corresponding training and test sets
    return lin_reg_model, lin_reg_model_params, lin_reg_model_metrics #, X_train, X_test, y_train, y_test

## PredictLinearRegression() uses a Linear Regression model and predicts with input dataset
# lin_reg_model is a Linear Regression Model object, assumedly from BuildLinearRegression()
# prediction_dataset is the Pandas dataframe, to generate predictions on. Assumes prediction column is excluded here
# target_var is the variable to predict on (needs to be in quotes)
def PredictLinearRegression(lin_reg_model, predict_on_dataset, target_var):

    # Use Trained Linear Regression Object to predict
    y_predictions = lin_reg_model.predict(predict_on_dataset)

    # Create name for prediction column
    pred_variable_name = target_var + "_Pred"

    # Append predictions to dataset to predict on
    prediction_dataset = predict_on_dataset.copy(deep = True)
    prediction_dataset[pred_variable_name] = y_predictions

    # Returns: a Pandas dataframe, containing the prediction_dataset along with appended prediction values
    return prediction_dataset

## RunLinearRegression executes all related Linear Regression functions, and push prediction output, Model Parameters/Metrics and Run Details to corresponding tables in Snowflake DB
def RunLinearRegression(learning_dataset, target_var, pred_on_dataset, pred_table_name, param_table_name, metrics_table_name):
    # Get Run Timestamp
    run_timestamp = datetime.datetime.now()

    # Run Core Linear Regression Functions
    LinRegRun = BuildLinearRegression(learning_dataset, target_var)
    LinRegPredOutput = PredictLinearRegression(LinRegRun[0], pred_on_dataset, target_var)
    LinRegPredOutput.columns = map(lambda x: str(x).upper(), LinRegPredOutput.columns)

    # Stage tables and Append Run Time to Link tables
    StagingParamsDF = LinRegRun[1]
    StagingParamsDF["run_timestamp"] = run_timestamp
    StagingParamsDF.columns = map(lambda x: str(x).upper(), StagingParamsDF.columns)
    StagingMetricsDF = LinRegRun[2]
    StagingMetricsDF["run_timestamp"] = run_timestamp
    StagingMetricsDF.columns = map(lambda x: str(x).upper(), StagingMetricsDF.columns)

    # Drop Indexes for all 3 Tables
    LinRegPredOutput.to_csv("linreg.csv")
    StagingParamsDF.to_csv("stagingparams.csv")
    StagingMetricsDF.to_csv("stagingmetrics.csv")

    #Create connection to Snowflake using your account and user
    account_identifier = 'DESENSITIZED'
    user = 'DESENSITIZED'
    password = 'DESENSITIZED'
    database_name = 'DESENSITIZED'
    schema_name = 'MODEL_OUTPUT'

    conn_string = f"snowflake://{user}:{password}@{account_identifier}/{database_name}/{schema_name}"
    engine = create_engine(conn_string)
    engine2 = create_engine(conn_string)
    engine3 = create_engine(conn_string)

    with engine.connect() as conn:
        LinRegPredOutput.to_sql(name = pred_table_name, con=engine, schema = "MODEL_OUTPUT", if_exists="append", method = pd_writer, index = False)
    
    with engine2.connect() as conn2:
        StagingParamsDF.to_sql(name = param_table_name, con = engine2, schema = "MODEL_OUTPUT", if_exists = "append", method = pd_writer, index = False)

    with engine3.connect() as conn3:
        StagingMetricsDF.to_sql(name = metrics_table_name, con = engine3, schema = "MODEL_OUTPUT", if_exists = "append", method = pd_writer, index = False)
