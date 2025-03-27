## Modelling: Logistic Regression (.py)

# Import Code & Helper Modules
import sys
sys.path.append(r"C:\\Users\\david.li\\Desktop\\WorkCode\\data-science-dump")

# Constants
import pandas as pd
import uuid

# Data Transformations
import numpy as np
import snowflake.connector as sconn
import datetime

# Modelling
import sklearn as sk
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from snowflake.connector.pandas_tools import pd_writer
from sqlalchemy.dialects import registry

## Import Constants File
# 1. Bring in Code Constants to Use
import MLConstants as mlc

## LogisticRegressionDetails() takes a built Logistic Regression model and captures the model details (parameters)
# log_reg_model is a Logistic Regression Model object, assumedly from BuildLogisticRegression()
# Returns: Pandas Dataframe of the Logistic Regression Model Parameters
def LogisticRegressionDetails(log_reg_model):
    # Obtain Feature names in Order
    feature_names = (log_reg_model.feature_names_in_).tolist()
    # Obtain Logistic Regression Model Coefficients in Order
    model_coefficients = (log_reg_model.coef_).tolist()
    # Obtain Logistic Regression Model Intercept
    model_intercept = (log_reg_model.intercept_).tolist()

    # Consolidate into Lists
    detail_data = model_coefficients + model_intercept
    detail_cols = feature_names + ["model_intercept"]

    return detail_data, detail_cols

## LogisticRegressionMetrics() takes a built Logistic Regression model and captures the model classification metrics
# log_reg_model is a Logistic Regression Model object, assumedly from BuildLogisticRegression()
# X/y train/test inputs are Pandas Dataframes of mentioned parts of the learning dataset
# Returns: Pandas Dataframe of the Logistic Regression Model's Metrics
def LogisticRegressionMetrics(log_reg_model, X_train, y_train, X_test, y_test):

    # Generate Various Predictions and Probabilities on Training and Test Sets
    y_train_pred = log_reg_model.predict(X_train)
    y_test_pred = log_reg_model.predict(X_test)

    # Obtain Logistic Regression Model Metrics (which are Classification metrics)
    # In Order: Balanced Accuracy, Precision, Recall, F1 Score, Area under ROC/AUC Curve
    # https://scikit-learn.org/stable/modules/classes.html#classification-metrics

    # Training
    training_acc = round(sk.metrics.balanced_accuracy_score(y_train, y_train_pred), 4)
    training_precision = round(sk.metrics.precision_score(y_train, y_train_pred), 4)
    training_recall = round(sk.metrics.recall_score(y_train, y_train_pred), 4)
    training_f1score = round(sk.metrics.f1_score(y_train, y_train_pred), 4)
    training_rocauc = round(sk.metrics.roc_auc_score(y_train, y_train_pred), 4)
    # Test
    test_acc = round(sk.metrics.balanced_accuracy_score(y_test, y_test_pred), 4)
    test_precision = round(sk.metrics.precision_score(y_test, y_test_pred), 4)
    test_recall = round(sk.metrics.recall_score(y_test, y_test_pred), 4)
    test_f1score = round(sk.metrics.f1_score(y_test, y_test_pred), 4)
    test_rocauc = round(sk.metrics.roc_auc_score(y_test, y_test_pred), 4)

    # Consolidate into Pandas Dataframe
    metrics_data = [training_acc, training_precision, training_recall, training_f1score, training_rocauc,
                    test_acc, test_precision, test_recall, test_f1score, test_rocauc]
    metrics_cols = ["training_acc", "training_precision", "training_recall", "training_f1score", "training_rocauc",
                    "test_acc", "test_precision", "test_recall", "test_f1score", "test_rocauc"]
    metrics_df = pd.DataFrame(columns = metrics_cols)
    metrics_df.loc[len(metrics_df)] = metrics_data

    return metrics_df

## BuildLogisticRegression() constructs a logistic regression model
# input_df is a Pandas dataframe, to build model off of
# id_var is the name of the id column in the dataframe (in string form)
# target_var is the variable to predict on (in string form)
# preprocess_list supplies what pre-processing needs to be done
def BuildLogisticRegression(input_df, id_var, target_var, preprocess_list, version_num):
    # Identify Data Chunks / Predictor and Target Sets
    X_DF = input_df.drop([target_var, id_var], axis = 1)
    y_DF = input_df[target_var]

    # Apply One-Hot-Encode to Categorical Variables, necessary for logistic regression
    # Get Columns that are Categorical, and get columns thats are numerical
    num_cols = [col for col in X_DF.columns if X_DF[col].dtype != "object"]
    cat_cols = [col for col in X_DF.columns if X_DF[col].dtype == "object"]
    # If there are categorical columns, then proceed
    if len(cat_cols) > 0:
        # Designate Columns to be one-hot-encoded, if their cardinality is reasonable (<= 5)
        low_cardinality_cols = [col for col in cat_cols if X_DF[col].nunique() <= 5]
        # Otherwise, drop higher cardinality categorical columns (> 5)
        high_cardinality_cols = list(set(cat_cols)-set(low_cardinality_cols))
        X_DF.drop(high_cardinality_cols, axis = 1, inplace = True)
        # Apply One-Hot-Encoder, Replace the learning dataframe with one-hot-encoded dataframe
        oh = sk.preprocessing.OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        one_hot_encoded = oh.fit_transform(X_DF[low_cardinality_cols])
        X_DF = pd.concat([X_DF, one_hot_encoded], axis=1).drop(columns = low_cardinality_cols)

    # Interpret Preprocess Tasks Requested
    # Known Items are SMOTE, Normalize
    preprocess_upper_tasks = [x.upper() for x in preprocess_list]

    # Prepare a dictionary to hold Preprocesser Engines if they were used
    engine_dict = {}

    # Apply Normalize to Numeric Variables, if requested
    if mlc.NORMALIZE_TASK in preprocess_upper_tasks:
        # Create Normalizer Engine
        normalizer = sk.preprocessing.MinMaxScaler()
        # Apply Normalizer to Training Set Numeric Variables
        X_DF[num_cols] = normalizer.fit_transform(X_DF[num_cols])
        # Add to list of engines used
        engine_dict[mlc.NORMALIZE_TASK] = normalizer

    # Split into Train and Test Sets
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X_DF, y_DF, test_size=0.2, random_state=42)

    # Apply SMOTE, if requested
    if mlc.SMOTE_TASK in preprocess_upper_tasks:
        # Create Smote Object
        sm = SMOTE(random_state = 42)
        # Apply SMOTE to Training Set
        X_train, y_train = sm.fit_resample(X_train, y_train)
        # Add to list of engines used
        engine_dict[mlc.SMOTE_TASK] = sm

    # Start Stopwatch for training time
    training_start_time = datetime.datetime.now()

    # Based on Input Version Num, Retrieve Hyperparameters from Snowflake
    paramTable = mlc.getSnowflakeData([mlc.tableToQuery(mlc.LOGREG_PARAM_TBL)], mlc.CRED_DS_DICT)[0]
    penalty_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'PENALTY'].item()
    dual_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'DUAL'].item()
    tol_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'TOL'].item()
    c_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'C'].item()
    fit_intercept_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'FIT_INTERCEPT'].item()
    intercept_scaling_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'INTERCEPT_SCALING'].item()
    class_weight_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'CLASS_WEIGHT'].item()
    random_state_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'RANDOM_STATE'].item()
    solver_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'SOLVER'].item()
    max_iter_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'MAX_ITER'].item()
    multi_class_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'MULTI_CLASS'].item()
    verbose_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'VERBOSE'].item()
    warm_start_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'WARM_START'].item()
    n_jobs_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'N_JOBS'].item()
    l1_ratio_param = paramTable.loc[paramTable['VERSION_NUM'] == version_num, 'L1_RATIO'].item()

    # Train the Logistic Regression Model
    # Use hyperparameters that correspond to the version number, in hyperparamter table in snowflake
    log_reg_model = sk.linear_model.LogisticRegression(penalty=penalty_param, dual=dual_param, tol=tol_param, 
                                                       C=c_param, fit_intercept=fit_intercept_param, intercept_scaling=intercept_scaling_param, 
                                                       class_weight=class_weight_param, random_state=random_state_param, solver=solver_param, 
                                                       max_iter=max_iter_param, multi_class=multi_class_param, verbose=verbose_param, 
                                                       warm_start=warm_start_param, n_jobs=n_jobs_param, l1_ratio=l1_ratio_param).fit(X_train, y_train)

    # End Stopwatch for training time
    training_end_time = datetime.datetime.now()

    # Capture the fitted parameters in Pandas dataframe table
    # Function to convert a full list to string representation
    def listToString(s):
        # initialize an empty string
        sep = ","
        # return string
        return (sep.join(map(str, s)))
    log_reg_model_details_data = listToString(LogisticRegressionDetails(log_reg_model)[0])
    log_reg_model_details_cols = listToString(LogisticRegressionDetails(log_reg_model)[1])
    details_data = {'model_features': log_reg_model_details_cols, 'model_coefficients': log_reg_model_details_data}
    details_data = {k:[v] for k,v in details_data.items()}  # WORKAROUND
    log_reg_model_details_df = pd.DataFrame(data = details_data)

    # Add Additional Info
    # Calculate training time duration in seconds
    training_time_sec = (training_end_time - training_start_time).total_seconds()
    log_reg_model_details_df["training_time_sec"] = training_time_sec
    # Capture Pre-processing Tasks
    if mlc.SMOTE_TASK in engine_dict:
        log_reg_model_details_df["smote_applied"] = "Yes"
    else:
        log_reg_model_details_df["smote_applied"] = "No"
    if mlc.NORMALIZE_TASK in engine_dict:
        log_reg_model_details_df["normalize_applied"] ="Yes"
    else:
        log_reg_model_details_df["normalize_applied"] = "No"
    
    # Capture the model performance metrics in Pandas dataframe table
    log_reg_model_metrics = LogisticRegressionMetrics(log_reg_model, X_train, y_train, X_test, y_test)

    # Returns: a Logistic Regression Object, a Pandas Dataframe of Fitted Model Parameters, a Pandas Dataframe of Model Metrics
    # optionally: corresponding training and test sets
    return log_reg_model, log_reg_model_details_df, log_reg_model_metrics, engine_dict, X_train, y_train, X_test, y_test

## PredictLogisticRegression() uses a Logistic Regression model and predicts with input dataset
# log_reg_model is a Logistic Regression Model object, assumedly from BuildLogisticRegression()
# prediction_dataset is the Pandas dataframe, to generate predictions on. Handles if ID and Target Variable Columns are present
# id_var is the id column in the dataset (in string form)
# target_var is the variable to predict on (in string form)
def PredictLogisticRegression(log_reg_model, engine_dict, predict_on_dataset, id_var, target_var):
    # Get 2 Copy Forms of Dataset to Work wtih, 1 for running the model on and one for visual output of dataset original representation
    pred_df = predict_on_dataset.copy(deep = True)
    output_df = predict_on_dataset.copy(deep = True)

    # Predict_on_dataset should only contain columns that log_reg_model is training on (i.e. no id_var, target_var) 
    # Remove these if present during prediction, but save them and append back later
    if id_var in pred_df.columns:
        pred_df.drop(id_var, inplace = True, axis = 1)
    if target_var in pred_df.columns:
        pred_df.drop(target_var, inplace = True, axis = 1)

    # Re-form Dataset to Predict on, as required by logistic regression (i.e. One Hot Encoding)
    # Apply One-Hot-Encode to Categorical Variables, necessary for logistic regression
    # Get Columns that are Categorical, and get columns thats are numerical
    num_cols = [col for col in pred_df.columns if pred_df[col].dtype != "object"]
    cat_cols = [col for col in pred_df.columns if pred_df[col].dtype == "object"]
    # If there are categorical columns, then proceed
    if len(cat_cols) > 0:
        # Designate Columns to be one-hot-encoded, if their cardinality is reasonable (<= 5)
        low_cardinality_cols = [col for col in cat_cols if pred_df[col].nunique() <= 5]
        # Otherwise, drop higher cardinality categorical columns (> 5)
        high_cardinality_cols = list(set(cat_cols)-set(low_cardinality_cols))
        pred_df.drop(high_cardinality_cols, axis = 1, inplace = True)
        # Apply One-Hot-Encoder, Replace the learning dataframe with one-hot-encoded dataframe
        oh = sk.preprocessing.OneHotEncoder(sparse_output = False).set_output(transform="pandas")
        one_hot_encoded = oh.fit_transform(pred_df[low_cardinality_cols])
        pred_df = pd.concat([pred_df, one_hot_encoded], axis=1).drop(columns = low_cardinality_cols)

    # Normalize In Same Way as During Model Building, if it was a pre-processing step done
    if mlc.NORMALIZE_TASK in engine_dict:
        # Apply Normalizer to Training Set Numeric Variables
        pred_df[num_cols] = engine_dict[mlc.NORMALIZE_TASK].fit_transform(pred_df[num_cols])

    # Use Trained Logistic Regression Object to predict labels and probabilities
    y_predictions = log_reg_model.predict(pred_df)
    y_prediction_probs = log_reg_model.predict_proba(pred_df)

    # Create name for prediction column
    pred_variable_name = target_var + mlc.PRED_SUFFIX 
    proba_variable_name = mlc.PROBA_PREFIX + target_var + mlc.PRED_SUFFIX

    # Append predictions to dataset to predict on
    output_df[pred_variable_name] = y_predictions
    output_df[proba_variable_name] = y_prediction_probs[:, 1]
        
    # Returns: a Pandas dataframe, containing the prediction_dataset along with appended prediction values
    return output_df

## RunLogisticRegression executes all related Logistic Regression functions, and push prediction output, Model Parameters/Metrics and Run Details to corresponding tables in Snowflake DB
def RunLogisticRegression(learning_df, preprocess_list, version_num, id_var, target_var, pred_df, use_case_desc, 
                          pred_table_name, details_table_name, metrics_table_name, write_snowflake, display_output):
    # Get Run Timestamp
    run_timestamp = datetime.datetime.now(tz = datetime.timezone.utc)
    string_run_timestamp = run_timestamp.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Generate a RunID
    run_ID = str(uuid.uuid4())

    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Beginning RunLogisticRegression() with Run Timestamp: " + string_run_timestamp + " and Run ID as: " + str(run_ID))
        print("")

    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Running BuildLogisticRegression() at: " + string_run_timestamp)
    # Run Core Logistic Regression Functions
    LogRegRun = BuildLogisticRegression(learning_df, id_var, target_var, preprocess_list, version_num)

    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Finished BuildLogisticRegression() at: " + string_run_timestamp)
        print("")

    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Running PredictLogisticRegression() at: " + string_run_timestamp)
    # Generate Full Model Predictions Table
    FullLogRegPredOutput = PredictLogisticRegression(LogRegRun[0], LogRegRun[3], pred_df, id_var, target_var)
    # Keep Only ID and Output Columns
    LogRegPredOutput = FullLogRegPredOutput[[id_var, target_var, target_var + mlc.PRED_SUFFIX, mlc.PROBA_PREFIX + target_var + mlc.PRED_SUFFIX]]
    # Append Run Info
    LogRegPredOutput["run_timestamp"] = string_run_timestamp
    LogRegPredOutput["run_id"] = run_ID
    LogRegPredOutput["model_name"] = mlc.LOG_MODEL_NAME
    LogRegPredOutput["model_version_num"] = version_num
    # Assert Column Names to All Caps
    LogRegPredOutput.columns = map(lambda x: str(x).upper(), LogRegPredOutput.columns)
    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Finished PredictLogisticRegression() at: " + string_run_timestamp)
        print("")

    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Building Model Details Table at: " + string_run_timestamp)
    # Generate Model Parameters Table
    StagingModelDetailsDF = LogRegRun[1]
    # Append Run Info
    StagingModelDetailsDF["run_timestamp"] = string_run_timestamp
    StagingModelDetailsDF["run_id"] = run_ID
    StagingModelDetailsDF["use_case"] = use_case_desc
    StagingModelDetailsDF["model_version_num"] = version_num
    # Assert Column Names to All Caps
    StagingModelDetailsDF.columns = map(lambda x: str(x).upper(), StagingModelDetailsDF.columns)
    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Model Details Table Built at: " + string_run_timestamp)
        print("")

    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Building Metrics Table at: " + string_run_timestamp)
    # Generate Model Metrics Table
    StagingMetricsDF = LogRegRun[2]
    # Append Run Info
    StagingMetricsDF["run_timestamp"] = string_run_timestamp
    StagingMetricsDF["run_id"] = run_ID
    StagingMetricsDF["use_case"] = use_case_desc
    StagingMetricsDF["model_version_num"] = version_num
    # Assert Column Names to All Caps
    StagingMetricsDF.columns = map(lambda x: str(x).upper(), StagingMetricsDF.columns)
    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        print("Metrics Table Built at: " + string_run_timestamp)
        print("")

    # If we choose to want to write output to Snowflake
    if write_snowflake.upper() in ["Y", "YES"]:
        # Print Status of Execution
        if display_output.upper() in ["Y", "YES"]:
            print("Writing Output to Snowflake")
        #Create connection to Snowflake using your account and user
        account_identifier = 'DESENSITIZED'
        user = 'DESENSITIZED'
        password = 'DESENSITIZED'
        database_name = 'DESENSITIZED'
        schema_name = 'DESENSITIZED'

        conn_string = f"snowflake://{user}:{password}@{account_identifier}/{database_name}/{schema_name}"
        registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')
        engine = create_engine(conn_string)
        engine2 = create_engine(conn_string)
        engine3 = create_engine(conn_string)

        with engine.connect() as conn:
            LogRegPredOutput.to_sql(name = pred_table_name, con=engine, schema = mlc.MODEL_OUTPUT_SCHEMA, if_exists="append", method = pd_writer, index = False)
        
        with engine2.connect() as conn2:
            StagingModelDetailsDF.to_sql(name = details_table_name, con = engine2, schema = mlc.MODEL_DETAILS_SCHEMA, if_exists = "append", method = pd_writer, index = False)

        with engine3.connect() as conn3:
            StagingMetricsDF.to_sql(name = metrics_table_name, con = engine3, schema = mlc.MODEL_METRICS_SCHEMA, if_exists = "append", method = pd_writer, index = False)

        # Print Status of Execution
        if display_output.upper() in ["Y", "YES"]:
            print("Output Written to Snowflake")
            print("")
    # Print Status of Execution
    if display_output.upper() in ["Y", "YES"]:
        time_now = datetime.datetime.now(tz = datetime.timezone.utc)
        total_timesec_spent = (time_now - run_timestamp).total_seconds()
        print("RunLogisticRegression() Complete. Time Taken (in Seconds): " + str(round(float(total_timesec_spent), 2)))
        print("")

    # Alias the training and testing sets for output access
    X_train_output = LogRegRun[4]
    y_train_output = LogRegRun[5]
    X_test_output = LogRegRun[6]
    y_test_output = LogRegRun[7]

    return LogRegRun, FullLogRegPredOutput, LogRegPredOutput, StagingModelDetailsDF, StagingMetricsDF, X_train_output, y_train_output, X_test_output, y_test_output