## Packages in Environment Needed
import pandas as pd
import numpy as np
import snowflake.connector as sconn
from sqlalchemy import create_engine
from snowflake.connector.pandas_tools import pd_writer
from sqlalchemy.dialects import registry

# Import Code & Helper Modules
import sys
sys.path.append(r"C:\\Users\\david.li\\Desktop\\WorkCode\\data-science-dump")

# Bring in Code Constants to Use
import MLConstants as mlc

def insertLogisticRegressionParams(version_num, penalty_param, dual_param, tol_param, C_param, fit_intercept_param, intercept_scaling_param,
                                   class_weight_param, random_state_param, solver_param, max_iter_param, multi_class_param, verbose_param,
                                   warm_start_param, n_jobs_param, l1_ratio_param):
    param_names = ["version_num", "penalty", "dual", "tol", "C", "fit_intercept", "intercept_scaling", "class_weight", "random_state", "solver",
                   "max_iter", "multi_class", "verbose", "warm_start", "n_jobs", "l1_ratio"]
    param_data = [version_num, penalty_param, dual_param, tol_param, C_param, fit_intercept_param, intercept_scaling_param,
                                   class_weight_param, random_state_param, solver_param, max_iter_param, multi_class_param, verbose_param,
                                   warm_start_param, n_jobs_param, l1_ratio_param]
    param_df = pd.DataFrame(columns = param_names)
    param_df.columns = map(lambda x: str(x).upper(), param_df.columns)
    param_df.loc[len(param_df)] = param_data

    #Create connection to Snowflake using your account and user
    account_identifier = 'DESENSITIZED'
    user = 'DESENSITIZED'
    password = 'DESENSITIZED'
    database_name = mlc.DATASCI_DB
    schema_name = mlc.MODEL_PARAMS_SCHEMA

    conn_string = f"snowflake://{user}:{password}@{account_identifier}/{database_name}/{schema_name}"
    registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')
    engine = create_engine(conn_string)

    with engine.connect() as conn:
        param_df.to_sql(name = mlc.LOGREG_PARAM_TBL, con=engine, schema = mlc.MODEL_PARAMS_SCHEMA, if_exists="append", method = pd_writer, index = False)

    return param_df

execute = insertLogisticRegressionParams(version_num = "v0", penalty_param = "l2", dual_param = False, tol_param = 0.0001, C_param = 1.0, 
                                         fit_intercept_param = True, intercept_scaling_param = 1, class_weight_param = None, random_state_param = None, 
                                         solver_param = "lbfgs", max_iter_param = 100, multi_class_param = "auto", verbose_param = 0,
                                         warm_start_param = False, n_jobs_param = None, l1_ratio_param = None)