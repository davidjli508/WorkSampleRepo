## Start-to-End Execution for BJ Client Churn Use Case with Logistic Regression

# Import Code & Helper Modules
# Specify Path of Files
import sys
sys.path.append(r"C:\\Users\\david.li\\Desktop\\WorkCode\\data-science-dump")

# 1. Bring in Code Constants to Use
import MLConstants as mlc

# 2. Run Churn Data Transformation File
import Data.ChurnDT as DT
model_df = DT.ChurnDT(mlc.HH_DICT, mlc.TEAMMEM_DICT, mlc.OPP_DICT, mlc.FEESCHED_DICT, 
                   mlc.BILLING_DICT, mlc.WMS_DICT, mlc.EVENT_DICT, mlc.CRED_SF_DICT, 
                   display_output = "y")

# 3. Run Churn Modelling, with Specified Model TYpe
import Modeling.Classification.LogisticRegression as modeling
model_result = modeling.RunLogisticRegression(learning_df = model_df, preprocess_list = ["SMOTE", "NORMALIZE"], version_num = "v0", 
                               id_var = "CLIENT_ID_C", target_var = "CHURN", pred_df = model_df, use_case_desc = "Client Churn Use Case",
                               pred_table_name = mlc.CHURN_OUTPUT_TBL, details_table_name = mlc.LOGREG_DETAIL_TBL, metrics_table_name = mlc.LOGREG_METRICS_TBL, 
                               write_snowflake = "y", display_output = "y")
