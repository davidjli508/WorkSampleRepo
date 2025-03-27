### ML CONSTANTS ###

## Packages in Environment Needed
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

## DATA CONSTANTS
# RecordType Values 
RTYPE_EVENT = ["012E0000000NCC9IAO"]
RTYPE_REIMBURSABLES = ["012E0000000NBNsIAO"]
RTYPE_TAG_LIST = ["012E0000000MqTiIAK", "012E0000000NAWlIAO", "012E0000000NCwxIAG"]
RTYPE_IAM_LIST = ["01244000000RfrhAAC", "01244000000DJ5aAAG"]

# CODING Constants
# 3 Letter Use Case Codes, for Dictionary/Table Lookup
CHN = "CHURN"
# Preprocessing tasks
NORMALIZE_TASK = "NORMALIZE"
SMOTE_TASK = "SMOTE"
# Probability Varaible Prefix for Output
PROBA_PREFIX = "Proba_"
# Prediction Variable Suffix for Output
PRED_SUFFIX = "_Pred"
# Model Names
LOG_MODEL_NAME = "LogisticRegression"
RF_MODEL_NAME = "RandomForest"

# Columns for Modeling in Use Cases
CHURN_MODELING_COLUMNS = ["CLIENT_ID_C", "VALUE_STACK_C", "CLIENT_ENGAGEMENT_RANK_C", 
                                "BILLABLE_NET_WORTH_C", "LAST_BPS",
                                "TO_OVER_TIME", "NUM_EVENTS_365", 
                                "NUM_LA_TRANS_IN_PAST_180", "CHURN"] # Churn

##INTAKE SNOWFLAKE TABLES & COLUMN SPECS
# Snowflake Connector Creds for Salesforce in fivetran_staging in Snowflake (May need to Secure this in Future)
CRED_SF_DICT = {"USER": "DESENSITIZED",
    "PW": "",
    "ACCT":"DESENSITIZED",
    "WH": "DESENSITIZED",
    "DB": "DESENSITIZED",
    "SCHEMA": "DESENSITIZED",
    "ROLE": "DESENSITIZED"
}

# Snowflake Connector Creds for DataScience Parameters Tables in Snowflake (May need to Secure this in Future)
CRED_DS_DICT = {"USER": "DESENSITIZED",
    "PW": "",
    "ACCT":"DESENSITIZED",
    "WH": "DESENSITIZED",
    "DB": "DESENSITIZED",
    "SCHEMA": "DESENSITIZED",
    "ROLE": "DESENSITIZED"
}

# Household Table and Columns
HH_DICT = {
    "TBL": "ACCOUNT", 
    "FOR": "ACCOUNT_FORMULAS", 
    "CHURN": {
        "STR_COLS": ["ID", "CLIENT_ID_C", "WEALTH_CLIENT_C", "TAG_CLIENT_C", "OFFICE_LOCATION_C", "NAME", "WEALTH_SEGMENT_C", "VALUE_STACK_C", "CLIENT_ENGAGEMENT_RANK_C"], 
        "FLOAT_COLS": ["BILLABLE_NET_WORTH_C"],
        "DATE_COLS": ["WEALTH_START_DATE_C", "WEALTH_END_DATE_C"],
        "STR_F_COLS": ["ID", "POD_NAME_C"],
        "FLOAT_F_COLS": ["CLIENT_TENURE_C"],
        "DATE_F_COLS": []
    }
}
# Team Member Table and Columns
# Dev Notes: Joining Info. team_member_c.client_c = account.id
TEAMMEM_DICT = { 
    "TBL": "TEAM_MEMBER_C", 
    "FOR": "TEAM_MEMBER_FORMULAS", 
    "CHURN": {
        "STR_COLS": ["ID", "CLIENT_C", "NAME", "TEAM_ROLE_C", "ROLE_STATUS_C"], 
        "FLOAT_COLS": [],
        "DATE_COLS": ["START_DATE_C", "END_DATE_C"],
        "STR_F_COLS": ["ID"],
        "FLOAT_F_COLS": ["TENURE_C"],
        "DATE_F_COLS": []
    }
}
# Opportunity Table and Columns
# Dev Notes: Joining Info. opportunity.account_id = account.id, opportunity.fee_schedule_c = fee_schedule_c.id, opportunity.id = billing_c.opportunity.c
OPP_DICT = {
    "TBL": "OPPORTUNITY", 
    "FOR": "OPPORTUNITY_FORMULAS", 
    "CHURN": {
        "STR_COLS": ["ID", "ACCOUNT_ID", "FEE_SCHEDULE_C" ,"NAME", "LEAD_SOURCE", "SEGMENT_C", "STAGE_NAME"], 
        "FLOAT_COLS": [],
        "DATE_COLS": ["CLOSE_DATE"],
        "STR_F_COLS": ["ID"],
        "FLOAT_F_COLS": [],
        "DATE_F_COLS": []
    }
}
# Fee Schedule Table and Columns
# Dev Notes: fee_schedule_c.id = opportunity.fee_schedule_c
FEESCHED_DICT = {
    "TBL": "FEE_SCHEDULE_C", 
    "FOR": "", 
    "CHURN": {
        "STR_COLS": ["ID", "NAME", "SCHEDULE_DESCRIPTION_C"], 
        "FLOAT_COLS": [],
        "DATE_COLS": [],
        "STR_F_COLS": [],
        "FLOAT_F_COLS": [],
        "DATE_F_COLS": []
    }
}
# Billing Table and Columns
# Dev Notes: billing_c.client_id_2_c = account.client_id_c, billing_c.opportunity_c = opportunity.id
BILLING_DICT = {
    "TBL": "BILLING_C", 
    "FOR": "BILLING_FORMULAS", 
    "CHURN": {
        "STR_COLS": ["ID","NAME", "RECORD_TYPE_ID"], 
        "FLOAT_COLS": [],
        "DATE_COLS": [],
        "STR_F_COLS": ["ID", "CLIENT_ID_2_C", "STATEMENT_NAME_C", "BILL_PERIOD_C"],
        "FLOAT_F_COLS": ["TOTAL_INVOICE_C", "BPS_C"],
        "DATE_F_COLS": ["BILL_PERIOD_DATE_C"]
    }
}
# WMS Table and Columns
# Dev Notes: wms_planning_c.household_c = account.ID
WMS_DICT = {
    "TBL": "WMS_PLANNING_C", 
    "FOR": "WMS_PLANNING_FORMULAS", 
    "CHURN": {
        "STR_COLS": ["HOUSEHOLD_C", "MEMBER_OF_BJIP_I_C", "MEMBER_OF_BJIP_II_C"], 
        "FLOAT_COLS": [],
        "DATE_COLS": [],
        "STR_F_COLS": [],
        "FLOAT_F_COLS": [],
        "DATE_F_COLS": []
    }
}
# Events Table and Columns
# Dev Notes: event.account_id = account.ID
EVENT_DICT = {
    "TBL": "EVENT", 
    "FOR": "", 
    "CHURN": {
        "STR_COLS": ["ACCOUNT_ID", "DESCRIPTION", "ID", "SUBJECT", "SUBTYPE_C", "RECORD_TYPE_ID"], 
        "FLOAT_COLS": [],
        "DATE_COLS": ["START_DATE_TIME"],
        "STR_F_COLS": [],
        "FLOAT_F_COLS": [],
        "DATE_F_COLS": []
    }
}

## DESTINATION SNOWFLAKE TABLES
DATASCI_DB = "DESENSITIZED"
# MODEL_METRICS
MODEL_METRICS_SCHEMA = "MODEL_METRICS"
LOGREG_METRICS_TBL = "LOGREG_METRICS" # Logistic Regression
RF_METRICS_TBL = "RF_METRICS" # Random Forest 
XGB_METRICS_TBL = "XGB_METRICS" # Gradient Boosted Machine/Trees
# MODEL_DETAILS
MODEL_DETAILS_SCHEMA = "MODEL_DETAILS"
LOGREG_DETAIL_TBL = "LOGREG_MODEL_DETAILS"
RF_DETAIL_TBL = "RF_MODEL_DETAILS"
XGB_DETAIL_TBL = "XGB_MODEL_DETAILS"
# MODEL_OUTPUT
MODEL_OUTPUT_SCHEMA = "MODEL_OUTPUT"
CHURN_OUTPUT_TBL = "CHURN_OUTPUT"
# MODEL_PARAMS
MODEL_PARAMS_SCHEMA = "MODEL_PARAMS"
LOGREG_PARAM_TBL = "LOGREG_PARAM_SET"
RF_PARAM_TBL = "RF_PARAM_SET"
XGB_PARAM_TBL = "XGB_PARAM_SET"


## FUNCTIONS
# Helper Function to Translate Table Name into Select Query Form
def tableToQuery(tbl_name):
    select_string = "SELECT * FROM " + tbl_name
    return select_string

# Get Snowflake Data Helper Function. Accesses all scripts in list using 1 specified credential list
# Needs: scriptList is list of Scripts from Constants, credentialDict is list of credentials from Constants
# Returns: List of Pandas Dataframes
def getSnowflakeData(scriptList, credentialDict):
    # Connect to Snowflake Warehouse
    conn = sconn.connect(
        user = credentialDict["USER"],
        password = credentialDict["PW"],
        account = credentialDict["ACCT"],
        warehouse = credentialDict["WH"],
        database = credentialDict["DB"],
        schema = credentialDict["SCHEMA"],
        role = credentialDict["ROLE"]
    )
    # Create Cursor Object
    cur = conn.cursor()
    # Create List to Hold SQL Results/Pandas Dataframes
    df_list = []
    # Execute Multiple Scripts to get Data Tables
    for script in scriptList:
        cur.execute(script)
        # Get in Pandas Dataframe Form
        all_rows = cur.fetchall()
        field_names = [i[0] for i in cur.description]
        target_data = pd.DataFrame(all_rows)
        target_data.columns = field_names
        # Append to running list of Dataframes
        df_list.append(target_data)

    return df_list

# Subset Data & Coerce Types Helper Function. Accesses all scripts in list using 1 specified credential list
# Needs: sourceTable, being a Source Table to apply these transformations to. Lists of what columns to subset and their types
# Returns: A Pandas Dataframe, with subsetting and coercetypes applied
def subsetAndCoerce(source_table, str_col_list, float_col_list, date_col_list):
    subset_cols_list = []
    # Provide columns to retain
    for listItem in [str_col_list, float_col_list, date_col_list]:
        if (len(listItem) > 0):
            subset_cols_list.extend(listItem)
    # Get Subsetted Dataframe
    subset_df = source_table[subset_cols_list]
    # Coerce Types
    subset_df[str_col_list] = subset_df[str_col_list].astype(str) # String
    subset_df[float_col_list] = subset_df[float_col_list].astype(float) # FLoats/Decimals
    subset_df[date_col_list] = subset_df[date_col_list].apply(pd.to_datetime, errors = "coerce") # DateTime

    return subset_df
