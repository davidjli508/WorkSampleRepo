## Data Ingestion & Transformation for Churn Use Case

# Run Pre-Requisite/Helper Files/Functions
import sys
sys.path.append(r"C:\\Users\\david.li\\Desktop\\WorkCode\\data-science-dump\\")
# Packages Required for this Module to Run
import pandas as pd
import numpy as np
import MLConstants as mlc
import datetime

# Define Data Transformation Function
def ChurnDT(HH_DICT, TEAMMEM_DICT, OPP_DICT, FEESCHED_DICT, BILLING_DICT, WMS_DICT, EVENT_DICT, SF_CREDS, display_output):
    # ChurnDT: Data Transformation File for Churn at Client/Household Level
    # Get Run Timestamp
    run_timestamp = datetime.datetime.now(tz = datetime.timezone.utc)
    string_run_timestamp = run_timestamp.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Begin Grabbing Snowflake Data at: " + string_run_timestamp)
        print("")

    ## Assign into Individual DataFrame tables for Parsing
    hh_src = mlc.getSnowflakeData([mlc.tableToQuery(HH_DICT["TBL"])], SF_CREDS)[0]
    teammem_src = mlc.getSnowflakeData([mlc.tableToQuery(TEAMMEM_DICT["TBL"])], SF_CREDS)[0]
    opp_src = mlc.getSnowflakeData([mlc.tableToQuery(OPP_DICT["TBL"])], SF_CREDS)[0]
    feesched_src = mlc.getSnowflakeData([mlc.tableToQuery(FEESCHED_DICT["TBL"])], SF_CREDS)[0]
    billing_src = mlc.getSnowflakeData([mlc.tableToQuery(BILLING_DICT["TBL"])], SF_CREDS)[0]
    wms_src = mlc.getSnowflakeData([mlc.tableToQuery(WMS_DICT["TBL"])], SF_CREDS)[0]
    event_src = mlc.getSnowflakeData([mlc.tableToQuery(EVENT_DICT["TBL"])], SF_CREDS)[0]

    hh_f_src = mlc.getSnowflakeData([mlc.tableToQuery(HH_DICT["FOR"])], SF_CREDS)[0]
    teammem_f_src = mlc.getSnowflakeData([mlc.tableToQuery(TEAMMEM_DICT["FOR"])], SF_CREDS)[0]
    opp_f_src = mlc.getSnowflakeData([mlc.tableToQuery(OPP_DICT["FOR"])], SF_CREDS)[0]
    billing_f_src = mlc.getSnowflakeData([mlc.tableToQuery(BILLING_DICT["FOR"])], SF_CREDS)[0]
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Grabbing Snowflake Data Done at: " + string_run_timestamp)
        print("")

    ## Preemptive Cleaning/Transforming: 
    # Drop Duplicates for Tables
    for tbl in [hh_src, teammem_src, opp_src, feesched_src, billing_src, wms_src, event_src]:
        tbl.drop_duplicates(inplace = True)
    for tbl in [hh_f_src, teammem_f_src, opp_f_src, billing_f_src]:
        tbl.drop_duplicates(inplace = True)

    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Subsetting and Coercing Types in Pandas Dataframes at: " + string_run_timestamp)
        print("")
    # Generate Dataframes to prep joining
    # Household 
    hh_df_temp = mlc.subsetAndCoerce(hh_src, HH_DICT[mlc.CHN]["STR_COLS"], HH_DICT[mlc.CHN]["FLOAT_COLS"], HH_DICT[mlc.CHN]["DATE_COLS"])
    hh_f_df_temp = mlc.subsetAndCoerce(hh_f_src, HH_DICT[mlc.CHN]["STR_F_COLS"], HH_DICT[mlc.CHN]["FLOAT_F_COLS"], HH_DICT[mlc.CHN]["DATE_F_COLS"])
    # Team Member
    teammem_df_temp = mlc.subsetAndCoerce(teammem_src, TEAMMEM_DICT[mlc.CHN]["STR_COLS"], TEAMMEM_DICT[mlc.CHN]["FLOAT_COLS"], TEAMMEM_DICT[mlc.CHN]["DATE_COLS"])
    teammem_f_df_temp = mlc.subsetAndCoerce(teammem_f_src, TEAMMEM_DICT[mlc.CHN]["STR_F_COLS"], TEAMMEM_DICT[mlc.CHN]["FLOAT_F_COLS"], TEAMMEM_DICT[mlc.CHN]["DATE_F_COLS"])
    # Opportunity
    opp_df_temp = mlc.subsetAndCoerce(opp_src, OPP_DICT[mlc.CHN]["STR_COLS"], OPP_DICT[mlc.CHN]["FLOAT_COLS"], OPP_DICT[mlc.CHN]["DATE_COLS"])
    opp_f_df_temp = mlc.subsetAndCoerce(opp_f_src, OPP_DICT[mlc.CHN]["STR_F_COLS"], OPP_DICT[mlc.CHN]["FLOAT_F_COLS"], OPP_DICT[mlc.CHN]["DATE_F_COLS"])
    # Fee Schedule
    feesched_df_temp = mlc.subsetAndCoerce(feesched_src, FEESCHED_DICT[mlc.CHN]["STR_COLS"], FEESCHED_DICT[mlc.CHN]["FLOAT_COLS"], FEESCHED_DICT[mlc.CHN]["DATE_COLS"])
    # Billing
    billing_df_temp = mlc.subsetAndCoerce(billing_src, BILLING_DICT[mlc.CHN]["STR_COLS"], BILLING_DICT[mlc.CHN]["FLOAT_COLS"], BILLING_DICT[mlc.CHN]["DATE_COLS"])
    billing_f_df_temp = mlc.subsetAndCoerce(billing_f_src, BILLING_DICT[mlc.CHN]["STR_F_COLS"], BILLING_DICT[mlc.CHN]["FLOAT_F_COLS"], BILLING_DICT[mlc.CHN]["DATE_F_COLS"])
    # WMS
    wms_df_temp = mlc.subsetAndCoerce(wms_src, WMS_DICT[mlc.CHN]["STR_COLS"], WMS_DICT[mlc.CHN]["FLOAT_COLS"], WMS_DICT[mlc.CHN]["DATE_COLS"])
    # Event
    event_df_temp = mlc.subsetAndCoerce(event_src, EVENT_DICT[mlc.CHN]["STR_COLS"], EVENT_DICT[mlc.CHN]["FLOAT_COLS"], EVENT_DICT[mlc.CHN]["DATE_COLS"])
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Subsetting and Coercing Pandas Dataframes Done at: " + string_run_timestamp)
        print("")

    ## Merge and Create Joined/Trimmed Dataframes
    # Helper function to reduce input dataframe to Use Case Scope Parameters
    def limitDFScope(inputDF):
        limitDF = inputDF[inputDF["WEALTH_CLIENT_C"].isin(["Client", "Archived"])]
        limitDF = limitDF[~limitDF["WEALTH_SEGMENT_C"].isin(["Open", "Friends & Family"])]
        limitDF = limitDF[limitDF["WEALTH_START_DATE_C"] > datetime.datetime(2010, 1, 1)]
        enddate_chunk1 = limitDF[limitDF["WEALTH_END_DATE_C"] > datetime.datetime(2020, 1, 1)]
        enddate_chunk2 = limitDF[limitDF["WEALTH_END_DATE_C"].isnull()] # handles NaT values
        limitDF = pd.concat([enddate_chunk1, enddate_chunk2])
        return limitDF
    
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Appending Tables to Their Formulas at: " + string_run_timestamp)
        print("")
    # Append tables to their formula counterpart tables
    hh_master = hh_df_temp.merge(hh_f_df_temp, how = 'left', on = 'ID')
    teammem_master = teammem_df_temp.merge(teammem_f_df_temp, how = 'left', on = 'ID')
    opp_master = opp_df_temp.merge(opp_f_df_temp, how = 'left', on = 'ID')
    billing_master = billing_df_temp.merge(billing_f_df_temp, how = 'left', on = 'ID')
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Appending Tables to Formulas Done at: " + string_run_timestamp)
        print("")

    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Recreating SOQL, Limiting Dataframes to Scope of Use Cases at: " + string_run_timestamp)
        print("")
    # Household Final
    hh_final = limitDFScope(hh_master)
    hh_final.drop(["ID", "WEALTH_SEGMENT_C"], axis = 1, inplace = True)
    # Team Member Turnover Final
    teammem_final = limitDFScope(
        teammem_master.merge(hh_master, 
            how = 'left', 
            left_on = 'CLIENT_C', 
            right_on = 'ID', 
            suffixes = ('_tm', None))
        )
    teammem_final = teammem_final[teammem_final["TEAM_ROLE_C"].isin(["Analyst", 
                                                                    "Associate Advisor", 
                                                                    "Lead Advisor", 
                                                                    "Senior Lead Advisor"])]
    teammem_final.drop(["ID_tm", "CLIENT_C", "ID", "TAG_CLIENT_C", "OFFICE_LOCATION_C", "VALUE_STACK_C", 
                        "CLIENT_ENGAGEMENT_RANK_C", "BILLABLE_NET_WORTH_C", "WEALTH_START_DATE_C", 
                        "WEALTH_END_DATE_C", "POD_NAME_C", "WEALTH_SEGMENT_C"], 
                        axis = 1, inplace = True)
    # LA Transitions Final
    latrans_final = limitDFScope(
        teammem_master.merge(hh_master, 
            how = 'left', 
            left_on = 'CLIENT_C', 
            right_on = 'ID', 
            suffixes = ('_la', None)
        )
    )
    latrans_final = latrans_final[latrans_final["TEAM_ROLE_C"].isin(["Lead Advisor"])]
    latrans_final = latrans_final[~latrans_final["ROLE_STATUS_C"].isin(["Future"])]
    latrans_final.drop(["ID_la", "CLIENT_C", "ID", "TAG_CLIENT_C", "TEAM_ROLE_C", "ROLE_STATUS_C", "OFFICE_LOCATION_C", 
                        "VALUE_STACK_C", "NAME", "CLIENT_TENURE_C","TENURE_C", "CLIENT_ENGAGEMENT_RANK_C", "BILLABLE_NET_WORTH_C", 
                        "WEALTH_START_DATE_C", "POD_NAME_C", "WEALTH_SEGMENT_C"], 
                        axis = 1, inplace = True)
    # Opportunity Final
    opp_final = limitDFScope(
        opp_master.merge(hh_master,
            how = 'left',
            left_on = 'ACCOUNT_ID',
            right_on = 'ID',
            suffixes = ('_opp', None)
        )
    )
    opp_final = opp_final[opp_final["STAGE_NAME"] == "Stage 4 - New Client"]
    opp_final = opp_final[opp_final["NAME_opp"].str.contains("Advisory", case = False)]
    opp_final.drop(["ID_opp", "ACCOUNT_ID", "STAGE_NAME", "ID", "TAG_CLIENT_C", "OFFICE_LOCATION_C", 
                        "VALUE_STACK_C", "CLIENT_TENURE_C", "CLIENT_ENGAGEMENT_RANK_C", "BILLABLE_NET_WORTH_C", 
                        "WEALTH_START_DATE_C","WEALTH_END_DATE_C", "POD_NAME_C", "WEALTH_SEGMENT_C", "WEALTH_CLIENT_C"], 
                        axis = 1, inplace = True)
    opp_final = opp_final.merge(feesched_df_temp, 
                                how = 'left', 
                                left_on = "FEE_SCHEDULE_C", 
                                right_on = "ID",
                                suffixes = (None, '_fee'))
    opp_final.drop(["ID", "FEE_SCHEDULE_C"], axis = 1, inplace = True)
    # Billing/BPS Final
    billing_final = limitDFScope(
        billing_master.merge(hh_master,
            how = 'left',
            left_on = 'CLIENT_ID_2_C',
            right_on = 'CLIENT_ID_C',
            suffixes = ('_bill', None)
        )
    )
    billing_final = billing_final[~billing_final["RECORD_TYPE_ID"].isin(mlc.RTYPE_REIMBURSABLES + mlc.RTYPE_TAG_LIST + mlc.RTYPE_IAM_LIST)]
    billing_final = billing_final[billing_final["TOTAL_INVOICE_C"] != 0.0]
    billing_final.drop(["ID_bill","RECORD_TYPE_ID","CLIENT_ID_2_C","ID",
                        "WEALTH_CLIENT_C","TAG_CLIENT_C","OFFICE_LOCATION_C","WEALTH_SEGMENT_C",
                        "VALUE_STACK_C","CLIENT_ENGAGEMENT_RANK_C","BILLABLE_NET_WORTH_C","WEALTH_START_DATE_C",
                        "WEALTH_END_DATE_C","POD_NAME_C","CLIENT_TENURE_C"], 
                        axis = 1, inplace = True)
    # BJIP Final  
    wms_final = limitDFScope(
        wms_df_temp.merge(hh_master,
            how = 'left',
            left_on = 'HOUSEHOLD_C',
            right_on = 'ID',
            suffixes = ('_wms', None)
        )
    )
    wms_final.drop(["HOUSEHOLD_C", "ID", "WEALTH_SEGMENT_C", "VALUE_STACK_C",
                    "WEALTH_CLIENT_C","TAG_CLIENT_C","OFFICE_LOCATION_C", "NAME",
                    "CLIENT_ENGAGEMENT_RANK_C","BILLABLE_NET_WORTH_C","WEALTH_START_DATE_C",
                    "WEALTH_END_DATE_C","POD_NAME_C","CLIENT_TENURE_C"], 
                    axis = 1, inplace = True)
    # Events Final
    event_final = limitDFScope(
        event_df_temp.merge(hh_master,
            how = 'left',
            left_on = 'ACCOUNT_ID',
            right_on = 'ID',
            suffixes = ('_event', None)
        )
    )
    event_final = event_final[event_final["RECORD_TYPE_ID"].isin(mlc.RTYPE_EVENT)]
    event_final.drop(["ACCOUNT_ID", "RECORD_TYPE_ID", "ID", "TAG_CLIENT_C", "OFFICE_LOCATION_C",
                    "NAME", "WEALTH_SEGMENT_C","CLIENT_ENGAGEMENT_RANK_C","BILLABLE_NET_WORTH_C",
                    "WEALTH_START_DATE_C", "POD_NAME_C","CLIENT_TENURE_C", "VALUE_STACK_C"], 
                    axis = 1, inplace = True)
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("SOQL Translations Done, Final Form of Dataframes Ready at: " + string_run_timestamp)
        print("")

    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Generating Summary Metrics of Groupbys at: " + string_run_timestamp)
        print("")
    ## Data Cleaning & Summary Attributes ##
    # Average Quarterly Fees, calculated from billing information, Annual_fees.csv
    avg_q_fee_df = billing_final.copy(deep = True).dropna()
    # Create a Rank Column, descending starting from most recent date
    avg_q_fee_df["RANK"] = avg_q_fee_df.groupby("CLIENT_ID_C")["BILL_PERIOD_DATE_C"].rank(method = 'dense', ascending = False)
    # Only Retain Last 4 Quarters
    avg_q_fee_df = avg_q_fee_df[avg_q_fee_df["RANK"].isin([1.0, 2.0, 3.0, 4.0])]
    # Compute Average Quarterly Fee, an average from last 4 applicable quarters
    avg_q_fee_df = avg_q_fee_df.groupby("CLIENT_ID_C")["TOTAL_INVOICE_C"].mean().reset_index()
    # Rename Column
    avg_q_fee_df.rename(columns = {"TOTAL_INVOICE_C": "AVG_QUARTERLY_FEE"}, inplace = True)

    # Get Last Known BPS, from most recent date, 7last_BPS.csv
    bps_df = billing_final.copy(deep = True).dropna()
    # Group by and obtain latest BPS per client_id
    bps_df = bps_df.sort_values("BILL_PERIOD_DATE_C", ascending = True).groupby("CLIENT_ID_C").tail(1)
    bps_df = bps_df[["CLIENT_ID_C", "BPS_C"]]
    bps_df.rename(columns = {"BPS_C": "LAST_BPS"}, inplace = True)

    # Get Number of Distinct PCFO Team Members, 5team_turnover.csv
    teamturnover_df = teammem_final.copy(deep = True).dropna()
    teamturnover_df = teamturnover_df.groupby("CLIENT_ID_C", as_index = False)["NAME_tm"].nunique()
    teamturnover_df.rename(columns = {"NAME_tm": "NUM_DISTINCT_MEM"}, inplace = True)

    # Get turnover over time by distinct PCFO Team Members, 5turnover_over_time.csv
    teamturnovertime_df = teamturnover_df.copy(deep = True)
    teamturnovertime_df = teamturnovertime_df.merge(hh_final[["CLIENT_ID_C", "CLIENT_TENURE_C"]], how = "left", on = "CLIENT_ID_C")
    teamturnovertime_df["TO_OVER_TIME"] = teamturnovertime_df["CLIENT_TENURE_C"] / teamturnovertime_df["NUM_DISTINCT_MEM"]
    teamturnovertime_df.drop(["CLIENT_TENURE_C", "NUM_DISTINCT_MEM"], axis = 1, inplace = True)

    # Select max close date to identify/select most recent opportunity, 4opportunity.csv
    recentopp_df = opp_final.copy(deep = True).dropna()
    recentopp_df = recentopp_df.sort_values("CLOSE_DATE", ascending = True).groupby("CLIENT_ID_C").tail(1)
    recentopp_df = recentopp_df[["CLIENT_ID_C", "LEAD_SOURCE", "NAME_fee", "SCHEDULE_DESCRIPTION_C", "CLOSE_DATE"]]

    # Create feature indicating if client is BJIP investor (Yes/No), 2BJIP.csv
    bjip_df = wms_final.copy(deep = True).dropna()
    bjip_df[['MEMBER_OF_BJIP_I_C', 'MEMBER_OF_BJIP_II_C']] = (bjip_df[['MEMBER_OF_BJIP_I_C', 'MEMBER_OF_BJIP_II_C']] == 'True').astype(int)
    bjip_df['BJIP_INVESTOR'] = np.where((bjip_df["MEMBER_OF_BJIP_I_C"] + bjip_df["MEMBER_OF_BJIP_II_C"]) > 0, 'Yes', 'No')
    bjip_df.drop_duplicates(subset = ["CLIENT_ID_C"], inplace = True)
    bjip_df.drop(["MEMBER_OF_BJIP_I_C", "MEMBER_OF_BJIP_II_C"], axis = 1, inplace = True)

    # Determine count of events per client in last 365 days from departure or current date, depending on archived or current, 3Events.csv
    eventcount_df = event_final.copy(deep = True)
    # Determine Last Date feature, which is either current date or their wealth end date if client is respectively current or archived
    eventcount_df["LAST_DATE"] = np.where(eventcount_df["WEALTH_CLIENT_C"] == "Archived", eventcount_df["WEALTH_END_DATE_C"], datetime.datetime.now())
    eventcount_df["LAST_DATE"] = eventcount_df["LAST_DATE"].apply(pd.to_datetime, errors = "coerce", utc = True) # DateTime
    # Filter out duplicates depending on combination of start date + subject
    eventcount_df.drop_duplicates(subset = ["START_DATE_TIME", "SUBJECT"], inplace = True)
    # Filter out Internal Events
    eventcount_df = eventcount_df[~eventcount_df["SUBTYPE_C"].isin(["Internal Client Review", "Internal Meeting Prep"])]
    # Filter to just last 365 days of events
    eventcount_df['DIFF_DAYS'] = (eventcount_df['LAST_DATE'] - eventcount_df['START_DATE_TIME']) / np.timedelta64(1, 'D')
    eventcount_df = eventcount_df[eventcount_df["DIFF_DAYS"] <= 365.00]
    # Group by and count
    eventcount_df = eventcount_df.groupby("CLIENT_ID_C", as_index = False).size()
    eventcount_df.rename(columns = {"size": "NUM_EVENTS_365"}, inplace = True)

    # Get Info on LA Transitions, 6LAtransition.csv
    # add feature if this LA was the only one in the entire client relationship
    temp_num_LA_df = latrans_final.groupby("CLIENT_ID_C", as_index = False).size()
    latranscount_df = latrans_final.merge(temp_num_LA_df, how = "left", on = "CLIENT_ID_C")
    latranscount_df["size"] = np.where(latranscount_df["size"] > 1, 'FALSE', 'TRUE')
    latranscount_df.rename(columns = {"size": "UNIQUE_MEM_FOR_CLIENT"}, inplace = True)

    # Calculate number of transitions between 180 days of departure or current date, depending on archived or current client
    # First determine the 'past 180 days' range, depending if archived or current client
    latranscount_df["DAYS_180"] = np.where(latranscount_df["WEALTH_CLIENT_C"] == "Archived", latranscount_df["WEALTH_END_DATE_C"] - datetime.timedelta(days = 180), datetime.datetime.now() - datetime.timedelta(days = 180))
    latranscount_df["DAYS_180"] = latranscount_df["DAYS_180"].apply(pd.to_datetime, errors = "coerce") # DateTime
    latranscount_df["DAYS_LAST"] = np.where(latranscount_df["WEALTH_CLIENT_C"] == "Archived", latranscount_df["WEALTH_END_DATE_C"], datetime.datetime.now())
    latranscount_df["DAYS_LAST"] = latranscount_df["DAYS_LAST"].apply(pd.to_datetime, errors = "coerce") # DateTime
    # Then Determine if this LA was assigned within 180 days of last relevant client date
    latranscount_df["ALL_BEFORE"] = np.where(((latranscount_df["START_DATE_C"] < latranscount_df["DAYS_180"]) & (latranscount_df["END_DATE_C"] < latranscount_df["DAYS_180"])), 1, 0)
    latranscount_df["ALL_AFTER"] = np.where(((latranscount_df["START_DATE_C"] > latranscount_df["DAYS_LAST"]) & (latranscount_df["END_DATE_C"] > latranscount_df["DAYS_LAST"])), 1, 0)
    latranscount_df["VALID_FLAG"] = np.where((latranscount_df["ALL_BEFORE"] + latranscount_df["ALL_AFTER"]) == 0, "IN_PAST_180", "OUT_OF_180")
    # Filter to only LAs in last 180 days, then count by number per client
    latranscount_df = latranscount_df[latranscount_df["VALID_FLAG"] == "IN_PAST_180"]
    latranscount_df = latranscount_df.groupby("CLIENT_ID_C", as_index = False)["VALID_FLAG"].count()
    # We just calculated number of LAs, so number of transitions is 1 less than this number
    latranscount_df["VALID_FLAG"] -= 1
    latranscount_df.rename(columns = {"VALID_FLAG": "NUM_LA_TRANS_IN_PAST_180"}, inplace = True)
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Generating Summary Metrics of Groupbys Done at: " + string_run_timestamp)
        print("")

    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Creating Final Dataframe at: " + string_run_timestamp)
        print("")
    ## Merge All Summary Info Together
    joined_final = hh_final.merge(
        avg_q_fee_df, how = "outer", on = "CLIENT_ID_C").merge( # Adds avg_quarterly fee
        bps_df, how = "outer", on = "CLIENT_ID_C").merge( # Adds BPS
        teamturnover_df, how = "outer", on = "CLIENT_ID_C").merge( # Adds Numer of Distinct team members
        teamturnovertime_df, how = "outer", on = "CLIENT_ID_C").merge( # Adds Turnover over time
        recentopp_df, how = "outer", on = "CLIENT_ID_C").merge( # Adds Lead source, Fee Schedule Name, Schedule Desc. Close_date
        bjip_df, how = "outer", on = "CLIENT_ID_C").merge( # Adds if they are a BJIP investor
        eventcount_df, how = "outer", on = "CLIENT_ID_C").merge( # Adds Number of Significant events in last 365 days
        latranscount_df, how = "outer", on = "CLIENT_ID_C" # Adds number of LA transitions
        )
    # Print Info
    if display_output.upper() in ["Y", "YES"]:
        print("Created Final Dataframe at: " + string_run_timestamp)
        print("")
    
    # Do Feature Engineering
    if display_output.upper() in ["Y", "YES"]:
        print("Begin Feature Engineering at: " + string_run_timestamp)
        print("")
    # Remove spaces in Value Stack Column
    joined_final["VALUE_STACK_C"].replace(' ', '_', regex=True, inplace = True)

    # Create Dual Client Feature
    joined_final["DUAL_CLIENT"] = np.where((joined_final["WEALTH_CLIENT_C"].isin(["Client", "Archived"]) & 
                                            (joined_final["TAG_CLIENT_C"].isin(["Client", "archived"]))), "Yes", "No")
    # Create Churn Feature
    joined_final["CHURN"] = np.where((joined_final["WEALTH_CLIENT_C"] == "Archived"), 1, 0)
                                                
    ## Impute/Handle Missing Features ##

    # Remove Rows where None was a substituted value for a missing value
    joined_final = joined_final[~joined_final["VALUE_STACK_C"].isin(["None"])]
    joined_final = joined_final[~joined_final["CLIENT_ENGAGEMENT_RANK_C"].isin(["None"])]

    # Drop NAs for Used Features, these are unclear on how to resolve for now besides dropping
    joined_final.dropna(subset = mlc.CHURN_MODELING_COLUMNS, inplace = True)

    ## Model Dataset Preparation ##
    joined_final = joined_final[mlc.CHURN_MODELING_COLUMNS]
    
    return joined_final

