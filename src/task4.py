import pandas as pd
import psycopg2 # type: ignore
from sqlalchemy import create_engine # type: ignore
import numpy as np
from IPython.display import Image # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import FunctionTransformer # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore












# Function to calculate missing values by column
def missing_values_table(df: pd.DataFrame)-> pd.DataFrame:
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


## remove rows without customer Id
def remove_rows_without_customerid(df: pd.DataFrame)-> pd.DataFrame:
    #remove rows
    clean_data= df[~df["MSISDN/Number"].isna()]
    return clean_data


## fIX nas
def fix_na(df: pd.DataFrame)-> pd.DataFrame:
    columns = df.columns
    for column in columns:
        column_mode = df[column].mode()
        df[column] = np.where(df[column].isna(), column_mode, df[column])
    return df


## Finding and treating outliers a
def fix_outlier(df: pd.DataFrame)-> pd.DataFrame:
    non_object_columns = df.select_dtypes(exclude=["object"]).columns
    for column in non_object_columns:
        column_mode = df[column].mode()
        column_quantile = df[column].quantile(0.95)
        df[column] = np.where(df[column] > column_quantile, column_mode, df[column])
    return df




# function to aggregate column per user
def aggregate_column(df: pd.DataFrame, groupby_column: str = 'name', 
                     aggregate_column: str = 'data_collection',
                     aggregate_function: str = 'data_collection' ) -> pd.DataFrame:
    # target columns
    targets= df[[groupby_column, aggregate_column]]
    
    # make sure the columns are in the dataframe
    assert groupby_column in df.columns, f'"groupby_column", {groupby_column}, '\
                                         'does not appear to be in the input '\
                                         'DataFrame columns.'
    assert aggregate_column in df.columns, f'"aggregate_column", {aggregate_column}, '\
                                           'does not appear to be in the input '\
                                           'DataFrame columns.'
    
    # create aggregated dataframe
    try:
        agg_df = targets.groupby(groupby_column).aggregate({aggregate_column: aggregate_function})
    except:
        agg_df= df.groupby(groupby_column)[aggregate_column].first().reset_index()

    # add group by column as variable not index
    agg_df[groupby_column]=agg_df.index
    agg_df.reset_index(drop=True, inplace=True)
    agg_df=agg_df[[groupby_column,aggregate_column]]
    return  agg_df

# total data for metric
def total_data_metric(df, dictionary):
    for key, values in dictionary.items():
        df[key]=df[values[0]] + df[values[1]]
    return df

def aggregate_merge_metrics(df: pd.DataFrame, groupby_column: str = 'name', 
                     aggregate_function: str = 'data_collection' ) -> pd.DataFrame:
    dfs=[]
    metrics=['Handset Type', 'RTT', 'TCP', 'TP']
    for i in range(len(metrics)):
            aggdf=aggregate_column(df,groupby_column , 
                        metrics[i],
                        aggregate_function)
            if i==0:
                dfs.append(aggdf)
            else:
                dfs.append(aggdf.iloc[:,-1])
    aggregate_data = pd.concat(dfs, axis=1)
    return aggregate_data


def plot_count_bottom(df: pd.DataFrame, column: str, bottom: int) -> None:
    plt.figure(figsize=(12, 7))
    top_modalities = df[column].value_counts().nsmallest(bottom).index  # corrected from nlowest to nsmallest
    sns.countplot(data=df[df[column].isin(top_modalities)], x=column, order=top_modalities)
    plt.xticks(rotation=75, fontsize=14)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()



def plot_bar(df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()