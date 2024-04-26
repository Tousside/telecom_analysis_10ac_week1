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



def connect_to_database(connection_params: dict):
    """
    Connects to the PostgreSQL database.
    paramters:
        connection_params is a dictionary that define the following:
        {
            'dbname': 'your_database_name',
            'user': 'your_username',
            'password': 'your_password',
            'host': 'your_host',
            'port': 'your_port'
            }
    """
    try:
        connection = psycopg2.connect(**connection_params)
        return connection
    except psycopg2.Error as e:
        print(f"Error: Unable to connect to the database. {e}")
        return None
    

def read_table_to_dataframe(table_name, connection_params):
    """
    Reads a PostgreSQL table into a pandas dataframe.
    """
    connection = connect_to_database(connection_params)
    if connection:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df
    else:
        print("Error, no connection detected!")
        return None



# Function to remove variable with more than 30% missing values
def missing_values_table(df):
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

    # subset the variables to the one with more than 30% missin values
    df30=mis_val_table_ren_columns[mis_val_table_ren_columns["% of Total Values"]>30]
    # extract these variables names
    missing_variables= df30.index.tolist()
    # drop these variables
    df_whithout_30= df.drop(missing_variables, axis=1)
    # Return the dataframe with more than 30% missing values v ariables removed
    return df_whithout_30

## remove each row without xDr session identifier
def remove_rows_without_sessionid(df):
    #remove rows
    clean_data= df[~df["Bearer Id"].isna()]
    return clean_data
## remove rows without customer Id
def remove_rows_without_customerid(df):
    #remove rows
    clean_data= df[~df["MSISDN/Number"].isna()]
    return clean_data



## remove last locations since they are all None 

def remove_last_location(df):
    df=df.drop("Last Location Name", axis=1)
    return df

## Finding and treating outliers and fix remainin nas
def fix_na_outlier(df):
    non_object_columns = df.select_dtypes(exclude=['object']).columns
    for column in non_object_columns:
        column_mean = df[column].mean()
        column_quantile = df[column].quantile(0.95)
        df[column] = np.where((df[column] > column_quantile) | df[column].isna(), column_mean, df[column])
    return df

### function to create the total data volume
def total_data(df):
    df["total data (Bytes)"]=df["Total UL (Bytes)"]+\
    df["Total DL (Bytes)"]
    return df

 
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
    agg_df = targets.groupby(groupby_column).aggregate({aggregate_column: aggregate_function})
    
    # create a dataframe of values without the aggregation column ready to join
    df_alias = 	targets.drop(columns=aggregate_column).\
        set_index(groupby_column)
    
    # join the aliases on and clean up the result
    out_df = agg_df.join(df_alias).\
        reset_index(groupby_column).\
        drop_duplicates(groupby_column).\
        reset_index(drop=True)
    
    return out_df

def total_data_app(df, dictionary):
    for key, values in dictionary.items():
        df[key]=df[values[0]] + df[values[1]]
        df=df.drop(values, axis=1)
    return df

def plot_bar(df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str, top_manufacturer: int, top_handset: int)->None:
    top_manufacturers_modalities = df[x_col].value_counts().nlargest(top_manufacturer).index

    top_manufacturer_data= df[df["Handset Manufacturer"].isin(top_manufacturers_modalities)]

    top_handsets_modalities = top_manufacturer_data[y_col].value_counts().nlargest(top_handset).index

    top_manufacturer_handset_data= top_manufacturer_data[df["Handset Type"].isin(top_handsets_modalities)]
    # selected_data= data = df[df[x_col].isin(top_manufacturers_modalities) & df[y_col].isin(top_handsets_modalities) ]

    CrosstabResult=pd.crosstab(index=top_manufacturer_handset_data["Handset Manufacturer"],columns=top_manufacturer_handset_data["Handset Type"])
    CrosstabResult.plot.bar(rot=0)
    # # plt.figure(figsize=(12, 7))
    # # sns.barplot(data = df[df[x_col].isin(top_manufacturers_modalities) & df[y_col].isin(top_handsets_modalities) ], x=x_col, y=y_col)
    # plt.title(title, size=20)
    # plt.xticks(rotation=75, fontsize=14)
    # plt.yticks( fontsize=14)
    # plt.xlabel(xlabel, fontsize=16)
    # plt.ylabel(ylabel, fontsize=16)
    # plt.show()
  


def plot_hist(df:pd.DataFrame, column:str, color:str)->None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()

## function to plot the total data per decile
def decile_plot(df):
    # Sort users based on total duration
    sorted_users = df.sort_values(by='Dur. (ms)')

    # Divide users into deciles
    num_users = len(sorted_users)
    deciles = pd.qcut(sorted_users['Dur. (ms)'], q=10, labels=False, duplicates='drop')

    # assign users their resective decile class
    sorted_users['Decile'] = deciles

    # Identify the top five decile classes
    top_five_deciles = deciles.value_counts().nlargest(5).index
    # filter the top five deciles
    top_five_decile_data = sorted_users[sorted_users["Decile"].isin(top_five_deciles)]

    # Aggregate total data per decile class
    total_data_per_decile = top_five_decile_data .groupby('Decile')['total data (Bytes)'].sum().sort_values(ascending=False)

    # Display total data per decile class
    total_data_per_decile.plot(kind='bar', ylabel="total data")
    

# def plot_heatmap(df:pd.DataFrame, title:str, cbar=False)->None:
#     plt.figure(figsize=(12, 7))
#     sns.heatmap(df.corr(), annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
#     plt.title(title, size=18, fontweight='bold')
#     plt.show()

def plot_heatmap(df: pd.DataFrame, title: str, cbar=False) -> None:
    plt.figure(figsize=(20, 13))
    sns.heatmap(df.corr(), annot=True, fmt='.4f', linewidths=0.7, cbar=cbar, annot_kws={"fontsize": 12})
    plt.title(title, size=18, fontweight='bold')
    plt.show()
    








### function for count plot
def plot_count(df: pd.DataFrame, column: str, top: int) -> None:
    plt.figure(figsize=(12, 7))
    top_modalities = df[column].value_counts().nlargest(top).index
    sns.countplot(data=df[df[column].isin(top_modalities)], x=column, order=top_modalities)
    plt.xticks(rotation=75, fontsize=14)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()



def top_users_per_metrics(df:pd.DataFrame, userid:str,  metric:str,  top: int):
    return df.nlargest(top, metric)[[userid, metric]]





def plot_top_users_per_metrics(df: pd.DataFrame, userid: str, metric: str, title: str, top: int) -> None:
    top_users = df.nlargest(top, metric)[[userid, metric]]
    
    # Sort top_users DataFrame by the metric in descending order
    top_users = top_users.sort_values(by=metric, ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=top_users, x=userid, y=metric)
    plt.title(title, size=18)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(userid, fontsize=16)
    plt.ylabel(metric, fontsize=16)
    plt.show()

  
def normalizer(df, engagement_metrics):
  metric_data=df[engagement_metrics]
  minmax_scaler = MinMaxScaler()
  normalized_metrics= minmax_scaler.fit_transform(metric_data)
  return normalized_metrics

def kmeans(normalized_metrics, n_clusters):
    clustering= KMeans(n_clusters=n_clusters)
    clustering.fit(normalized_metrics)
    return clustering



def calculate_group_stats(df, group_column, metrics_columns, stat):
    """
    Calculate group statistics for mean, min,  max, or count.
    
    Args:
    - df: DataFrame containing the data
    - group_column: Name of the column to group by
    - metrics_column: Name of the column containing the metrics
    - stat: Statistic to calculate ('mean', 'min', or 'max')
    
    Returns:
    - grouped_stats: DataFrame containing the calculated group statistics
    """
    if stat not in ['mean', 'min', 'max', 'total']:
        raise ValueError("Invalid value for 'stat'. Please provide 'mean', 'min', 'max', or 'total")
    
    if stat == 'mean':
        grouped_stats = df.groupby(group_column)[metrics_columns].mean()
    elif stat == 'min':
        grouped_stats = df.groupby(group_column)[metrics_columns].min()
    elif stat == 'total':
        grouped_stats = df.groupby(group_column)[metrics_columns].sum()

    elif stat == 'max':
        grouped_stats = df.groupby(group_column)[metrics_columns].max()
    
    return grouped_stats



def plot_group_stats(non_normalized_stat, stat):
    for column in non_normalized_stat.columns:
        non_normalized_stat[column] = non_normalized_stat[column]/non_normalized_stat[column].max()
    non_normalized_stat.plot(kind="bar", figsize=(15,5), title=f"Metrics {stat} distribution per engagement group")
    

# Let keep the ten most frequent
def handset_type_handeling(df: pd.DataFrame, handset_type_column: str = 'name' ) -> pd.DataFrame:
    df["Modified Handset Type"]=df[handset_type_column]
    look_for= ["Apple", "Huawei", "Samsung", "Techno", "undefined"]
    for word in look_for:
        if word=="undefined":
            mode=df["Modified Handset Type"].mode()
            df.loc[df["Modified Handset Type"].str.contains(word, case=False), "Modified Handset Type"] = mode
        else:
            df.loc[df["Modified Handset Type"].str.contains(word, case=False), "Modified Handset Type"] = word
    return df