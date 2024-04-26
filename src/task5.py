import os
import sys
module_path = os.path.abspath(os.path.join('../')) 
sys.path.append(module_path)

from scipy.spatial.distance import cdist # type: ignore
from src.EDA import normalizer
import pandas as pd

# function to assign engagement or experience score to the user
def score(df: pd.DataFrame, metrics: list, metrics_category: str,
        centroids: pd.DataFrame, worst_group: int ) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        del df["Unnamed: 0"]

    if "Unnamed: 0" in centroids.columns:
        del centroids["Unnamed: 0"]
    #  get the worst group coordinates
    centroid= centroids.iloc[worst_group,:].to_numpy()
    centroid = centroid.reshape(1, -1)

    # normalize metrics

    normalized_metrics=normalizer(df,metrics)

    # compute distance
    distances = cdist(normalized_metrics, centroid , metric='euclidean')

    df[f"{metrics_category} score"]= distances
    return df
    
    