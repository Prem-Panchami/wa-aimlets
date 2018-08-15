# This code originated from https://github.com/Prem-Panchami/wa-aimlets
# and is being shared under Creative Commons (4.0) Attribution-Non-Commercial
# license. Please see https://github.com/Prem-Panchami/wa-aimlets/blob/master/LICENSE.md
#
# Retain the above attribution in all copies and derivatives. Thanks!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df, method='pearson', min_periods=1, figsize=(10, 8)):
    """
    Plot the bottom half below the diagonal of the pairwise correlation between numeric
    columns, excluding NA/null values
    
    df : Dataframe with the data
    
    method : str; Default='pearson'. Allowed={‘pearson’, ‘kendall’, ‘spearman’}
             - pearson : standard correlation coefficient
             - kendall : Kendall Tau correlation coefficient
             - spearman : Spearman rank correlation
    
    min_periods : int; Default=1
                  Minimum number of observations required per pair of columns to have a 
                  valid result. Currently only available for pearson and spearman correlation.
    
    figsize : Tuple of ints - (width, height); Default=(10,8)
    """
    corr = df.corr(method=method, min_periods=min_periods)

    plt.figure(figsize=figsize)
    cmap = cmap=sns.diverging_palette(250, 5,n=9, as_cmap=True)
    sns.heatmap(np.tril(corr), 
                vmin=-1, 
                vmax=1,
                center=0,
                cmap=cmap, 
                annot=True,
                cbar=True,
                square=True,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.show()