import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from StatsAnalytics import standardize, apply_PCA, extract_numerical
from typing import Optional, List, Tuple, Iterable



#------------------------------------------------------------------------------------------------------------
def _exclude_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Return a copy of the input dataframe without the records marked as outliers.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe. It must have an `is_outlier` boolean column,
            which is `True` if the correspondent record should be removed.

    Returns:
    --------
        out_df: (pd.DataFrame)
            The input dataframe without outliers.  
    '''

    out_df = df[~df['is_outlier']]
    return out_df

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def corr_matrix_plot(
    df: pd.DataFrame,
    numerical_features: Iterable[str],
    standardize_data: Optional[bool] = True,
    save_dir: Optional[str] = None, 
    show: Optional[bool] = False
) -> None:
    '''
    Plot the correlation matrix of the numerical features of the standardized dataframe.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.

        numerical_features: (Iterable[str])
            A list of numerical features contained in the input dataframe.

        standardize_data: (Optional[bool], deafult=True)
            If true standardize the input data before plotting.

        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''

    if standardize_data:
        df = standardize(df, numerical_features)

    corr = df[numerical_features].corr()

    fig, ax = plt.figure(figsize=(24, 12))

    sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap='viridis',
        square=True,
        ax=ax,
        cbar_kws={"shrink": .5}
    )

    features = [
        feature.title() 
        for feature in numerical_features
    ]

    ax.set_xticklabels(
        features,
        rotation=45,
        horizontalalignment='right',
        fontsize=12)

    ax.set_title("Correlation Matrix", fontsize=24)
    ax.set_yticklabels(features, fontsize=12)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"correlation_matrix_heatmap.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    if show:
        plt.show()
    else:
        plt.close()
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def pca_plots(
    df: pd.DataFrame,
    numerical_features: Iterable[str],
    standardize_data: Optional[bool] = True,
    save_dir: Optional[str] = None, 
    show: Optional[bool] = False
) -> None:
    '''
    Plot PCA scatterplot and loadings barplots. 

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.

        numerical_features: (Iterable[str])
            A list of numerical features contained in the input dataframe.

        standardize_data: (Optional[bool], deafult=True)
            If true standardize the input data before plotting.

        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    pca_data, pca_loadings, explained_var = apply_PCA(df, numerical_features, 2, standardize_data)

    # PCA loadings Barplot
    plt.rcParams['figure.figsize'] = [5, 9]
    fig = plt.Figure()

    features = [feature.title() for feature in numerical_features]

    ax1 = plt.subplot(211)
    sns.barplot(
        x=features,
        y=pca_loadings[0],
        ax=ax1
    )
    ax1.set_title("PC1 loadings", fontsize=22)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax2 = plt.subplot(212)
    sns.barplot(
        x=features,
        y=pca_loadings[1],
        ax=ax2
    )
    ax2.set_title("PC2 loadings", fontsize=22)
    ax2.tick_params(axis='x', labelrotation=60, labelsize=18)

    save_name = f"pca_loadings_barplots.jpg"
    plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    if show:
        plt.show()
    else:
        plt.close()

    # PCA Scatterplot
    tissues = df['tissue'].unique()
    tissue_to_float = dict(zip(tissues, np.linspace(0, 1, len(tissues))))
    tissue_ids = [tissue_to_float[tissue] for tissue in tissues]

    fig, ax = plt.figure(figsize=(20, 8))

    scatter = ax.scatter(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        c=tissue_ids,
        cmap='viridis'
    )

    ax.set_xlabel(f"PC1, explained variance = {round(explained_var[0], 3)}", fontsize=18)
    ax.set_ylabel(f"PC2, explained variance = {round(explained_var[1], 3)}", fontsize=18)

    ax.set_title("Principal Components Scatterplot", fontsize=28)

    ax.legend(handles=scatter.legend_elements()[0], 
        labels=list([f"Tissue: {name}" for name in tissues]),
        loc="upper right",
        fontsize=22
    )

    save_name = f"pca_scatterplot.jpg"
    plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    if show:
        plt.show()
    else:
        plt.close()

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def features_grid_kdplots(
    df: pd.DataFrame,
    features: Iterable[str],
    units_of_measure: Iterable[str], 
    save_dir: Optional[str] = None, 
    show: Optional[bool] = False 
) -> None:
    '''
    Make a grid of kernel density estimation plots of the chosen features. 

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.

        features: (Iterable[str])
            A list of numerical features to plot.

        units_of_measure: (Iterable[str])
            A collection of units of measure associated to the 
            features to plot.
        
        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''

    tissues = df['tissue'].unique()
    tissue_types = df['tissue_type'].unique()

    colors = sns.color_palette('viridis', len(tissues))

    fig = plt.figure(figsize=(24, 12))
    subplot_id = 1

    for i, tissue in enumerate(tissues):
        for j, column in enumerate(features):

            # Find the max on the x and y-axes to have the same axes length
            max_x = max(df[column])
            max_x = max_x + 0.1*max_x

            # Get the unit of measure
            unit_of_measure = units_of_measure[j]

            # Get the current axis object
            ax = fig.add_subplot(len(tissues), len(features), subplot_id)
            subplot_id += 1

            # Subset the data for the current tissue
            tissue_df = df[df['tissue'] == tissue]

            # Map kernel density plot onto the axes, using shading and color
            sns.kdeplot(data=tissue_df, x=column, fill=True, color=colors[i], ax=ax, clip=(0.0, max_x))

            # Map rugplot to the axes, using height to adjust the size of the ticks
            sns.rugplot(data=tissue_df, x=column, height=0.125, color=colors[i], ax=ax)

            # Set title and axes labels
            if j == 0:
                ax.set_title(f'{tissue.title()}: {tissue_types[i]}', fontsize=20)

            if unit_of_measure:
                xlab = column.replace("_", " ").title() + f" ({unit_of_measure})"
            else:
                xlab = column.replace("_", " ").title()
            ax.set_xlabel(xlab, fontsize=20)
            ax.set_ylabel('Density', fontsize=16)

            # Remove y-axis ticks and set x and y-axis limits for the current plot
            ax.set_yticks([])
            ax.set_xlim([0, max_x])

            # Remove the square around the plot
            sns.despine(left=False, bottom=False, top=True, right=True)

            # Remove x-axis from the first 2 plots
            if i < len(tissues)-1:
                ax.set_xticks([])
                ax.set_xlabel("")
                ax.spines['bottom'].set_visible(False)

    fig.suptitle("Morphological cell statistics comparison", fontsize=24)

    # Save the current plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"grid_kdeplots.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def num_neighbors_barplot(
    df: pd.DataFrame, 
    save_dir: Optional[str] = None, 
    show: Optional[bool] = False 
) -> None:
    '''
    Make a barplot of the number of neighbors for the different samples,

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.
        
        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''
    
    tissues = df['tissue'].unique()
    tissue_types = df['tissue_type'].unique()
    colors = sns.color_palette('viridis', len(tissues))

    # Find the max on the x-axis to have the same axes length
    max_x = max(df['num_neighbors']) + 1

    fig = plt.figure(figsize=(20, 5))
    subplot_id = 1
    for i, tissue in enumerate(tissues):
        # Get the current axis object
        ax = fig.add_subplot(1, len(tissues), subplot_id)
        subplot_id += 1

        # Subset the data for the current tissue
        data = df[df['tissue'] == tissue]['num_neighbors']

        # Count the frequency of each unique value
        unique_values, counts = np.unique(data, return_counts=True)

        # Create a bar plot using Seaborn
        sns.barplot(x=unique_values, y=counts, color=colors[i], ax=ax)

        # Set title and axes labels
        ax.set_title(f'{tissue.title()}: {tissue_types[i]}', fontsize=20)

        xlab = 'num_neighbors'.replace("_", " ").title()
        ax.set_xlabel(np.arange(1, max(xlab)), fontsize=20)
        ax.set_ylabel('Counts', fontsize=16)
        ax.set_xlim([0, max_x])

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)

    fig.suptitle("Number of neighbors comparison", fontsize=24)

    # Save the current plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"num_neighbors_barplot.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()

#------------------------------------------------------------------------------------------------------------