import os
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from StatsAnalytics import standardize, apply_PCA, extract_numerical, _exclude_outliers
from typing import Optional, List, Tuple, Iterable, Literal, Union


#------------------------------------------------------------------------------------------------------------
def create_cmap(
    color_list: np.ndarray
) -> ListedColormap:
    '''
    Creates a colormap for plots from a list of colors.

    Parameters:
    -----------
        color_list: (np.ndarray)
            An Nx3 array of colors in RGB format.

    Returns:
    --------
        A ListedColormap object to be used for plots.
    '''

    return  ListedColormap(
        colors=color_list,
        name='cell_stats_cmap'
    )

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def corr_matrix_plot(
    df: pd.DataFrame,
    numerical_features: Iterable[str],
    standardize_data: Optional[bool] = True,
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis',  
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

        remove_outliers: (Optional[bool], default=True)
            If true, outliers are removed from the dataframe.
        
        color_map: (Optional[Union[ListedColormap, str]], default='viridis')
            Either a pre-defined colormap or a user defined ListedColormap object.

        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''

    if standardize_data:
        df = standardize(df, numerical_features)

    if remove_outliers:
        df = _exclude_outliers(df)

    corr = df[numerical_features].corr()

    fig = plt.figure(figsize=(24, 12))
    ax = plt.subplot()

    sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=color_map,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 1}
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    features = [
        feature.replace('_', ' ').title() 
        for feature in numerical_features
    ]

    ax.set_xticklabels(
        features,
        rotation=45,
        horizontalalignment='right',
        fontsize=20)

    ax.set_title("Correlation Matrix", fontsize=30, pad=25)
    ax.set_yticklabels(features, fontsize=20)

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
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis', 
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

        remove_outliers: (Optional[bool], default=True)
            If true, outliers are removed from the dataframe.

        color_map: (Optional[Union[ListedColormap, str]], default='viridis')
            Either a pre-defined colormap or a user defined ListedColormap object.

        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    if standardize_data:
        df = standardize(df, numerical_features)
    
    if remove_outliers:
        df =  _exclude_outliers(df)

    pca_data, pca_loadings, explained_var = apply_PCA(df, numerical_features, 2, standardize_data)

    # PCA loadings Barplot
    fig = plt.figure(figsize=(8, 12))

    features = [
        feature.replace('_', ' ').title() 
        for feature in numerical_features
    ]

    ax1 = plt.subplot(211)
    sns.barplot(
        x=features,
        y=pca_loadings[0],
        ax=ax1
    )
    ax1.set_title("PC1 loadings", fontsize=28, pad=15)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax2 = plt.subplot(212)
    sns.barplot(
        x=features,
        y=pca_loadings[1],
        ax=ax2
    )
    ax2.set_title("PC2 loadings", fontsize=28, pad=15)
    ax2.tick_params(axis='x', labelrotation=75, labelsize=18)

    if save_dir:
        save_name = f"pca_loadings_barplots.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    if show:
        plt.show()
    else:
        plt.close()

    # PCA Scatterplot
    tissues = df['tissue'].unique()
    tissue_types = df['tissue_type'].unique()
    tissue_to_float = dict(zip(tissues, np.linspace(0, 1, len(tissues))))
    tissue_ids = [tissue_to_float[tissue] for tissue in df['tissue']]

    fig = plt.figure(figsize=(20, 12))
    ax = plt.subplot()

    scatter = ax.scatter(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        c=tissue_ids,
        cmap=color_map
    )

    ax.set_xlabel(
        f"PC1, explained variance = {round(explained_var[0], 3)}", 
        fontsize=22
    )
    ax.set_ylabel(
        f"PC2, explained variance = {round(explained_var[1], 3)}", 
        fontsize=22        
    )

    ax.set_title("Principal Components Scatterplot", fontsize=30, pad=15)

    ax.legend(
        handles=scatter.legend_elements()[0], 
        labels=[
            f"{name.replace('_', ' ').title()}: {t_type.replace('_', ' ')}" 
            for name, t_type in zip(tissues, tissue_types)],
        loc="upper right",
        fontsize=22,
        markerscale=3
    )

    if save_dir:
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
    remove_outliers: Optional[bool] = True, 
    color_map: Optional[Union[ListedColormap, str]] = 'viridis', 
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

        remove_outliers: (Optional[bool], default=True)
            If true, outliers are removed from the dataframe.

        units_of_measure: (Iterable[str])
            A collection of units of measure associated to the 
            features to plot.

        color_map: (Optional[Union[ListedColormap, str]], default='viridis')
            Either a pre-defined colormap or a user defined ListedColormap object.
        
        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''

    if remove_outliers:
        df = _exclude_outliers(df)

    tissues = df['tissue'].unique()
    tissue_types = df['tissue_type'].unique()

    if isinstance(color_map, str):
        colors = sns.color_palette(color_map, len(tissues))
    elif isinstance(color_map, ListedColormap):
        colors = color_map.colors

    fig = plt.figure(
        figsize=(len(features)*5, len(tissues)*4),
        constrained_layout=True
    )
    fig.suptitle("Morphological cell statistics comparison", fontsize=44)
    subfigs = fig.subfigures(len(tissues), 1)

    for i, tissue in enumerate(tissues):
        subfig = subfigs[i] 
        subfig.suptitle(
            f"{tissue.replace('_', ' ').title()}: {tissue_types[i].replace('_', ' ')}",
            fontsize=36
        )
        subplot_id = 1
        for j, column in enumerate(features):

            # Find the max on the x and y-axes to have the same axes length
            max_x = max(df[column])
            max_x = max_x + 0.1*max_x

            # Get the unit of measure
            unit_of_measure = units_of_measure[j]

            # Get the current axis object
            # ax = fig.add_subplot(len(tissues), len(features), subplot_id)
            ax = subfig.add_subplot(1, len(features), subplot_id)
            subplot_id += 1

            # Subset the data for the current tissue
            tissue_df = df[df['tissue'] == tissue]

            # Map kernel density plot onto the axes, using shading and color
            sns.kdeplot(
                data=tissue_df, 
                x=column, 
                fill=True, 
                color=colors[i], 
                alpha=0.66,
                ax=ax, 
                clip=(0.0, max_x)
            )

            # Map rugplot to the axes, using height to adjust the size of the ticks
            sns.rugplot(
                data=tissue_df, 
                x=column, 
                height=0.125, 
                color=colors[i], 
                ax=ax
            )

            if unit_of_measure:
                xlab = column.replace("_", " ").title() + f" ({unit_of_measure})"
            else:
                xlab = column.replace("_", " ").title()

            # Set x-axis stuff
            if i == (len(tissues)-1):
                ax.set_xlabel(xlab, fontsize=22)
            else:
                ax.set_xlabel("")
                ax.set_xticks([])
                ax.set_xlim([0, max_x])
            
            # Set y-axis stuff
            ax.set_ylabel("")
            ax.set_yticks([])
            if j == 0:
                # ax.set_title(f'{tissue.title()}: {tissue_types[i]}', fontsize=24)
                ax.set_ylabel('Density', fontsize=22)
            
            # Remove the square around the plot
            sns.despine(left=False, bottom=False, top=True, right=True)

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
def num_neighbors_barplots(
    df: pd.DataFrame, 
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis',
    save_dir: Optional[str] = None, 
    show: Optional[bool] = False 
) -> None:
    '''
    Make a barplot of the number of neighbors for the different samples,

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.

        remove_outliers: (Optional[bool], default=True)
            If true, outliers are removed from the dataframe.

        color_map: (Optional[Union[ListedColormap, str]], default='viridis')
            Either a pre-defined colormap or a user defined ListedColormap object.
        
        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''
    
    if remove_outliers:
        df = _exclude_outliers(df)

    tissues = df['tissue'].unique()
    tissue_types = df['tissue_type'].unique()

    if isinstance(color_map, str):
        colors = sns.color_palette(color_map, len(tissues))
    elif isinstance(color_map, ListedColormap):
        colors = color_map.colors

    fig = plt.figure(figsize=(len(tissues)*7, 5))
    subplot_id = 1
    for i, tissue in enumerate(tissues):
        # Get the current axis object
        ax = fig.add_subplot(1, len(tissues), subplot_id)
        subplot_id += 1

        # Subset the data for the current tissue
        data = df[df['tissue'] == tissue]['num_neighbors']

        # Count the frequency of each unique value
        unique_values, counts = np.unique(data, return_counts=True)
        
        # Make unique_values and counts cover all the span
        val_counts_dict = dict(zip(unique_values, counts))
        for j in range(1, max(unique_values)):
            if j not in unique_values:
                val_counts_dict[j] = 0

        # Create a bar plot using Seaborn
        sns.barplot(
            x=list(val_counts_dict.keys()), 
            y=list(val_counts_dict.values()), 
            color=colors[i], 
            ax=ax
        )

        # Set title and axes labels
        ax.set_title(f'{tissue.title()}: {tissue_types[i]}', fontsize=20)
        max_x = max(data) + 1
        xlab = 'num_neighbors'.replace("_", " ").title()
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_xlim([0, max_x])
        ax.set_xticks(list(val_counts_dict.keys()))
        ax.set_xticklabels(list(val_counts_dict.keys()))
        ax.set_ylabel('Counts', fontsize=16)

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)

    fig.suptitle("Number of neighbors comparison", fontsize=24)
    fig.subplots_adjust(top=0.8)

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



#------------------------------------------------------------------------------------------------------------
def lewis_law_plots(
    df: pd.DataFrame, 
    feature: Literal['volume', 'area'] = 'volume',
    fit_degrees: Optional[Iterable[int]] = [1,2],
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis',
    save_dir: Optional[str] = None,     
    show: Optional[bool] = False 
) -> None:
    '''
    Make a plot of to empirically check the 3D Lewis Law across different tissues.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.
        
        feature: (Literal['volume', 'area'], default='volume')
            A string to specify whether to plot the Lewis Law for area or volume.
        
        fit_degrees: (Optional[Iterable[int]], default=[1,2])
            The degree of the polynomial fitted on the data in the plots.

        remove_outliers: (Optional[bool], default=True)
            If true, outliers are removed from the dataframe.

        color_map: (Optional[Union[ListedColormap, str]], default='viridis')
            Either a pre-defined colormap or a user defined ListedColormap object.
        
        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    '''

    if remove_outliers:
        df = _exclude_outliers(df)   

    tissues = df['tissue'].unique()
    tissue_types = df['tissue_type'].unique()
    
    if isinstance(color_map, str):
        colors = sns.color_palette(color_map, len(tissues))
    elif isinstance(color_map, ListedColormap):
        colors = color_map.colors

    fig = plt.figure(
        figsize=(len(tissues)*6, 12),
        constrained_layout=True
    )
    fig.suptitle(f"Lewis' Law for {feature.replace('_', ' ')}", fontsize=36)
    subplot_id = 1
    for i, tissue in enumerate(tissues):
        # Get the current axis object
        if len(tissues) <= 3:
            ax = fig.add_subplot(1, len(tissues), subplot_id)
        else:
            ax = fig.add_subplot(2, int(np.ceil(len(tissues)/2)), subplot_id)
        subplot_id += 1

        # Subset the data for the current tissue
        tissue_df = df[df['tissue'] == tissue]

        # Compute global volume average
        global_avg = tissue_df[feature].mean()

        # Compute average volume for each value of n
        num_neighbors_values = np.asarray(tissue_df['num_neighbors'].unique())
        local_avgs, local_sds = {}, {}
        for n in num_neighbors_values:
            #subset the df
            data = tissue_df[tissue_df['num_neighbors'] == n][feature]
            # compute the average for this n
            local_avgs[n] = data.mean()/global_avg
            local_sds[n] = (data/global_avg).std()/np.sqrt(len(data))
        # Sort dict by key
        local_avgs = dict(sorted(local_avgs.items()))
        local_sds = dict(sorted(local_sds.items()))
        std_devs = list(local_sds.values())

        # Compute fitted lines
        assert len(fit_degrees) == 2, 'Cannot fit more than 2 different degrees of polynomials.'
        x = np.asarray(list(local_avgs.keys()), dtype=np.int64)
        y = list(local_avgs.values())
        coeff_sets = [np.polyfit(x, y, degree) for degree in fit_degrees]
        polylines = [np.poly1d(coeff_set) for coeff_set in coeff_sets] 
        x_fit = np.linspace(min(x), max(x), max(x)-min(x)+1, dtype=np.int32) 
        y_linear, y_quadratic = (polyline(x_fit) for polyline in polylines)

        # Plot the values and the fitted lines
        ax.errorbar(x, y, yerr=std_devs, fmt='o', color=colors[i], ecolor='grey', capsize=4)
        linear, = ax.plot(x_fit, y_linear, color='red', linestyle='--', label='Linear fit')   
        quadratic, = ax.plot(x_fit, y_quadratic, color='green', linestyle='-.', label='Quadratic fit')

        # Set title and axes labels
        ax.set_title(f'{tissue.title()}: {tissue_types[i]}', fontsize=28)
        ax.set_xlabel(r'Number of neighbors $(n)$', fontsize=24)
        if feature == 'volume':
            ax.set_ylabel(r'$\bar{V}_n / \bar{V}$', fontsize=24)
        elif feature == 'surface_area':
            ax.set_ylabel(r'$\bar{A}_n / \bar{A}$', fontsize=24)
        ax.set_xticks(x_fit)
        ax.legend(handles=[linear, quadratic], loc='lower right', fontsize=18)

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)
    
    # plt.subplots_adjust(top=0.9)

    # Save the current plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"lewis_law_{feature}_plots.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def violin_plots(
    df: pd.DataFrame, 
    tissue: str,
    features: Iterable[str],
    units_of_measure: Iterable[str],
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis',
    save_dir: Optional[str] = None, 
    show: Optional[bool] = False 
) -> None:
    """
    Generate violin plots to .

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.
        
        tissue: (str)
            Specify the tissue to generate the plot for.

        features: (Iterable[str])
            A list of numerical features to plot.

        units_of_measure: (Iterable[str])
            The of units of measure associated to the features to plot.

        remove_outliers: (Optional[bool], default=True)
            If true, outliers are removed from the dataframe.

        color_map: (Optional[Union[ListedColormap, str]], default='viridis')
            Either a pre-defined colormap or a user defined ListedColormap object.
        
        save_dir: (Optional[str], default=None)
            The path to the directory in which the plot is saved.
        
        show: (Optional[bool], default=False)
            If `True` show the plot when calling the function.
    """

    if remove_outliers:
        df = _exclude_outliers(df)  

    sns.set(style="whitegrid")

    tissue_df = df[df['tissue' == tissue]]
    num_plots = len(features)
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 10), sharey=False)
    fig.suptitle(f'Morphological statistics distribution in {tissue} sample', fontsize=30)
    fig.tight_layout(pad=6)

    if isinstance(color_map, str):
        colors = sns.color_palette(color_map, num_plots)
    elif isinstance(color_map, ListedColormap):
        colors = color_map.colors

    for i, feature in enumerate(features):
        data = tissue_df.loc[[feature]]
        unit_of_measure = units_of_measure[i]
        sns.violinplot(data=data, orient="v", cut=0, inner="quartile", ax=axes[i], color=colors[i])
        sns.stripplot(data=data, color=".3", size=4, jitter=True, ax=axes[i])

        axes[i].xaxis.set_tick_params(labelbottom=False)
        if unit_of_measure:
            xlab = f"{feature.replace('_', ' ').title()} ({units_of_measure[i]})"
        else:
            xlab = f"{feature.replace('_', ' ').title()}"
        axes[i].set(xlabel=xlab)
        axes[i].set_title("", pad=-15)

    sns.despine(left=True)

    # Save the current plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"{tissue}_violin_plot.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()
