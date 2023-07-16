import os
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats
from StatsAnalytics import standardize, apply_PCA, extract_numerical, _exclude_outliers, _get_lewis_law_2D_stats, _get_aboav_law_2D_stats, _get_area_CV
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

    fig = plt.figure(figsize=(2*len(numerical_features), 1.5*len(numerical_features)))
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
    fig = plt.figure(figsize=(6, 10))

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

    fig = plt.figure(figsize=(16, 10))
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
        fontsize=18,
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
    y_lims: Iterable[float],
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

        units_of_measure: (Iterable[str])
            A collection of units of measure associated to the 
            features to plot.
        
        y_lims: (Iterable[float])
            A collection of floats that determines the y-axis upper limits
            for all the features (to be set empirically). 

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
        figsize=(len(features)*7, len(tissues)*5.5),
        constrained_layout=True
    )
    fig.suptitle("Morphological cell statistics comparison", fontsize=60)
    subfigs = fig.subfigures(len(tissues), 1)

    for i, tissue in enumerate(tissues):
        subfig = subfigs[i] 
        subfig.suptitle(
            f"{tissue.replace('_', ' ').title()}: {tissue_types[i].replace('_', ' ')}",
            fontsize=44
        )
        subplot_id = 1
        for j, column in enumerate(features):

            # Find the max on the x and y-axes to have the same axes length
            max_x = max(df[column])
            min_x = min(df[column])
            lim_x = [min_x - 0.2*min_x, max_x + 0.1*max_x]

            # Get the unit of measure
            unit_of_measure = units_of_measure[j]

            # Get the current axis object
            # ax = fig.add_subplot(len(tissues), len(features), subplot_id)
            ax = subfig.add_subplot(1, len(features), subplot_id)
            subplot_id += 1

            # Subset the data for the current tissue
            tissue_df = df[df['tissue'] == tissue]

            if column != 'num_neighbors':
                # Map kernel density plot onto the axes, using shading and color
                sns.kdeplot(
                    data=tissue_df, 
                    x=column, 
                    fill=True, 
                    color=colors[i], 
                    alpha=0.66,
                    ax=ax, 
                    # clip=lim_x
                )

                # Map rugplot to the axes, using height to adjust the size of the ticks
                sns.rugplot(
                    data=tissue_df, 
                    x=column, 
                    height=0.125, 
                    color=colors[i], 
                    ax=ax
                )
                # format the x-axis ticks in scientific notation
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                x_offset = ax.xaxis.get_offset_text()
                x_offset.set_size(18)

            else:
                # Count the frequency of each unique value
                unique_values, counts = np.unique(tissue_df[column], return_counts=True)
                
                # Make unique_values and counts cover all the span
                val_counts_dict = dict(zip(unique_values, counts/len(tissue_df)))
                for k in range(1, max(unique_values)):
                    if k not in unique_values:
                        val_counts_dict[k] = 0

                # Create a bar plot using Seaborn
                sns.barplot(
                    x=list(val_counts_dict.keys()), 
                    y=list(val_counts_dict.values()), 
                    color=colors[i], 
                    ax=ax
                )
                ax.set_xticks(np.arange(min_x-1, max_x, 1))

            if unit_of_measure:
                xlab = column.replace("_", " ").title() + f" ({unit_of_measure})"
            else:
                xlab = column.replace("_", " ").title()

            # Set x-axis stuff
            ax.set_xlim(lim_x) if column != 'num_neighbors' else ax.set_xlim(lim_x[0]-1, lim_x[1])
            if i == (len(tissues)-1):
                ax.set_xlabel(xlab, fontsize=32)
            else:
                ax.set_xlabel("")
                ax.set_xticks([])
            # set the x-axis tick font size
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(20)  

            # Set y-axis stuff
            ax.set_ylabel("")
            # ax.set_yticks([])
            ax.set_ylim([0, y_lims[j]])
            if j == 0:
                ax.set_ylabel('Density', fontsize=32)
            # set the y-axis tick font size
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(20)  
            # format the y-axis ticks in scientific notation
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            y_offset = ax.yaxis.get_offset_text()
            y_offset.set_size(18)
            
            # Remove the square around the plot
            sns.despine(left=False, bottom=False, top=True, right=True)
            
        
        # Set common y-axis for a certain column

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
    version: Literal['2D', '3D'],
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
        if version == '3D':
            data = df[df['tissue'] == tissue]['num_neighbors']
        elif version == '2D':
            tissue_df = df[df['tissue'] == tissue]
            data = []
            for _, row in tissue_df.iterrows():
                if not row['exclude_cell'] and len(row['num_neighbors_2D_principal']):
                    data = data + row['num_neighbors_2D_principal']

        # Count the frequency of each unique value
        unique_values, counts = np.unique(data, return_counts=True)
        freqs = counts / len(data)
        avg_num_neighbors = np.mean(data)
        
        # Make unique_values and counts cover all the span
        val_freqs_dict = dict(zip(unique_values, freqs))
        val_counts_dict = dict(zip(unique_values, counts))
        for j in range(1, max(unique_values)):
            if j not in unique_values:
                val_freqs_dict[j] = 0
                val_counts_dict[j] = 0

        # Create a bar plot using Seaborn
        sns.barplot(
            x=list(val_freqs_dict.keys()), 
            y=list(val_freqs_dict.values()), 
            color=colors[i], 
            ax=ax
        )

        # Print counts on top of bars
        for num, count in val_counts_dict.items():
            ax.annotate(
                count, xy=(num-1, val_freqs_dict[num]), ha='center', va='bottom'
            )

        # Set title and axes labels
        ax.set_title(f'{tissue.replace("_", " ").title()}: {tissue_types[i]}', fontsize=20)
        max_x = max(data) + 1
        xlab = 'num_neighbors'.replace("_", " ").title()
        ax.set_xlabel(xlab, fontsize=20)
        ax.set_xlim([-1, max_x])
        ax.set_ylim([0, 0.55])
        # ax.set_xticks(np.arange(0, max_x))
        # ax.set_xticklabels(np.arange(1, max_x))
        ax.set_ylabel('Frequency', fontsize=16)
        ax.text(
            x=6, 
            y=0.3, 
            s=(
                f'Average: {round(avg_num_neighbors, 2)}'
                '\n'
                f'Tot cells in slices: {len(data)}'
            ), 
            style='italic', 
            fontsize=14
        )

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)

    fig.suptitle(f"Number of {version} neighbors comparison", fontsize=24)
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

    min_x, max_x = min(df['num_neighbors'])-1, max(df['num_neighbors'])+1
    max_y = 3.0

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
        fits = [np.polyfit(x, y, degree, cov=True) for degree in fit_degrees]
        coeff_sets = [fit[0] for fit in fits]
        std_err_sets = [np.sqrt(np.diag(fit[1])) for fit in fits] 
        confint_width_sets = [
            [
                stats.t(df=len(x)-len(coeff_set)).ppf(0.975)*std_err
                for coeff, std_err in zip(coeff_set, std_errs)
            ]
            for coeff_set, std_errs in zip(coeff_sets, std_err_sets)
        ]
        polylines = [np.poly1d(coeff_set) for coeff_set in coeff_sets] 
        x_fit = np.arange(min_x, max_x, dtype=np.int32) 
        y_linear, y_quadratic = (polyline(x_fit) for polyline in polylines)
        y_th_linear = (x_fit - 2) / 4
        y_th_quadratic = (x_fit / 6)**2

        # Plot the values and the fitted lines
        ax.errorbar(x, y, yerr=std_devs, fmt='o', color=colors[i], ecolor='grey', capsize=8, markersize=10)
        coeffs = [round(coeff, 2) for coeff in coeff_sets[0]]
        ci_widths = [round(ci_width, 2) for ci_width in confint_width_sets[0]]
        linear, = ax.plot(
            x_fit, y_linear, 
            color='red', linestyle='--', 
            label=f'Linear fit, coeff: a={coeffs[1]}\u00B1{ci_widths[1]}, b={coeffs[0]}\u00B1{ci_widths[0]}'
        )
        coeffs = [round(coeff, 2) for coeff in coeff_sets[1]]  
        ci_widths = [round(ci_width, 2) for ci_width in confint_width_sets[1]]
        quadratic, = ax.plot(
            x_fit, y_quadratic, 
            color='green', linestyle='-.', 
            label=f'Quadratic fit, coeff: a={coeffs[2]}\u00B1{ci_widths[2]}, b={coeffs[1]}\u00B1{ci_widths[1]}, c={coeffs[0]}\u00B1{ci_widths[0]}'
        )
        th_linear, = ax.plot(
            x_fit, y_th_linear,
            color='orange', linestyle='--',
            label=f"Theoretical linear fit"
        )
        th_quadratic, = ax.plot(
            x_fit, y_th_quadratic,
            color='blue', linestyle='-.',
            label=f"Theoretical quadratic fit"
        ) 

        # Set title and axes labels
        ax.set_title(f'{tissue.replace("_", " ").title()}: {tissue_types[i].replace("_", " ")}', fontsize=28)
        ax.set_xlabel(r'Number of neighbors $(n)$', fontsize=24)
        if feature == 'volume':
            ax.set_ylabel(r'$\bar{V}_n / \bar{V}$', fontsize=24)
        elif feature == 'surface_area':
            ax.set_ylabel(r'$\bar{A}_n / \bar{A}$', fontsize=24)
        ax.set_xticks(x_fit)
        ax.legend(
            handles=[linear, th_linear, quadratic, th_quadratic], 
            loc='upper left', 
            fontsize=16
        )
        ax.text(
            x=19, 
            y=0.15, 
            s=(
                r'Fitted line: $y=a+bx+c{x}^2$'
                '\n'
                r'Theoretical linear fit: $\bar{A}_n / \bar{A} = \frac{(n-2)}{4}$'
                '\n'
                r'Theoretical quadratic fit: $\bar{A}_n / \bar{A} \sim\ (\frac{n}{6})^2$'
            ), 
            style='italic', 
            fontsize=12
        )


        # # Set axes limits
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([0, max_y])

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)
    
    plt.subplots_adjust(top=0.8)

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
def lewis_law_2D_plots(
    df: pd.DataFrame, 
    fit_degrees: Optional[Iterable[int]] = [1,2],
    version: Literal['standard', 'principal'] = 'standard',
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis',
    save_dir: Optional[str] = None,     
    show: Optional[bool] = False 
) -> None:
    '''
    Make a plot of to empirically check the 2D Lewis Law across different slices of tissues.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.
        
        fit_degrees: (Optional[Iterable[int]], default=[1,2])
            The degree of the polynomial fitted on the data in the plots.
            If `None`, no polynomial fitting is done.

        version: (Literal['standard', 'principal'], default='standard')
            If 'standard', 2D statistics collected along coordinate axis
            are considered. If 'principal', 2D stats collected along cells'
            principal axes are considered instead.

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

    lewis_law_stats = _get_lewis_law_2D_stats(
        df, principal_axis=(version=='principal')
    )

    min_x, max_x = 2, 16
    max_y = 4.0

    fig = plt.figure(
        figsize=(len(tissues)*6, 12),
        constrained_layout=True
    )
    fig.suptitle(f"Lewis' Law for 2D cell area collected along {version} axes", fontsize=36)
    subplot_id = 1
    for i, tissue in enumerate(tissues):
        # Get the current axis object
        if len(tissues) <= 3:
            ax = fig.add_subplot(1, len(tissues), subplot_id)
        else:
            ax = fig.add_subplot(2, int(np.ceil(len(tissues)/2)), subplot_id)
        subplot_id += 1

        data = lewis_law_stats[tissue]
        x = np.asarray(list(data.keys()), dtype=np.int64)
        y_val = list([val[0] for val in data.values()])
        y_err = list([val[1] for val in data.values()])
        x_fit = np.arange(min_x, max_x, dtype=np.int32) 
        if fit_degrees:
            fits = [np.polyfit(x, y_val, degree, cov=True) for degree in fit_degrees]
            coeff_sets = [fit[0] for fit in fits]
            std_err_sets = [np.sqrt(np.diag(fit[1])) for fit in fits] 
            confint_width_sets = [
                [
                    stats.t(df=len(x)-len(coeff_set)).ppf(0.975)*std_err
                    for coeff, std_err in zip(coeff_set, std_errs)
                ]
                for coeff_set, std_errs in zip(coeff_sets, std_err_sets)
            ]
            polylines = [np.poly1d(coeff_set) for coeff_set in coeff_sets] 
            y_linear, y_quadratic = (polyline(x_fit) for polyline in polylines)
        y_th_linear = (x_fit - 2) / 4
        y_th_quadratic = (x_fit / 6)**2

        # Plot the values and the fitted lines
        ax.errorbar(x, y_val, yerr=y_err, fmt='o', color=colors[i], ecolor='grey', capsize=8, markersize=10)
        if fit_degrees:
            coeffs = [round(coeff, 2) for coeff in coeff_sets[0]]
            ci_widths = [round(ci_width, 2) for ci_width in confint_width_sets[0]]
            linear, = ax.plot(
                x_fit, y_linear, 
                color='red', linestyle='--', 
                label=f'Linear fit, coeff: a={coeffs[1]}\u00B1{ci_widths[1]}, b={coeffs[0]}\u00B1{ci_widths[0]}'
            )
            coeffs = [round(coeff, 2) for coeff in coeff_sets[1]]  
            ci_widths = [round(ci_width, 2) for ci_width in confint_width_sets[1]]
            quadratic, = ax.plot(
                x_fit, y_quadratic, 
                color='green', linestyle='-.', 
                label=f'Quadratic fit, coeff: a={coeffs[2]}\u00B1{ci_widths[2]}, b={coeffs[1]}\u00B1{ci_widths[1]}, c={coeffs[0]}\u00B1{ci_widths[0]}'
            )
        th_linear, = ax.plot(
            x_fit, y_th_linear,
            color='orange', linestyle='--',
            label=f"Theoretical linear fit"
        )
        th_quadratic, = ax.plot(
            x_fit, y_th_quadratic,
            color='blue', linestyle='-.',
            label=f"Theoretical quadratic fit"
        ) 

        # Set title and axes labels
        ax.set_title(f'{tissue.replace("_", " ").title()}: {tissue_types[i].replace("_", " ")}', fontsize=28)
        ax.set_xlabel(r'Number of neighbors $(n)$', fontsize=24)
        ax.set_ylabel(r'$\bar{A}_n / \bar{A}$', fontsize=24)
        ax.set_xticks(x_fit)
        if fit_degrees:
            ax.legend(
                handles=[linear, th_linear, quadratic, th_quadratic], 
                loc='upper left', 
                fontsize=16
            )
            ax.text(
                x=11, 
                y=0.25, 
                s=(
                    r'Fitted line: $y=a+bx+c{x}^2$'
                    '\n'
                    r'Theoretical linear fit: $\bar{A}_n / \bar{A} = \frac{(n-2)}{4}$'
                    '\n'
                    r'Theoretical quadratic fit: $\bar{A}_n / \bar{A} \sim\ (\frac{n}{6})^2$'
                ), 
                style='italic', 
                fontsize=12
            )
        else:
            ax.legend(
                handles=[th_linear, th_quadratic], 
                loc='upper left', 
                fontsize=16
            )
            ax.text(
                x=11, 
                y=0.25, 
                s=(
                    r'Theoretical linear fit: $\bar{A}_n / \bar{A} = \frac{(n-2)}{4}$'
                    '\n'
                    r'Theoretical quadratic fit: $\bar{A}_n / \bar{A} \sim\ (\frac{n}{6})^2$'
                ), 
                style='italic', 
                fontsize=12
            )


        # Set axes limits
        # ax.set_xlim([min_x, max_x])
        ax.set_ylim([0, max_y])

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)
    
    # plt.subplots_adjust(top=0.8)

    # Save the current plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"lewis_law_2D_{version}_plots.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def aboav_wearie_2D_plots(
    df: pd.DataFrame,
    version: Literal['standard', 'principal'] = 'standard',
    fitted: Optional[bool] = True,
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis',
    save_dir: Optional[str] = None,     
    show: Optional[bool] = False 
) -> None:
    '''
    Make a plot of to empirically check the 2D Lewis Law across different slices of tissues.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.
        
        version: (Literal['standard', 'principal'], default='standard')
            If 'standard', 2D statistics collected along coordinate axis
            are considered. If 'principal', 2D stats collected along cells'
            principal axes are considered instead.
        
        fitted: (Optional[bool], default=True)
            If true fit a line through the data and plot it.

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

    aboav_law_stats = _get_aboav_law_2D_stats(
        df, principal_axis=(version=='principal')
    )

    if version == 'standard':
        min_x, max_x = 2, 16
        min_y, max_y = 10, 85
    else:
        min_x, max_x = 1, 12
        min_y, max_y = 5, 65     

    fig = plt.figure(
        figsize=(len(tissues)*6, 12),
        constrained_layout=True
    )
    fig.suptitle(f"Aboav-Weaire Law for 2D slices along {version} axes", fontsize=36)
    subplot_id = 1
    for i, tissue in enumerate(tissues):
        # Get the current axis object
        if len(tissues) <= 3:
            ax = fig.add_subplot(1, len(tissues), subplot_id)
        else:
            ax = fig.add_subplot(2, int(np.ceil(len(tissues)/2)), subplot_id)
        subplot_id += 1

        data = aboav_law_stats[tissue]
        x = np.asarray(list(data.keys()), dtype=np.int64)
        y_val = list([val[0] for val in data.values()])*x
        y_err = list([val[1] for val in data.values()])*x
        x_fit = np.arange(min_x, max_x, dtype=np.int32) 
        y_th = 5*x_fit + 8
        coeff_set = np.polyfit(x, y_val, 1)
        polyline = np.poly1d(coeff_set) 
        y_fit = polyline(x_fit)

        # Plot the values and the fitted lines
        ax.errorbar(x, y_val, yerr=y_err, fmt='o', color=colors[i], ecolor='grey', capsize=8, markersize=10)
        theoretical, = ax.plot(
            x_fit, y_th, 
            color='blue', linestyle='-.', 
            label=f'Theoretical line'
        )
        if fitted:
            linear, = ax.plot(
                x_fit, y_fit, 
                color='red', linestyle='--', 
                label=f'Linear fit, coeff: a={round(coeff_set[1], 2)}, b={round(coeff_set[0], 2)}'
            )

        # Set title and axes labels
        ax.set_title(f'{tissue.replace("_", " ").title()}: {tissue_types[i].replace("_", " ")}', fontsize=28)
        ax.set_xlabel(r'Number of neighbors $(n)$', fontsize=24)
        ax.set_ylabel(r'${m_n}n$', fontsize=24)
        ax.set_xticks(x_fit)
        if fitted:
            legend_items = [theoretical, linear]
            legend_text = 'Theoretical line eq: ${m_n}n = 8 + 5n$ \nFitted line eq: $y=a+bx$'
        else:
            legend_items = [theoretical]
            legend_text = 'Theoretical line eq: ${m_n}n = 8 + 5n$'
        ax.legend(handles=legend_items, loc='upper left', fontsize=16)
        ax.text(
            x=7, 
            y=25, 
            s=legend_text, 
            style='italic', 
            fontsize=12
        )  

        # Set axes limits
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)
    
    # plt.subplots_adjust(top=0.8)

    # Save the current plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"aboav_law_2D_{version}_plots.jpg"
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', dpi=150) 

    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def area_variability_plots(
    df: pd.DataFrame,
    version: Literal['standard', 'principal'] = 'standard',
    remove_outliers: Optional[bool] = True,
    color_map: Optional[Union[ListedColormap, str]] = 'viridis',
    save_dir: Optional[str] = None,     
    show: Optional[bool] = False 
) -> None:
    '''
    Make a plot of to empirically check the 2D Lewis Law across different slices of tissues.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe.

        version: (Literal['standard', 'principal'], default='standard')
            If 'standard', 2D statistics collected along coordinate axis
            are considered. If 'principal', 2D stats collected along cells'
            principal axes are considered instead.

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

    area_stats = _get_area_CV(
        df, principal_axis=(version=='principal')
    )

    min_x, max_x = 2, 12
    max_y = 2.5

    fig = plt.figure(
        figsize=(len(tissues)*6, 12),
        constrained_layout=True
    )
    fig.suptitle(f"Area variability vs. Number of Neighbors", fontsize=36)
    subplot_id = 1
    for i, tissue in enumerate(tissues):
        # Get the current axis object
        if len(tissues) <= 3:
            ax = fig.add_subplot(1, len(tissues), subplot_id)
        else:
            ax = fig.add_subplot(2, int(np.ceil(len(tissues)/2)), subplot_id)
        subplot_id += 1

        data = area_stats[tissue]
        x = np.asarray(list(data.keys()), dtype=np.int64)
        y = np.asarray(list(data.values()), dtype=np.float64)

        # Plot the values and the fitted lines
        ax.scatter(x, y, color=colors[i])

        # Set title and axes labels
        ax.set_title(f'{tissue.replace("_", " ").title()}: {tissue_types[i].replace("_", " ")}', fontsize=28)
        ax.set_xlabel(r'Number of neighbors $(n)$', fontsize=24)
        ax.set_ylabel(r'Area CV', fontsize=24)
        ax.set_xticks(np.arange(min_x, max_x))

        # Set axes limits
        # ax.set_xlim([min_x, max_x])
        ax.set_ylim([0, max_y])

        # Remove the square around the plot
        sns.despine(left=False, bottom=False, top=True, right=True)
    
    # plt.subplots_adjust(top=0.8)

    # Save the current plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = f"area_variability_plots.jpg"
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

    tissue_df = df[df['tissue_type'] == tissue]
    num_plots = len(features)
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 10), sharey=False)
    fig.suptitle(f'Morphological statistics distribution in {tissue} sample', fontsize=30)
    fig.tight_layout(pad=6)

    if isinstance(color_map, str):
        colors = sns.color_palette(color_map, num_plots)
    elif isinstance(color_map, ListedColormap):
        colors = color_map.colors

    for i, feature in enumerate(features):
        data = tissue_df.loc[:, feature]
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
#------------------------------------------------------------------------------------------------------------