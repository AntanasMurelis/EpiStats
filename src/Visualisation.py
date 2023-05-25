import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

#------------------------------------------------------------------------------------------------------------
def generate_plots(input_data, columns_to_plot=None, plot_type='violin', output_folder: str = None, exclude_columns=None):
    """
    Generate plots for the provided columns using the exported CSV file or the pandas DataFrame.

    Parameters:
    -----------
    input_data: (str or pd.DataFrame)
        Path to the exported CSV file containing cell statistics or a pandas DataFrame containing the cell statistics.

    columns_to_plot: (list of str, optional)
        List of columns to plot. If None, all numeric columns from the input data will be plotted.

    plot_type: (str, optional, default='violin')
        Type of plot to generate. Supported types are 'violin', 'box', and 'histogram'.

    output_folder: (str, optional)
        Path to the folder where the plots will be saved. If None, the plots will be displayed but not saved.

    exclude_columns: (list of str, optional)
        List of columns to exclude from plotting. If None, all columns containing 'id' in their titles will be excluded.

    Returns:
    --------
    None
    """

    # Load the cell statistics DataFrame from a CSV file or use the input DataFrame
    if isinstance(input_data, str):
        data_df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        data_df = input_data
    else:
        raise ValueError("input_data should be either a str (path to CSV file) or a pandas DataFrame")

    # If columns_to_plot is None, plot all numeric columns except the excluded ones
    if columns_to_plot is None:
        columns_to_plot = data_df.select_dtypes(include=['number']).columns

    # If exclude_columns is None, exclude all columns containing 'id' in their titles
    if exclude_columns is None:
        exclude_columns = [col for col in columns_to_plot if 'id' in col]

    # Exclude specified columns and filter out columns with None values
    columns_to_plot = [col for col in columns_to_plot if col not in exclude_columns and data_df[col].notna().any()]

    sns.set(style="whitegrid")

    # Create subplots for each column
    num_plots = len(columns_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 10), sharey=False)
    fig.tight_layout(pad=6)

    colors = sns.color_palette("hls", num_plots)

    for i, col in enumerate(columns_to_plot):
        data = data_df[[col]]

        if plot_type == 'violin':
            sns.violinplot(data=data, orient="v", cut=0, inner="quartile", ax=axes[i], color=colors[i])
            sns.stripplot(data=data, color=".3", size=4, jitter=True, ax=axes[i])
        elif plot_type == 'box':
            sns.boxplot(data=data, orient="v", ax=axes[i], color=colors[i])
        elif plot_type == 'histogram':
            sns.histplot(data=data, kde=True, ax=axes[i], color=colors[i])
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}. Supported types are 'violin', 'box', and 'histogram'.")

        axes[i].xaxis.set_tick_params(labelbottom=False)
        axes[i].set(xlabel=col)
        axes[i].set_title("", pad=-15)

    sns.despine(left=True)

    # Save the plot to a file or display it
    if output_folder is not None:
        plt.savefig(output_folder)
    else:
        plt.show()

    # Close all figures to release memory
    plt.close('all')
#------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------
def generate_required_plots(plot: str, output_directory: str, cell_statistics_df: pd.DataFrame, filtered_cell_statistics: pd.DataFrame, plot_type: str):
    """
    This function generates the requested plots based on the specified plot type and saves them in the provided output directory.

    Parameters:
    - plot: A string indicating which plots to generate. Can be 'filtered', 'all', 'both' or None.
    - output_directory: The directory where the generated plots will be saved.
    - cell_statistics_df: A DataFrame containing statistics for all cells.
    - filtered_cell_statistics: A DataFrame containing statistics for only the filtered cells.
    - plot_type: The type of plot to generate. 

    No return value. The function saves the generated plots directly to the specified output directory.
    """
    
    if plot in ('filtered', 'both'):
        generate_plots(input_data = filtered_cell_statistics, plot_type = plot_type, output_folder = os.path.join(output_directory, 'filtered_cell_plots.png'))
    
    if plot in ('all', 'both'):
        generate_plots(input_data = cell_statistics_df, plot_type = plot_type,  output_folder = os.path.join(output_directory,'all_cell_plots.png'))
#--------------------------------------------------------------------------------------------------