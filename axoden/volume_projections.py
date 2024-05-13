import os
import re
from pathlib import Path
from typing import Tuple

# this allows pyinstaller on github actions to detect pdf backend
import matplotlib.backends.backend_pdf  # noqa: F401
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage.filters import threshold_otsu
from tqdm import tqdm


def intensity_along_axis(img, ax=None):
    """Compute the intensity along the x or y axis of the image."""
    if ax == "x":
        s = np.nansum(img, axis=0)  # Sum along the x axis
    elif ax == "y":
        s = np.nansum(img, axis=1)  # Sum along the y axis
    elif ax is None:
        s = (
            np.nansum(img, axis=0),
            np.nansum(img, axis=1),
        )  # Sum along the x and y axis
    else:
        raise ValueError(
            f"Expected argument ax to be one of 'x' or 'y' or None, but received '{ax}'"
        )
    return s


def remove_spines_plot(ax, loc=["all"]):
    """Remove the spines of the plot."""
    if len(loc) == 1 and loc[0] == "all":
        loc = ["top", "right", "bottom", "left"]
        [ax.spines[spine_location].set_visible(False) for spine_location in loc]
    else:
        [ax.spines[spine_location].set_visible(False) for spine_location in loc]


def remove_ticks_plot(ax, loc="all"):
    """Remove the ticks of the plot."""
    if loc == "all":
        ax.set_xticks([])
        ax.set_yticks([])
    elif loc == "x":
        ax.set_xticks([])
    elif loc == "y":
        ax.set_yticks([])
    else:
        raise ValueError(
            "Expected argument 'loc' to be one of 'all', 'x', 'y', "
            f"but received '{loc}'"
        )


def plot_intensity_along_axis(ax, x, y, loc):
    if loc == "y":
        ax.fill_betweenx(x, y, color="grey", alpha=0.5)
    elif loc == "x":
        ax.fill_between(x, y, color="grey", alpha=0.5)
    ax.margins(0)


def control_plot_steps(ax, img, title, axs_info):
    """
    Create the subplots of the control plots for each input image.

    Args:
        ax (list of matplotlib Axes): List of Axes objects to plot the images.
        img (numpy array): Input image data.
        title (str): Title for the plot.
        axs_info (dict or list): Information about the axes.

    Returns:
        ax (list of matplotlib Axes): List of Axes objects with the plotted images.
    """

    # Compute x and y axis signals
    x_ax_signal, y_ax_signal = intensity_along_axis(img)

    # Define the label of the quantification plots
    if title != "Mask":
        label_quant = "Signal\nintensity"
    else:
        label_quant = "Dimensions\nmask"

    # Plot original image, binarized image and area of the original image
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title(title, pad=12)
    # Beautify
    remove_ticks_plot(ax[0], loc="x")
    remove_spines_plot(ax[0], loc=["top", "right", "bottom"])

    # Plot the signal in the y axis
    plot_intensity_along_axis(
        ax[1], np.arange(len(y_ax_signal)), y_ax_signal[::-1], "y"
    )
    # Set x label at given location
    ax[1].set_xlabel(label_quant, loc="center", rotation=45, color="grey")
    ax[1].xaxis.set_label_coords(1, -0.01)
    # Beautify
    remove_ticks_plot(ax[1])
    remove_spines_plot(ax[1])

    # Plot the signal in the x axis
    plot_intensity_along_axis(ax[2], np.arange(len(x_ax_signal)), -x_ax_signal, "x")
    # Beautify
    remove_ticks_plot(ax[2], loc="y")
    remove_spines_plot(ax[2], loc=["top", "right", "left"])

    # Match the size of ax[1] and ax[2] to ax[0]
    ax0_size = ax[0].get_position()
    ax1_size = ax[1].get_position()
    ax[1].set_position(
        [ax0_size.x0 + ax0_size.width, ax0_size.y0, ax1_size.width, ax0_size.height]
    )
    ax2_size = ax[2].get_position()
    ax[2].set_position(
        [ax0_size.x0, ax0_size.y0 - ax2_size.height, ax0_size.width, ax2_size.height]
    )

    # Set x ticks and labels for the medio-lateral axis
    x_ax_ticks = axs_info[0]
    x_ax_labels = axs_info[1]
    ax[2].set_xticks(x_ax_ticks)
    ax[2].set_xticklabels(x_ax_labels, rotation=45)
    ax[2].set_xlabel("Medio-Lateral axis (\u03bcm)")

    # Set y ticks and labels for the dorso-ventral axis
    y_ax_ticks = axs_info[2]
    y_ax_labels = axs_info[3]
    ax[0].set_yticks(y_ax_ticks)
    ax[0].set_yticklabels(y_ax_labels)
    ax[0].set_ylabel("Dorso-Ventral axis (\u03bcm)")

    return ax


def get_tif_files(folder_path):
    """Get all the tif files in the folder provided."""
    tif_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            tif_files.append(os.path.join(folder_path, file))
    return tif_files


def collect_info_from_filename(filename: str) -> Tuple[str, str, str]:
    """Collect the animal, brain area and group from the filename.

    Args:
        filename (str): Name of the file.

    Returns:
        Tuple[str, str, str]: A tuple containing the animal, brain area and group.
    """
    img_name = os.path.basename(filename)
    img_name, _ = os.path.splitext(img_name)

    # img_name = img_name.split('.')[0]
    name_parts = img_name.split("_")
    if len(name_parts) < 2:
        raise ValueError(
            "Expected filename to include at least one '_' character to split animal "
            "from brain area, e.g. 'animal_brainarea.tif', "
            f"but received the file name '{filename}'"
        )
    animal = name_parts[0]
    brain_area = name_parts[1]

    # backward compatibility for only animal and brain area
    group = "group1"
    if len(name_parts) >= 3:
        group = name_parts[2]

    return animal, brain_area, group


def open_tif_image(img_path):
    """Open a tif image.

    Args:
        img_path: A filename (string), os.PathLike object or a file object.

    Returns:
        Image: A PIL Image object.
    """
    return Image.open(img_path)  # works well also with Uploaded image


def generate_background_mask(img, is_masked):
    """Generate a background mask for the image provided."""

    image_matrix = np.array(img)

    # For non-masked images
    if not is_masked:
        mask = np.zeros(image_matrix.shape[0:2], dtype="bool")

    # For masked images
    else:
        if image_matrix.ndim == 3:
            mask = np.all(image_matrix == [0, 0, 0], axis=-1)
        elif image_matrix.ndim == 2:
            mask = image_matrix == 0
        else:
            raise ValueError(
                "Expected image of dimension 2 or 3, "
                f"but reiceived shape {image_matrix.shape}"
            )

    return mask


def convert_image_to_gray(img):
    return img.convert("L")


def collect_within_mask(img, mask):
    return np.where(mask, np.nan, img)


def collect_image_mask(img_path: any, is_masked: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Open the image and cut it to the area that contains tissue.

    Args:
        img_path: A filename (string), os.PathLike object or a file object.
        is_masked: True if the image was masked and contains regions that are set
            to a value of 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the cut image and the mask.
    """
    img = open_tif_image(img_path)
    # Find the area of the image that cotains tissue
    mask = generate_background_mask(img, is_masked)
    # Convert image to grayscale
    image_gray = convert_image_to_gray(img)
    # Cut the image to the area that contains tissue
    cut_image = collect_within_mask(image_gray, mask)
    return cut_image, mask


def binarize_image(image, threshold):
    """Binarize the image using the threshold provided."""
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image


def count_pixels(img):
    """Count the number of white and black pixels in the image."""

    # make sure the image is binarized and only contains black (0) and white (1) pixels
    # otherwise this function will not behave properly!
    unique_values = np.unique(img)
    if not (len(unique_values) == 2 and np.all(np.unique(img) == [0, 1])):
        raise ValueError(
            "Expected a binarized image with only zeros and ones, "
            f"but received values {unique_values}"
        )

    white_pixels = np.sum(img)
    black_pixels = np.sum(1 - img)
    all_pixels = white_pixels + black_pixels
    return [white_pixels, black_pixels, all_pixels]


def compute_area(img, pixel_size):
    """Compute the area of the image and the area of the white and black pixels."""
    counts = count_pixels(img)
    [w, b, total] = [c * (pixel_size**2) for c in counts]
    return w, b, total


def compute_threshold(img):
    non_nan_values = img[~np.isnan(img)]
    thr = threshold_otsu(np.array(non_nan_values))
    # If the threshold is too low, set it to 20
    if thr <= 10:
        thr = np.int32(20)
    return thr


def make_figures_pdf_editable():
    """Make the figures editable in Adobe Illustrator."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def prepare_axis_information(img, pixel_size):
    x_ax_pixels = np.linspace(0, img.shape[1], 6)
    x_ax_um = np.round(x_ax_pixels * pixel_size, decimals=1)
    y_ax_pixels = np.linspace(0, img.shape[0], 6)
    y_ax_um = np.round(y_ax_pixels * pixel_size, decimals=1)
    return [x_ax_pixels, x_ax_um, y_ax_pixels, y_ax_um]


def load_table(file_path: str):
    """Load a table from a csv file.

    Args:
        file_path (str): Path to the csv file.

    Returns:
        pd.DataFrame: A DataFrame with the data from the csv file.
    """

    def _string_to_ndarray(string_list):
        """Convert a string representation of a list of numbers to a numpy array."""
        return np.array([int(n) for n in re.findall(r"\d+", string_list)])

    col_types = {"animal": str, "brain_area": str}
    table_data = pd.read_csv(file_path, dtype=col_types, index_col=False)

    # Remove any unnamed columns. If the .csv was saved from the streamlit app,
    # it will contain an unnamed index column
    table_data = table_data.loc[:, ~table_data.columns.str.contains("^Unnamed")]

    for col in table_data.columns:
        if "signal_bin" in col or "signal_gray" in col:  # those are ndarray columns
            table_data[col] = table_data[col].apply(_string_to_ndarray)

    return table_data


def save_table(df: pd.DataFrame, folderpath: str, filename: str):
    """Save a table to a csv file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        folderpath (str): Path to the folder where the file will be saved.
        filename (str): Name of the file.
    """
    df = df.copy()  # make sure the original dataframe is not modified

    Path(folderpath).mkdir(parents=True, exist_ok=True)

    filepath = os.path.join(folderpath, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    # convert numpy arrays to lists, otherwise the string representation is saved
    # as [0, 5, 2, ..., 3, 4, 5]
    for col in df.columns:
        if isinstance(df[col][0], np.ndarray):
            # [1:-1] removes brackets
            # .replace(' ', '') removes spaces to make file size smaller
            df[col] = df[col].apply(
                lambda x: str(list(x.astype(int)))[1:-1].replace(" ", "")
            )
    df.to_csv(filepath, index=False)


def generate_control_plot(img, img_bin, msk, pixel_size, info_pie):
    # Create a grid for the subplots
    gs = gridspec.GridSpec(13, 11)

    # Prepare axis informationTransform x and y axes from pixels to mm
    axis_info = prepare_axis_information(img, pixel_size)

    # Subplot original image
    ax_ori = [
        plt.subplot(gs[0:4, 0:4]),
        plt.subplot(gs[0:4, 4]),
        plt.subplot(gs[4, 0:4]),
    ]
    ax_ori = control_plot_steps(ax_ori, img, "Grayscale image", axis_info)

    # Subplot binarized image
    ax_bin = [
        plt.subplot(gs[0:4, 6:10]),
        plt.subplot(gs[0:4, 10]),
        plt.subplot(gs[4, 6:10]),
    ]
    ax_bin = control_plot_steps(ax_bin, img_bin, "Binarized image", axis_info)

    # Subplot area of the original image
    ax_msk = [
        plt.subplot(gs[7:11, 0:4]),
        plt.subplot(gs[7:11, 4]),
        plt.subplot(gs[11, 0:4]),
    ]
    ax_msk = control_plot_steps(ax_msk, ~msk, "Mask", axis_info)

    # Subplot pie chart
    ax_pie = plt.subplot(gs[7:11, 6:11])
    ax_pie.pie(
        info_pie["sizes"],
        labels=info_pie["labels"],
        autopct="%1.1f%%",
        startangle=90,
        colors=info_pie["colors"],
        labeldistance=1.2,
        wedgeprops={"edgecolor": "black", "linewidth": 0.5, "antialiased": True},
    )
    # set title to subplot
    ax_pie.set_title("Proportion of signal vs background", pad=12)
    # Set aspect ratio to equal for a circular pie chart
    ax_pie.set_aspect("equal")
    # Add a line to the pie chart
    ax_pie.add_line(plt.Line2D([0.5, 0.5], [0.5, 0.5], color="black", linewidth=1))


def process_image(
    file_name: any,
    is_masked: bool,
    pixel_size: float,
    animal: str = "animal1",
    brain_area: str = "brain_area1",
    group: str = "group1",
) -> Tuple[plt.Figure, dict, dict]:
    """Process a single image and generate a control plot.

    Args:
        file_name (any): A filename (string), os.PathLike object or a file object.
        is_masked (bool): True if the image was masked and contains regions that are
            set to a value of 0.
        pixel_size (float): Pixel size in micrometers.
        animal (str): Name of the animal. Default is "animal1".
        brain_area (str): Name of the brain area. Default is "brain_area1".
        group (str): Name of the group. Default is "group1".

    Returns:
        Tuple[plt.Figure, dict, dict]: A tuple containing the figure, \
            the data for the image and the data with the axis projections.
    """
    img, msk = collect_image_mask(file_name, is_masked)
    msk_bool = msk.astype(bool)
    thr = compute_threshold(img[~msk_bool])
    img_bin = binarize_image(img, thr)

    [w, b, total] = count_pixels(img_bin[~msk_bool])
    area_w, area_b, area_img = compute_area(img_bin[~msk_bool], pixel_size)
    area_image_mm = area_img / 1000

    # Append the information to the DataFrame for the image
    data = {
        "animal": animal,
        "brain_area": brain_area,
        "group": group,
        "pixels_signal": w,
        "pixels_black": b,
        "pixels_total": total,
        "threshold": thr,
        "area_image": area_img,
        "area_signal": area_w,
        "area_black": area_b,
        "area_img_mm": area_image_mm,
    }

    # Append the information to the DataFrame for the axis
    axis_data = {
        "animal": animal,
        "brain_area": brain_area,
        "group": group,
        "signal_bin_x_ax": intensity_along_axis(img_bin, "x"),
        "signal_bin_y_ax": intensity_along_axis(img_bin, "y"),
        "signal_gray_x_ax": intensity_along_axis(img, "x"),
        "signal_gray_y_ax": intensity_along_axis(img, "y"),
    }

    # Generate control plot
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(
        f"Animal {animal} | {brain_area} | Area: {area_image_mm:.2f}mm\u00b2| "
        f"{group} | Threshold: {thr:.2f}",
        weight="bold",
    )
    info_pie = {
        "labels": ["Area receiving\nprojections", "Area without\nprojections"],
        "sizes": [area_w, area_b],
        "colors": ["white", "grey"],
    }
    generate_control_plot(img, img_bin, msk, pixel_size, info_pie)

    return fig, data, axis_data


def process_folder(
    folder_path: str,
    pixel_size: float,
    is_masked: bool,
    output_folder: str = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process all images in a folder.

    Optionally save the data to csv files and the control plots to pdf files.

    Args:
        folder_path (str): Path to the folder containing the images.
        pixel_size (float): Pixel size in micrometers.
        is_masked (bool): True if the image was masked and contains regions that are
            set to a value of 0.
        output_folder (str): Path to the folder where the data will be saved.
            If not set, the input folder will be used.
        save (bool): True if the control plots should be saved. Default is True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the data for the images \
            and the data for the axis.
    """

    if not output_folder:
        output_folder = folder_path

    # Set fonts editable in Adobe Illustrator
    make_figures_pdf_editable()

    # Create an empty DataFrame to store the information
    table_data = pd.DataFrame()
    table_data_axis = pd.DataFrame()

    # Create a folder for the control plots if it doesn't exist in the folder path
    control_plot_path = os.path.join(output_folder, "control_plots")
    if save:
        Path(control_plot_path).mkdir(parents=True, exist_ok=True)

    # Create a list with all the tif files in the folder
    file_list = get_tif_files(folder_path)
    if len(file_list) == 0:
        raise ValueError("No tif files found in the folder path")

    # Loop through all the images in the folder
    for filepath in tqdm(file_list, desc="Processing images"):
        animal, brain_area, group = collect_info_from_filename(filepath)
        fig, _temp_, _temp_axis_ = process_image(
            filepath,
            is_masked=is_masked,
            pixel_size=pixel_size,
            animal=animal,
            brain_area=brain_area,
            group=group,
        )

        if np.sum(table_data.shape) == 0:
            table_data = pd.DataFrame(columns=_temp_.keys())
        table_data.loc[len(table_data)] = _temp_

        if np.sum(table_data_axis.shape) == 0:
            table_data_axis = pd.DataFrame(columns=_temp_axis_.keys())
        table_data_axis.loc[len(table_data_axis)] = _temp_axis_

        # Save control plot
        if save:
            figure_name = f"{animal}_{brain_area}_control_plot.pdf"
            figure_path = os.path.join(control_plot_path, figure_name)
            plt.savefig(figure_path, dpi=300)
        plt.close(fig)

    # Compute the percentage of white pixels
    table_data["percent_signal"] = (
        table_data["pixels_signal"] / table_data["pixels_total"] * 100
    )

    # Save dataframes as csv files
    if save:
        save_table(table_data, output_folder, "projections_quantification.csv")
        save_table(
            table_data_axis, output_folder, "projections_quantification_axis.csv"
        )

    return table_data, table_data_axis


def write_summary_data_plot(
    folder_path: str, df_input: pd.DataFrame, project_name: str = None
):
    """Write the summary data plot to a pdf file in the folder provided.

    Args:
        folder_path (str): Path to the folder where the data will be saved.
        df_input (pd.DataFrame): Dataframe with the data to plot.
        project_name (str): Name of the project. If not set, default to the folder name.
    """

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    if not project_name:
        project_name = os.path.basename(folder_path)

    fig = plot_summary_data(df_input, project_name)

    # Save the figure
    figure_name = "projections_quantification.pdf"
    figure_path = os.path.join(folder_path, figure_name)
    fig.savefig(figure_path, dpi=300)
    plt.close()


def plot_summary_data(df_input, project_name):
    # Set fonts editable in Adobe Illustrator
    make_figures_pdf_editable()

    # Create a figure with four subplots
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    brain_areas = df_input["brain_area"].unique()
    fig.suptitle(f'Project "{project_name}"\nData Summary', weight="bold")

    # Set the color for the plots
    color_bars = "royalblue"
    color_scatter_in = "white"

    # Create a dictionary with the animals per brain area
    dict_animals_brain_area = {}

    # Define which columns to use for the plot
    data_cols = ["percent_signal", "area_img_mm", "threshold"]
    ylabel_dict = {
        "percent_signal": "Innervation (%)",
        "area_img_mm": "Area (mm\u00b2)",
        "threshold": "Pixel value",
    }
    sublplot_titles = [
        "Percentage of innervation",
        "Area of the brain region",
        "Threshold for binarization",
    ]

    # Plot the bar plot with the mean and sem of the white pixels percentage
    for i_ba, br_ar in enumerate(brain_areas):

        # Get the data for the brain area
        data = df_input[df_input["brain_area"] == br_ar]

        # Iterate over the columns to plot
        for i_col, col in enumerate(data_cols):
            vals = data[col]
            vals_mean = np.mean(vals)
            sem = np.std(vals) / np.sqrt(len(vals))

            # Plot the bar plot with the mean and sem
            ax[i_col].bar(i_ba, vals_mean, color=color_bars, yerr=sem)

            # Plot the scatter plot with round markers
            ax[i_col].scatter(
                np.repeat(i_ba, len(vals)),
                vals,
                edgecolors=color_bars,
                facecolors=color_scatter_in,
            )

        # Fill the dictionary with the animals in the brain area
        dict_animals_brain_area[br_ar] = data["animal"].unique().tolist()

    # Beautify data subplots
    for i_col, col in enumerate(data_cols):
        # Set the column name as the title
        ax[i_col].set_title(sublplot_titles[i_col], weight="bold")
        # Set the xticks and labels for each data column
        ax[i_col].set_xticks(np.arange(0, len(brain_areas)))
        ax[i_col].set_xticklabels(brain_areas, rotation=45)
        # Set the y maximum value
        yval_max = np.ceil(ax[i_col].get_ylim()[1] / 10) * 10
        ax[i_col].set_yticks(np.linspace(0, yval_max, 6))
        ax[i_col].set_ylim(0, yval_max)
        # Set the y label
        ax[i_col].set_ylabel(ylabel_dict[col])

    # Print the number of animals per brain area
    df = pd.DataFrame.from_dict(dict_animals_brain_area, orient="index")
    df["animals"] = df.apply(lambda row: ", ".join(row.astype(str)), axis=1)
    df["animals"] = df["animals"].apply(
        lambda x: x.replace(", None", "")
    )  # Remove None values
    i_anim_col = df.columns.get_loc("animals")
    df = df.iloc[:, i_anim_col:]  # Remove columns 0 to the one with all the animals
    df["brain_area_with_animals"] = (
        df.index + " (N=" + df["animals"].apply(lambda x: str(x.count(",") + 1)) + ")"
    )  # Count the number of animals
    df = df.set_index(
        df["brain_area_with_animals"]
    )  # Replace the index with the values in the new column
    df = df.drop(columns=["brain_area_with_animals"])  # Remove the column
    df.columns = [""] * len(df.columns)  # Remove column
    df.index.name = ""  # Remove index names
    ax[-1].set_title("Animals per Brain Region", weight="bold")
    ax[-1].axis("off")
    ax[-1].text(0.05, 0.1, df.to_string(header=False), fontsize=9)

    # Set the layout & despine
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True)

    # Delete the DataFrame
    del df, df_input
    return fig


def write_signal_intensity_along_axis_plot(
    output_folder: str, df: pd.DataFrame, pixel_size: float, project_name: str = None
):
    """Write the signal intensity along the axis to a pdf file in the folder provided.

    Args:
        folderpath (str): Path to the folder where the data will be saved.
        df (pd.DataFrame): Dataframe with the axis data to plot.
        pixel_size (float): Pixel size in micrometers.
        project_name (str): Name of the project.
            If not set, it will be the name of the output folder.
    """

    # Collect project name
    if not project_name:
        project_name = os.path.basename(output_folder)

    fig = plot_signal_intensity_along_axis(project_name, df, pixel_size)

    # Save the figure
    figure_name = "projections_quantification_along_axis.pdf"
    figure_path = os.path.join(output_folder, figure_name)
    if os.path.exists(figure_path):
        os.remove(figure_path)
    fig.savefig(figure_path, dpi=300)
    plt.close()


def plot_signal_intensity_along_axis(
    project_name: str, df: pd.DataFrame, pixel_size: float
):
    """Plot the signal intensity along the x and y axis of the DataFrame.

    Args:
        project_name (str): Name of the project.
        df (pd.DataFrame): DataFrame with the data to plot.
        pixel_size (float): Pixel size in micrometers.

    Returns:
        plt.Figure: A figure with the signal intensity along the x and y axis.
    """

    # Set fonts editable in Adobe Illustrator
    make_figures_pdf_editable()

    # Get the unique brain areas
    brain_areas = df["brain_area"].unique()
    n_ba = len(brain_areas)

    if n_ba > 5:
        n_y_ticks = 2
        size_tick_lbl = 8
    else:
        n_y_ticks = 4
        size_tick_lbl = 10

    # Define which columns to use for the plot
    data_cols = ["signal_bin_x_ax", "signal_bin_y_ax"]
    subplot_titles = ["Medio-Lateral axis", "Dorso-Ventral axis"]

    # Create a figure with four subplots
    fig, ax = plt.subplots(n_ba, len(data_cols), figsize=(16, 9), squeeze=False)
    fig.suptitle(
        f'Project "{project_name}"\n'
        "Signal Intensity Along the Medio-Lateral and Dorso-Ventral Axes",
        weight="bold",
    )

    # Set the color for the plots
    color_bars = "royalblue"

    # Plot the bar plot with the mean and sem of the white pixels percentage
    for i_ba, br_ar in enumerate(brain_areas):

        # Get the data for the brain area
        data = df[df["brain_area"] == br_ar]

        # Iterate over the columns to plot
        for i_col, col in enumerate(data_cols):

            # Collect data arrays
            vals = data[col].tolist()
            n_vals = len(vals)
            # Pad the shortest array(s) with nans
            max_len = np.max([len(v) for v in vals])
            val_eq_size = []
            for v in vals:
                if len(v) < max_len:
                    val_out = np.concatenate([v, np.zeros(max_len - len(v)) + np.nan])
                else:
                    val_out = v
                val_eq_size.append(val_out)
            vals = np.array(val_eq_size)
            del val_eq_size
            # Compute the mean and sem
            vals_mean = np.nanmean(vals, axis=0)
            sem = np.nanstd(vals, axis=0) / np.sqrt(n_vals)

            # Generate the x axis values
            n_pts = vals_mean.shape[0]
            x_ax = np.arange(0, n_pts, 1)
            x_ax_ticks = np.round(np.linspace(0, n_pts, 10), 1)
            x_ax_labels = np.round(x_ax_ticks * pixel_size, 1)
            # Plot the values of each animal
            ax[i_ba, i_col].plot(
                x_ax, vals.T, color="grey", alpha=0.5, linewidth=0.7
            )  # FIXME: crashes with only a single animal, need to test
            # Plot the bar plot with the mean and sem
            ax[i_ba, i_col].plot(x_ax, vals_mean, color=color_bars, linewidth=1.5)
            ax[i_ba, i_col].fill_between(
                x_ax, vals_mean - sem, vals_mean + sem, color=color_bars, alpha=0.2
            )

            # Beautify data subplots
            ax[i_ba, i_col].margins(0)
            ax[i_ba, i_col].set_xticks(x_ax_ticks)
            ax[i_ba, i_col].set_xticklabels(x_ax_labels, fontsize=size_tick_lbl)
            # Set the x label at the bottom
            if i_ba == n_ba - 1:
                ax[i_ba, i_col].set_xlabel(
                    f"{subplot_titles[i_col]} (\u03bcm)", weight="bold"
                )
            # Name the y axis as the brain area
            if i_col == 0:
                ax[i_ba, i_col].set_ylabel(
                    f"{br_ar}", weight="bold", rotation=0, labelpad=20, ha="right"
                )
            # Set the y maximum value
            yval_max = np.ceil(ax[i_ba, i_col].get_ylim()[1] / 10) * 10
            yticks = np.linspace(0, yval_max, n_y_ticks)
            ax[i_ba, i_col].set_yticks(yticks)
            ax[i_ba, i_col].set_yticklabels(yticks, fontsize=size_tick_lbl)
            ax[i_ba, i_col].set_ylim(0, yval_max)

    # Set the layout & despine
    sns.despine(fig=fig, top=True, right=True)
    plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.5, wspace=0.2)

    return fig


if __name__ == "__main__":
    pixel_size = (
        0.75521  # um, based on 20x objective from Keyence BZ-810X series microscope.
    )
    input_dir = "sample_images"
    output_dir = "output_data"
    is_masked = True

    # Collect the data and save it as a csv file
    quant_projections, quant_along_axis = process_folder(
        input_dir, pixel_size, is_masked, output_folder=output_dir
    )

    # Plot the summary data
    write_summary_data_plot(output_dir, quant_projections)

    # Plot the average of the signal intensity along the x and y axis
    write_signal_intensity_along_axis_plot(output_dir, quant_along_axis, pixel_size)
