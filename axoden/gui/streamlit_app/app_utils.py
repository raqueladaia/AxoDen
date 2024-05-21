import logging
from io import BytesIO
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from PIL import UnidentifiedImageError
from streamlit.runtime.uploaded_file_manager import UploadedFile

from axoden.gui.streamlit_app.pdf_utils import fig2stream
from axoden.volume_projections import (
    collect_info_from_filename,
    plot_signal_intensity_along_axis,
    plot_summary_data,
    process_image,
)

matplotlib.use("Agg")  # fixes hanging tests due to use of tk backend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


@st.cache_data(ttl=300)
def get_brain_regions(raw_files: List[UploadedFile]) -> Iterable[str]:
    """
    Get the brain regions from the uploaded files.

    Args:
        raw_files (List[UploadedFile]): The raw uploaded files to process.

    Returns:
        List[str]: A list of brain regions extracted from the raw files.
    """
    logger.info("get_brain_regions")
    brain_regions = set()
    for raw_file in raw_files:
        _, brain_region, _ = collect_info_from_filename(raw_file.name)
        brain_regions.add(brain_region)
    return list(brain_regions)


def cached_plot_summary_data(
    table_data: pd.DataFrame, project_name: str
) -> Tuple[Figure, BytesIO]:
    """
    Generate a cached plot of summary data.

    Args:
        table_data (pd.DataFrame): The input DataFrame containing the data for the plot.
        project_name (str): The name of the project.

    Returns:
        Tuple[Figure, BytesIO]: A tuple containing the generated plot (Figure object)
            and the BytesIO stream.
    """
    fig = plot_summary_data(table_data, project_name)
    fig_stream = fig2stream(fig)
    return fig, fig_stream


def cached_plot_signal_intensity_along_axis(
    project_name: str, table_data_axis: pd.DataFrame, pixel_size: float
) -> Tuple[Figure, BytesIO]:
    """
    Generate a plot of signal intensity along an axis.

    Args:
        project_name (str): The name of the project.
        table_data_axis (pd.DataFrame): The table data containing the signal intensity
            values along the axis.
        pixel_size (float): The size of each pixel.

    Returns:
        Tuple[Figure, BytesIO]: A tuple containing the generated figure and file stream.
    """
    logger.info("creating signal intensity along axis")
    fig = plot_signal_intensity_along_axis(project_name, table_data_axis, pixel_size)
    fig_stream = fig2stream(fig)
    return fig, fig_stream


def get_figure_by_brain_region(
    figures: List[BytesIO], brain_areas: List[str]
) -> Dict[str, List[BytesIO]]:
    """
    Group the given figures by brain region.

    Args:
        figures (List[BytesIO]): A list of figure streams to be grouped.
        brain_areas (List[str]): A list of brain areas corresponding to each figure.

    Returns:
        Dict[str, List[BytesIO]]: A dictionary where the keys are brain regions
            and the values are lists of figure streams belonging to each brain region.
    """
    logger.info("get_figure_by_brain_region")
    figures_out = {}
    for fig, brain_area in zip(figures, brain_areas):
        if brain_area not in figures_out:
            figures_out[brain_area] = []

        figures_out[brain_area].append(fig)

    return figures_out


def process_images(
    raw_files: List[UploadedFile],
    pixel_size: float,
    is_masked: bool,
) -> Tuple[List[BytesIO], pd.DataFrame, pd.DataFrame]:
    """Process a list of raw image files.

    Args:
        raw_files (List[UploadedFile]): A list of raw image files.
        pixel_size (float): The pixel size.
        is_masked (bool): A flag indicating whether the images are masked.

    Returns:
        Tuple[List[BytesIO], pd.DataFrame, pd.DataFrame]: A tuple containing the
            processed figure streams, table data, and table data axis.
    """

    if not raw_files:
        return [], None, None

    logger.info("process_images")
    table_data = pd.DataFrame()
    table_data_axis = pd.DataFrame()

    figures = []

    progress_bar = st.progress(0.0, text=f"Processing image {1}/{len(raw_files)}")
    progress_step_size = 1.0 / len(raw_files)

    for i, raw_image in enumerate(raw_files):
        progress_bar.progress(
            i * progress_step_size, text=f"Processing image {(i+1)}/{len(raw_files)}"
        )
        fig, data_row, data_axis_row = process_image_single_cached(
            raw_image,
            pixel_size,
            is_masked,
        )

        if np.sum(table_data.shape) == 0:
            table_data = pd.DataFrame(columns=data_row.keys())
        table_data.loc[len(table_data)] = data_row

        if np.sum(table_data_axis.shape) == 0:
            table_data_axis = pd.DataFrame(columns=data_axis_row.keys())
        table_data_axis.loc[len(table_data_axis)] = data_axis_row

        figures += [fig]

    # Compute the percentage of white pixels
    table_data["percent_signal"] = (
        table_data["pixels_signal"] / table_data["pixels_total"] * 100
    )

    progress_bar.empty()
    return figures, table_data, table_data_axis


def hash_uploaded_file(file: UploadedFile) -> str:
    """Hash util for an uploaded file."""
    return file.file_id


@st.cache_data(ttl=300, hash_funcs={UploadedFile: hash_uploaded_file}, max_entries=100)
def process_image_single_cached(
    raw_image: UploadedFile, pixel_size: float, is_masked: bool
) -> Tuple[BytesIO, Dict, Dict]:
    """Process a single image.

    Args:
        raw_image (UploadedFile): The input image object.
        pixel_size (float): The pixel size in um.
        is_masked (bool): Flag indicating whether the image is masked.

    Returns:
        Tuple[BytesIO, Dict, Dict]: A tuple containing the processed image as a stream
            and the data and axis data
    """
    try:
        animal, brain_area, group = collect_info_from_filename(raw_image.name)
    except ValueError as e:
        logger.error(f"Error: {e}")
        st.warning(f"Error: {e}")
        st.stop()
        return None, None, None
    try:
        fig, _temp_, _temp_axis_ = process_image(
            raw_image,
            is_masked,
            pixel_size,
            animal=animal,
            brain_area=brain_area,
            group=group,
        )
    except UnidentifiedImageError as e:
        logger.error(f"Error: {e}")
        st.warning(f"Error: {e}")
        st.stop()
        return None, None, None

    # pdf_fig = fig2pdfpage(fig)
    # pdf_fig = pages2pdf([pdf_fig])

    pdf_fig = fig2stream(fig)
    plt.close(fig)

    return pdf_fig, _temp_, _temp_axis_
