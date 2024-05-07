import logging
import pypdf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError
from matplotlib.figure import Figure
from typing import Tuple, Iterable
from io import BytesIO

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from axoden.volume_projections import collect_info_from_filename
from axoden.volume_projections import process_image
from axoden.gui.streamlit_app.pdf_utils import fig2pdfpage, pages2pdf

from axoden.volume_projections import plot_summary_data, plot_signal_intensity_along_axis
from axoden.gui.streamlit_app.pdf_utils import fig2stream


matplotlib.use('Agg')  # fixes hanging tests due to use of tk backend

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def init_session_state():
    """Initializes the session state variables."""
    if 'figures' not in st.session_state:
        st.session_state.figures = []

    if 'ctrl_plots_pdf' not in st.session_state:
        st.session_state.ctrl_plots_pdf = None

    if 'table_data' not in st.session_state:
        st.session_state.table_data = None

    if 'table_data_axis' not in st.session_state:
        st.session_state.table_data_axis = None

    if 'figure_cache' not in st.session_state:
        st.session_state.figure_cache = {}


def invalidate_figure_cache():
    """Invalidates the figure cache."""
    st.session_state.figure_cache = {}
    logger.info('invalidated figure cache')


@st.cache_data
def get_brain_regions(raw_files: list[UploadedFile]) -> Iterable[str]:
    """
    Get the brain regions from the uploaded files.

    Args:
        raw_files (List[UploadedFile]): The raw uploaded files to process.

    Returns:
        List[str]: A list of brain regions extracted from the raw files.
    """
    logger.info('get_brain_regions')
    brain_regions = set()
    for raw_file in raw_files:
        _, brain_region, _ = collect_info_from_filename(raw_file.name)
        brain_regions.add(brain_region)
    return list(brain_regions)


def cached_plot_summary_data(table_data: pd.DataFrame, project_name: str) -> Tuple[Figure, BytesIO]:
    """
    Generate a cached plot of summary data.

    Args:
        table_data (pd.DataFrame): The input DataFrame containing the data for the plot.
        project_name (str): The name of the project.

    Returns:
        Tuple[Figure, BytesIO]: A tuple containing the generated plot (Figure object) and the BytesIO stream.
    """
    fig = plot_summary_data(table_data, project_name)
    fig_stream = fig2stream(fig)
    return fig, fig_stream


def cached_plot_signal_intensity_along_axis(
        project_name: str,
        table_data_axis: pd.DataFrame,
        pixel_size: float
    ) -> Tuple[Figure, BytesIO]:
    """
    Generate a plot of signal intensity along an axis.

    Args:
        project_name (str): The name of the project.
        table_data_axis (pd.DataFrame): The table data containing the signal intensity values along the axis.
        pixel_size (float): The size of each pixel.

    Returns:
        Tuple[Figure, BytesIO]: A tuple containing the generated figure and a BytesIO object representing the figure.

    """
    logger.info('creating signal intensity along axis')
    fig = plot_signal_intensity_along_axis(project_name, table_data_axis, pixel_size)
    fig_stream = fig2stream(fig)
    return fig, fig_stream


def get_figure_by_brain_region(
        figures: list[pypdf.PdfWriter],
        brain_areas: list[str]
) -> dict[str, list[pypdf.PdfWriter]]:
    """
    Group the given figures by brain region.

    Args:
        figures (list[pypdf.PdfWriter]): A list of figures to be grouped.
        brain_areas (list[str]): A list of brain areas corresponding to each figure.

    Returns:
        dict[str, list[pypdf.PdfWriter]]: A dictionary where the keys are brain regions and the values are lists of figures belonging to each brain region.
    """
    logger.info('get_figure_by_brain_region')
    figures_out = {}
    for fig, brain_area in zip(figures, brain_areas):
        if brain_area not in figures_out:
            figures_out[brain_area] = []

        figures_out[brain_area].append(fig)

    return figures_out


def process_images(
        raw_files: list[UploadedFile],
        pixel_size: float,
        is_masked: bool,
        cache: dict[(str, float, bool), (pypdf.PdfWriter, dict, dict)] = None) \
            -> tuple[list[plt.Figure], pd.DataFrame, pd.DataFrame]:
    """
    Process a list of raw image files.

    Args:
        raw_files (list[UploadedFile]): A list of raw image files.
        pixel_size (float): The pixel size.
        is_masked (bool): A flag indicating whether the images are masked.
        cache (dict[(str, float, bool), (pypdf.PdfWriter, dict, dict)], optional):
            Use st.session_state.figure_cache as an intermediate cache. Defaults to None.

    Returns:
        tuple[list[plt.Figure], pd.DataFrame, pd.DataFrame]: A tuple containing the processed figures,
            table data, and table data axis.
    """
    
    if not raw_files:
        return [], None, None

    logger.info(f'process_images')
    table_data = pd.DataFrame()
    table_data_axis = pd.DataFrame()

    figures = []

    progress_bar = st.progress(0.0, text=f"Processing image {1}/{len(raw_files)}")
    progress_step_size = 1.0 / len(raw_files)

    for i, raw_image in enumerate(raw_files):
        progress_bar.progress(i*progress_step_size, text=f"Processing image {(i+1)}/{len(raw_files)}")
        fig, data_row, data_axis_row = process_image_single(raw_image, pixel_size, is_masked, cache=cache)
        
        if np.sum(table_data.shape) == 0:
            table_data = pd.DataFrame(columns=data_row.keys())
        table_data.loc[len(table_data)] = data_row

        if np.sum(table_data_axis.shape) == 0:
            table_data_axis = pd.DataFrame(columns=data_axis_row.keys())
        table_data_axis.loc[len(table_data_axis)] = data_axis_row

        figures += [fig]

    # Compute the percentage of white pixels
    table_data['percent_signal'] = table_data['pixels_signal'] / table_data['pixels_total'] * 100

    progress_bar.empty()
    return figures, table_data, table_data_axis


def process_image_single(raw_image, pixel_size, is_masked, cache={}):
    """
    Process a single image.

    Args:
        raw_image (object): The raw image object.
        pixel_size (float): The pixel size.
        is_masked (bool): Flag indicating whether the image is masked.
        cache (dict, optional): The cache dictionary to store processed images. Defaults to None.

    Returns:
        tuple: A tuple containing the processed image, temporary variables, and temporary axis.
    """
    
    cache_key = (raw_image.file_id, pixel_size, is_masked) 
    if cache_key in cache:
        logger.info(f'process_image_single found cache for {cache_key}')
        return cache[cache_key]

    try:
        animal, brain_area, group = collect_info_from_filename(raw_image.name)
    except ValueError as e:
        logger.error(f'Error: {e}')
        st.warning(f"Error: {e}")
        st.stop()
        return None, None, None
    try:
        fig, _temp_, _temp_axis_ = process_image(
            raw_image, is_masked, pixel_size,
            animal=animal, brain_area=brain_area, group=group
        )
    except UnidentifiedImageError as e:
        logger.error(f'Error: {e}')
        st.warning(f"Error: {e}")
        st.stop()
        return None, None, None

    pdf_fig = fig2pdfpage(fig)
    pdf_fig = pages2pdf([pdf_fig])
    plt.close(fig)

    # st.session_state.figure_cache[(raw_image.file_id, pixel_size, is_masked)] = (pdf_fig,  _temp_, _temp_axis_)
    cache[(raw_image.file_id, pixel_size, is_masked)] = (pdf_fig,  _temp_, _temp_axis_)
    logger.info(f'process_image_single created cache for {cache_key}')

    return pdf_fig, _temp_, _temp_axis_