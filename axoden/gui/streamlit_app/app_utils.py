import streamlit as st
import logging
import pandas as pd
import numpy as np
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt

from axoden.volume_projections import collect_info_from_filename
from axoden.volume_projections import plot_summary_data, plot_signal_intensity_along_axis, process_image
from axoden.gui.streamlit_app.pdf_utils import fig2pdfpage, fig2stream, pdf2stream, pages2pdf, join_pdfs


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def init_session_state():
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
    print("initiated session state")


def invalidate_figure_cache():
    st.session_state.figure_cache = {}
    logger.info('invalidated figure cache')


def process_images(raw_files, pixel_size, is_masked, cache=None):
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


def process_image_single(raw_image, pixel_size, is_masked, cache=None):
    cache_key = (raw_image.file_id, pixel_size, is_masked) 
    if cache_key in cache:
        logger.info(f'process_image_single found cache for {cache_key}')
        return cache[cache_key]

    try:
        animal, brain_area = collect_info_from_filename(raw_image.name)
    except ValueError as e:
        logger.error(f'Error: {e}')
        st.warning(f"Error: {e}")
        st.stop()
        return None, None, None
    try:
        fig, _temp_, _temp_axis_ = process_image(raw_image, is_masked, pixel_size, animal=animal, brain_area=brain_area)
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