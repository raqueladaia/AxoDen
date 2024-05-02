import os
import streamlit as st

import logging

import numpy as np
import pandas as pd
from typing import Iterable
from PIL import UnidentifiedImageError

from matplotlib import pyplot as plt
from streamlit_pdf_viewer import pdf_viewer

import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from axoden.volume_projections import collect_info_from_filename
from axoden.volume_projections import plot_summary_data, plot_signal_intensity_along_axis, process_image
from axoden.streamlit_app.pdf_utils import fig2pdfpage, fig2stream, pdf2stream, pages2pdf, join_pdfs

MAX_IMAGES = 200  # TODO: decide on file upload limit
DEFAULT_PIXEL_SIZE = 0.75521

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def invalidate_figure_cache():
    st.session_state.figure_cache = {}
    logger.info('invalidated figure cache')


@st.cache_data
def get_brain_regions(raw_files) -> Iterable[str]:
    logger.info('get_brain_regions')
    brain_regions = set()
    for raw_file in raw_files:
        _, brain_region = collect_info_from_filename(raw_file.name)
        brain_regions.add(brain_region)
    return list(brain_regions)


def process_image_single(raw_image, pixel_size, is_masked):
    cache_key = (raw_image.file_id, pixel_size, is_masked) 
    if cache_key in st.session_state.figure_cache:
        logger.info(f'process_image_single found cache for {cache_key}')
        return st.session_state.figure_cache[cache_key]

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

    st.session_state.figure_cache[(raw_image.file_id, pixel_size, is_masked)] = (pdf_fig,  _temp_, _temp_axis_)
    logger.info(f'process_image_single created cache for {cache_key}')

    return pdf_fig, _temp_, _temp_axis_


def process_images(raw_files, pixel_size, is_masked):
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
        fig, data_row, data_axis_row = process_image_single(raw_image, pixel_size, is_masked)
        
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


def cached_plot_summary_data(table_data, project_name):
    fig = plot_summary_data(table_data, project_name)
    fig_stream = fig2stream(fig)
    return fig, fig_stream


def cached_plot_signal_intensity_along_axis(project_name, table_data_axis, pixel_size):
    logger.info('creating signal intensity along axis')
    fig = plot_signal_intensity_along_axis(project_name, table_data_axis, pixel_size)
    fig_stream = fig2stream(fig)
    return fig, fig_stream


def get_figure_by_brain_region(_figures, brain_areas):
    logger.info('get_figure_by_brain_region')
    figures_out = {}
    for fig, brain_area in zip(_figures, brain_areas):
        if brain_area not in figures_out:
            figures_out[brain_area] = []

        figures_out[brain_area].append(fig)

    return figures_out




def axo_den_app():
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

    st.title("AxoDen")

    with st.container(border=True):
        st.write("If you're new to AxoDen, see the tutorial and learn how to use AxoDen:")
        st.page_link("pages/2_❔️_Tutorial_&_How_To.py", label="Tutorial: How To Use AxoDen", icon="❔️")

        st.write("If you use the app for your research, cite it using the following:")
        st.page_link("pages/3_📖️_Cite_Axoden.py", label="Cite AxoDen", icon="📖️")

    st.header("Input")
    project_name = st.text_input(label="Project Name", value="AxoDen Analysis")
    pixel_size = st.number_input("Pixel Size (um):", value=DEFAULT_PIXEL_SIZE, format='%f', on_change=invalidate_figure_cache)  # Set the pixel size
    st.text("Note:\nThe default pixel size is for the 20x Objective.\nPlease change it according to the objective used.\n\n4x Objective: 3.77396\n20x Objective: 0.75521\n")


    raw_files = st.file_uploader("Upload image here. You can add more or remove them later.", type=["tif"], accept_multiple_files=True)
    if len(raw_files) > MAX_IMAGES:
        st.warning(f"This application is limited to using {MAX_IMAGES} images concurrently. You uploaded {len(raw_files)}, remaining images will not be used in the analysis!")
        raw_files = raw_files[:MAX_IMAGES]

    is_masked = st.checkbox("Images are masked (desired brain region are cropped out, backround is at value 0)", value=True, on_change=invalidate_figure_cache)

    (
        st.session_state.figures,
        st.session_state.table_data,
        st.session_state.table_data_axis,
    ) = process_images(raw_files, pixel_size, is_masked)

    # plot table data results
    if st.session_state.table_data is not None:
        logger.info('Creating data section')
        st.header("Data")
    
        fig, fig_stream = cached_plot_summary_data(st.session_state.table_data, project_name)
    
        st.pyplot(fig)
        st.download_button("Download figure as pdf", fig_stream, "data.pdf")
        st.dataframe(st.session_state.table_data)

    # plot table data by axis results
    if st.session_state.table_data_axis is not None:
        logger.info('Creating data axis section')
        st.header("Data Axis")

        fig, fig_stream = cached_plot_signal_intensity_along_axis(project_name, st.session_state.table_data_axis, pixel_size)
        st.pyplot(fig)

        st.download_button("Download figure as pdf", fig_stream, "data_axis.pdf")
        st.dataframe(st.session_state.table_data_axis)

    logger.info('Creating control plots pdf')
    st.session_state.ctrl_plots_pdf = join_pdfs(st.session_state.figures)

    brain_regions = get_brain_regions(raw_files)
    if brain_regions:
        logger.info('Creating control plots by brain area')
        st.header("Control Plots by Brain Area")

        tabs = st.tabs(brain_regions)

        brain_areas = st.session_state.table_data.brain_area.to_list()
        figure_dict = get_figure_by_brain_region(st.session_state.figures, brain_areas)

        for tab, brain_region in zip(tabs, brain_regions):
            figures = figure_dict[brain_region]

            with tab:
                if len(figures) > 1:
                    indices = [str(x) for x in range(len(figures)) ]
                    tabs_brain_region = st.tabs(indices)
                    for i, tab_fig_nr in enumerate(tabs_brain_region):
                        with tab_fig_nr:
                            pdf_figure = pdf2stream(figures[i]).getvalue()
                            pdf_viewer(pdf_figure, key=f"{brain_region}_{i}")
                else:
                    pdf_figure = pdf2stream(figures[0]).getvalue()
                    pdf_viewer(pdf_figure, key=brain_region)

        st.download_button("Download plots as pdf", st.session_state.ctrl_plots_pdf, "control_plots.pdf")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Axoden App",
        page_icon="📊️"
    )
    axo_den_app()
