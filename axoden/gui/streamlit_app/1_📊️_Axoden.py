import logging
import os
import sys

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from axoden.gui.streamlit_app.app_utils import (
    cached_plot_signal_intensity_along_axis,
    cached_plot_summary_data,
    get_brain_regions,
    get_figure_by_brain_region,
    init_session_state,
    invalidate_figure_cache,
    process_images,
)
from axoden.gui.streamlit_app.pdf_utils import join_pdfs, pdf2stream

MAX_IMAGES = 200  # TODO: decide on file upload limit
DEFAULT_PIXEL_SIZE = 0.75521

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def axo_den_app():
    """Main Streamlit application for AxoDen."""
    init_session_state()

    st.title("AxoDen")

    with st.container(border=True):
        st.write(
            "If you're new to AxoDen, see the tutorial and learn how to use AxoDen:"
        )
        st.page_link(
            "pages/2_❔️_Tutorial_&_How_To.py",
            label="Tutorial: How To Use AxoDen",
            icon="❔️",
        )

        st.write("If you use the app for your research, cite it using the following:")
        st.page_link("pages/3_📖️_Cite_Axoden.py", label="Cite AxoDen", icon="📖️")

    st.header("Input")
    project_name = st.text_input(label="Project Name", value="AxoDen Analysis")
    pixel_size = st.number_input(
        "Pixel Size (um):",
        value=DEFAULT_PIXEL_SIZE,
        format="%f",
        on_change=invalidate_figure_cache,
        key="pixel_size",
    )  # Set the pixel size
    st.text(
        "Note:\nThe default pixel size is for the 20x Objective of the Keyence BZ-810X "
        "series.\n"
        "Please change it according to the objective used.\n\n4x Objective: 3.77396\n"
        "20x Objective: 0.75521\n"
    )

    raw_files = st.file_uploader(
        "Upload image here. You can add more or remove them later.",
        type=["tif"],
        accept_multiple_files=True,
    )
    if len(raw_files) > MAX_IMAGES:
        st.warning(
            f"This application is limited to using {MAX_IMAGES} images concurrently. "
            "You uploaded {len(raw_files)}, remaining images will not be used in the "
            "analysis!"
        )
        raw_files = raw_files[:MAX_IMAGES]

    is_masked = st.checkbox(
        "Images are masked (desired brain region are cropped out, "
        "backround fluorescence has values above 0,\n"
        "areas of the image lacking tissue have value 0)",
        value=True,
        on_change=invalidate_figure_cache,
        key="is_masked",
    )

    (
        st.session_state.figures,
        st.session_state.table_data,
        st.session_state.table_data_axis,
    ) = process_images(
        raw_files, pixel_size, is_masked, cache=st.session_state.figure_cache
    )

    # plot table data results
    if st.session_state.table_data is not None:
        logger.info("Creating data section")
        st.header("Summary Data")

        fig, fig_stream = cached_plot_summary_data(
            st.session_state.table_data, project_name
        )

        st.pyplot(fig)
        st.download_button(
            "Download figure as pdf", fig_stream, "axoden_summary_data.pdf"
        )
        st.dataframe(st.session_state.table_data)

    # plot table data by axis results
    if st.session_state.table_data_axis is not None:
        logger.info("Creating data axis section")
        st.header("Summary Data Axis")

        fig, fig_stream = cached_plot_signal_intensity_along_axis(
            project_name, st.session_state.table_data_axis, pixel_size
        )
        st.pyplot(fig)

        st.download_button(
            "Download figure as pdf", fig_stream, "axoden_summary_data_axis.pdf"
        )
        st.dataframe(st.session_state.table_data_axis)

    logger.info("Creating control plots pdf")
    st.session_state.ctrl_plots_pdf = join_pdfs(st.session_state.figures)

    brain_regions = get_brain_regions(raw_files)
    if brain_regions:
        logger.info("Creating control plots by brain area")
        st.header("Control Plots by Brain Area")

        tabs = st.tabs(brain_regions)

        brain_areas = st.session_state.table_data.brain_area.to_list()
        figure_dict = get_figure_by_brain_region(st.session_state.figures, brain_areas)

        for tab, brain_region in zip(tabs, brain_regions):
            figures = figure_dict[brain_region]

            with tab:
                if len(figures) > 1:
                    indices = [str(x) for x in range(len(figures))]
                    tabs_brain_region = st.tabs(indices)
                    for i, tab_fig_nr in enumerate(tabs_brain_region):
                        with tab_fig_nr:
                            pdf_figure = pdf2stream(figures[i]).getvalue()
                            pdf_viewer(pdf_figure, key=f"{brain_region}_{i}")
                else:
                    pdf_figure = pdf2stream(figures[0]).getvalue()
                    pdf_viewer(pdf_figure, key=brain_region)

        st.download_button(
            "Download plots as pdf",
            st.session_state.ctrl_plots_pdf,
            "control_plots.pdf",
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Axoden App", page_icon="📊️")
    axo_den_app()
