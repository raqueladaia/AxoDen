import logging
import os
import sys

import matplotlib.pyplot as plt
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from axoden.gui.streamlit_app.app_utils import (
    cached_plot_signal_intensity_along_axis,
    cached_plot_summary_data,
    get_brain_regions,
    get_figure_by_brain_region,
    process_images,
)
from axoden.gui.streamlit_app.pdf_utils import join_pdfs

MAX_IMAGES = 100
MAX_IMAGE_MB = 200
DEFAULT_PIXEL_SIZE = 0.75521

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def axo_den_app():
    """Main Streamlit application for AxoDen."""
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
        "Pixel Size (μm):",
        value=DEFAULT_PIXEL_SIZE,
        format="%f",
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
            f"You uploaded {len(raw_files)}, remaining images will not be used in the "
            "analysis!"
        )
        st.stop()

    is_masked = st.checkbox(
        "Images are masked (desired brain region are cropped out, "
        "backround fluorescence has values above 0, "
        "areas of the image lacking tissue have value 0)",
        value=True,
        key="is_masked",
    )

    import numpy as np

    total_memory = np.sum([f.size for f in raw_files]) / 1024**2
    if total_memory > MAX_IMAGE_MB:
        st.warning(
            f"We currently limit the uploaded images to {MAX_IMAGE_MB} MB "
            f"and {MAX_IMAGES} images.\n"
            f"The memory from your uploaded files is {total_memory} MB!\n"
            "Please reduce the number of images or the size of the images to continue. "
            "You can run the analysis in batches (e.g. per group or brain area)."
        )
        st.stop()

    (
        ctrl_figures,
        table_data,
        table_data_axis,
    ) = process_images(raw_files, pixel_size, is_masked)

    # plot table data results
    if table_data is not None:
        logger.info("Creating data section")
        st.header("Summary Data")

        fig, fig_stream = cached_plot_summary_data(
            table_data,
            project_name,
        )

        st.pyplot(fig)
        plt.close(fig)
        st.download_button(
            "Download figure as pdf", fig_stream, "axoden_summary_data.pdf"
        )
        st.dataframe(table_data)

    # plot table data by axis results
    if table_data_axis is not None:
        logger.info("Creating data axis section")
        st.header("Summary Data Axis")

        fig, fig_stream = cached_plot_signal_intensity_along_axis(
            project_name,
            table_data_axis,
            pixel_size,
        )
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download figure as pdf", fig_stream, "axoden_summary_data_axis.pdf"
        )
        st.dataframe(table_data_axis)

    logger.info("Creating control plots pdf")
    ctrl_plots_pdf = join_pdfs(ctrl_figures)

    brain_regions = get_brain_regions(raw_files)
    if brain_regions:
        logger.info("Creating control plots by brain area")
        st.header("Control Plots by Brain Area")

        tabs = st.tabs(brain_regions)

        brain_areas = table_data.brain_area.to_list()
        figure_dict = get_figure_by_brain_region(ctrl_figures, brain_areas)

        for tab, brain_region in zip(tabs, brain_regions):
            figures = figure_dict[brain_region]

            with tab:
                if len(figures) > 1:
                    indices = [str(x) for x in range(len(figures))]
                    tabs_brain_region = st.tabs(indices)
                    for i, tab_fig_nr in enumerate(tabs_brain_region):
                        with tab_fig_nr:
                            pdf_viewer(figures[i].getvalue(), key=f"{brain_region}_{i}")
                else:
                    pdf_viewer(figures[0].getvalue(), key=brain_region)

        st.download_button(
            "Download plots as pdf",
            ctrl_plots_pdf,
            "control_plots.pdf",
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Axoden App", page_icon="📊️")
    axo_den_app()
