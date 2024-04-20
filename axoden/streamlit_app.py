import os
from io import BytesIO
from pypdf import PdfReader, PdfWriter
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from typing import Iterable

from matplotlib import pyplot as plt

from axoden.volume_projections import collect_image_mask, compute_threshold, binarize_image, count_pixels
from axoden.volume_projections import compute_area, collect_info_from_filename, intensity_along_axis, generate_control_plot
from axoden.volume_projections import plot_summary_data, plot_signal_intensity_along_axis

MAX_IMAGES = 5
DEFAULT_PIXEL_SIZE = 0.75521


def img_to_pdf_page(img):
    # TODO: make pdf downloadable
    out_pdf = BytesIO()
    img.save(out_pdf, format="pdf")
    out_pdf = PdfReader(out_pdf)
    return out_pdf.pages[0]

def fig2pdfpage(fig):
    out_pdf = BytesIO()
    fig.savefig(out_pdf, format="pdf")
    out_pdf = PdfReader(out_pdf)
    return out_pdf.pages[0]


def fig2stream(fig):
    stream = BytesIO()
    fig.savefig(stream, format="pdf")
    return stream

    # stream = BytesIO()
    # return pdf.write_stream(stream)


@st.cache_data
def get_brain_regions(raw_files) -> Iterable[str]:
    brain_regions = set()
    for raw_file in raw_files:
        _, brain_region = collect_info_from_filename(raw_file.name)
        brain_regions.add(brain_region)
    return brain_regions


@st.cache_data
def process_image_single(raw_image, pixel_size, is_masked):
    animal, brain_area = collect_info_from_filename(raw_image.name)

    img, msk = collect_image_mask(raw_image, is_masked)
    msk_bool = msk.astype(bool)
    thr = compute_threshold(img[~msk_bool])
    img_bin = binarize_image(img, thr)

    [w, b, all] = count_pixels(img_bin[~msk_bool])
    area_w, area_b, area_img = compute_area(img_bin[~msk_bool], pixel_size)
    area_image_um = area_img / 1000

    info = {
        'animal': animal,
        'brain_area': brain_area, 
    }

    # Append the information to the DataFrame for the image
    _temp_ = {'animal': animal, 
                'brain_area': brain_area, 
                'pixels_signal': w, 
                'pixels_black': b, 
                'pixels_total': all,
                'threshold': thr, 
                'area_image': area_img,
                'area_signal': area_w,
                'area_black': area_b,
                'area_img_um': area_image_um}

    # Append the information to the DataFrame for the axis
    _temp_axis_ = {'animal': animal,
                    'brain_area': brain_area,
                    'signal_bin_x_ax': intensity_along_axis(img_bin, 'x'),
                    'signal_bin_y_ax': intensity_along_axis(img_bin, 'y'),
                    'signal_gray_x_ax': intensity_along_axis(img, 'x'),
                    'signal_gray_y_ax': intensity_along_axis(img, 'y')}

    # Generate control plot
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f'Animal {animal} | {brain_area} | Area: {area_image_um:.2f}\u03bcm\u00b2 | Threshold: {thr:.2f}', weight='bold')
    info_pie = {"labels": ['Area receiving\nprojections', 'Area without\nprojections'],
                "sizes": [area_w, area_b],
                "colors": ['white', 'grey']}
    generate_control_plot(img, img_bin, msk, pixel_size, info_pie)
    
    return fig, info, _temp_, _temp_axis_


@st.cache_data
def process_images(raw_files, pixel_size, is_masked):
    if not raw_files:
        return [], [], None, None

    print('process_images')

    # TODO: need to refactor the common code in volume_projections.
    # collect_data in volume_projections works on files, streamlit wors in memory...

    # Create an empty DataFrame to store the information
    table_data = pd.DataFrame()
    table_data_axis = pd.DataFrame()

    figures = []
    infos = []

    for raw_image in raw_files:
        fig, info, data_row, data_axis_row = process_image_single(raw_image, pixel_size, is_masked)
        
        if np.sum(table_data.shape) == 0:
            table_data = pd.DataFrame(columns=data_row.keys())
        table_data.loc[len(table_data)] = data_row

        if np.sum(table_data_axis.shape) == 0:
            table_data_axis = pd.DataFrame(columns=data_axis_row.keys())
        table_data_axis.loc[len(table_data_axis)] = data_axis_row

        figures += [fig]
        infos += [info]
        # TODO: not sure if table_data could be used as metadata for the figures. Do the rows stay constant? They should...

    # Compute the percentage of white pixels
    table_data['percent_signal'] = table_data['pixels_signal'] / table_data['pixels_total'] * 100

    return figures, infos, table_data, table_data_axis


@st.cache_data
def figures2pdf(_figures, metadata):
    print('figures2pdf', _figures)
    if not _figures:
        return None

    # TODO: we could sort the figures by brain area, and then plot them in that order
    pdf = PdfWriter()
    for fig in _figures:
        page = fig2pdfpage(fig)
        pdf.add_page(page)
    pdf_stream = BytesIO()
    pdf.write_stream(pdf_stream)
    print('pdf_stream', pdf_stream)
    return pdf_stream


@st.cache_data
def cached_plot_summary_data(table_data, project_name):
    return plot_summary_data(table_data, project_name)

# Streamlit App

if 'figures' not in st.session_state:
    st.session_state.figures = []

if 'figure_metadata' not in st.session_state:
    st.session_state.figure_metadata = [],

if 'ctrl_plots_pdf' not in st.session_state:
    st.session_state.ctrl_plots_pdf = None

if 'table_data' not in st.session_state:
    st.session_state.table_data = None

if 'table_data_axis' not in st.session_state:
    st.session_state.table_data_axis = None


st.title("AxoDen")


st.header("Input")
pixel_size = st.number_input("Pixel Size (um):", value=DEFAULT_PIXEL_SIZE, format='%f')  # Set the pixel size
st.text("Note:\nThe default pixel size is for the 20x Objective.\nPlease change it according to the objective used.\n\n4x Objective: 3.77396\n20x Objective: 0.75521\n")


raw_files = st.file_uploader("Upload image here. You can add more or remove them later.", type=["tif"], accept_multiple_files=True)
if len(raw_files) > MAX_IMAGES:
    st.warning(f"This application is limited to using {MAX_IMAGES} images concurrently. You uploaded {len(raw_files)}, remaining images will not be used in the analysis!")
    raw_files = raw_files[:MAX_IMAGES]

is_masked = not st.checkbox("Images are masked (have desired brain region cropped out, backround is at value 0)", value=True)

    # figures, table_data, table_data_axis = process_images(raw_files, pixel_size, is_masked)

print('about to run process_images')
(
    st.session_state.figures,
    st.session_state.figure_metadata,
    st.session_state.table_data,
    st.session_state.table_data_axis,
 ) = process_images(raw_files, pixel_size, is_masked)

print('about to run figures2pdf')
st.session_state.ctrl_plots_pdf = figures2pdf(st.session_state.figures, st.session_state.figure_metadata)

print('about to run get_brain_regions')
brain_regions = get_brain_regions(raw_files)
if brain_regions:
    st.header("Control Plots by Brain Area")
    tabs = st.tabs(brain_regions)

    tab_by_region = {r: t for r, t in zip(brain_regions, tabs)}

    print('plotting...')
    for fig, fig_info in zip(st.session_state.figures, st.session_state.figure_metadata):
        tab_by_region[fig_info['brain_area']].pyplot(fig)

    st.download_button("Download figure as pdf", st.session_state.ctrl_plots_pdf, "data.pdf", mime="application/pdf", key="download_data_pdf")

# for fig in st.session_state.figures:
#     st.pyplot(fig)

if st.session_state.table_data is not None:
    st.header("Data")
    
    project_name = "test project"
    fig = plot_summary_data(st.session_state.table_data, project_name)
    st.pyplot(fig)
    st.download_button("Download figure as pdf", fig2stream(fig), "data.pdf", mime="application/pdf", key="download_data_axis_pdf")
    st.dataframe(st.session_state.table_data)

if st.session_state.table_data_axis is not None:
    st.header("Data Axis")

    project_name = "test project"
    fig = plot_signal_intensity_along_axis(project_name, st.session_state.table_data_axis, pixel_size)
    st.pyplot(fig)
    st.download_button("Download figure as pdf", fig2stream(fig), "data_axis.pdf", "Download pdf")
    st.dataframe(st.session_state.table_data_axis)
