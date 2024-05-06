import pytest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pypdf
from io import BytesIO
from uuid import uuid4

from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs

from axoden.gui.streamlit_app.pdf_utils import fig2pdfpage, fig2stream, pdf2stream, pages2pdf, join_pdfs
from axoden.gui.streamlit_app.app_utils import (
    init_session_state, invalidate_figure_cache, process_images, process_image_single,
)


def _uploaded_file(file_name):
    with open (f"sample_images/{file_name}", "rb") as f:
        raw_file = f.read()

    file_id = str(uuid4())

    record = UploadedFileRec(file_id=file_id, name=file_name, type='image/tiff', data=raw_file)
    upload_url = f"/_stcore/upload_file/{uuid4()}/{file_id}"
    file_urls = FileURLs(upload_url=upload_url, delete_url=upload_url)

    return UploadedFile(record, file_urls)


def _streamlit_app():
    return AppTest.from_file("axoden/gui/streamlit_app/1_📊️_Axoden.py").run()


def _sample_fig():
    fig, ax = plt.subplots(1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    return fig, ax


def _sample_pdf_reader():
    pdf_test_file = "tests/data/sample.pdf"
    pdf_reader = pypdf.PdfReader(pdf_test_file)
    return pdf_reader


def _sample_pdf_writer():
    pdf_reader = _sample_pdf_reader()
    pdf_writer = pypdf.PdfWriter(pdf_reader)
    return pdf_writer


def _sample_pdf_page():
    pdf_reader = _sample_pdf_reader()
    pdf_page = pdf_reader.pages[0]
    return pdf_page


def test_fig2pdfpage():
    fig, ax = _sample_fig()

    page = fig2pdfpage(fig)
    assert isinstance(page, pypdf._page.PageObject)


def test_fig2stream():
    fig, ax = _sample_fig()

    stream = fig2stream(fig)
    assert isinstance(stream, BytesIO)

def test_pdf2stream():
    pdf_writer = _sample_pdf_writer()
    
    stream = pdf2stream(pdf_writer)
    assert isinstance(stream, BytesIO)

def test_pages2pdf():
    pdf_page = _sample_pdf_page()

    writer = pages2pdf([pdf_page])
    assert isinstance(writer, pypdf.PdfWriter)
    assert writer.get_num_pages() == 1

    writer = pages2pdf([pdf_page, pdf_page])
    assert isinstance(writer, pypdf.PdfWriter)
    assert writer.get_num_pages() == 2

def test_join_pdfs():
    pdf_reader = _sample_pdf_reader()
    pdf_reader = _sample_pdf_reader()

    stream = join_pdfs([pdf_reader, pdf_reader])
    assert isinstance(stream, BytesIO)

    pdf_reader_out = pypdf.PdfReader(stream)
    assert pdf_reader_out.get_num_pages() == 2


def test_app_init():
    at = _streamlit_app()

    # Assert default pixel size is set
    assert at.number_input("pixel_size").value > 0.0

    # Assert default is_masked value is set to True
    assert at.checkbox("is_masked").value == True



def test_app_upload():

    at = _streamlit_app()

    file_uploader = at.get('file_uploader')[0].proto
    assert all([x in ['.tiff', '.tif'] for x in file_uploader.type])
    # TODO: once streamlit supports file_uploader, test full file upload functionality

    pixel_size = 0.75521
    is_masked = True
    file_name = "2111_Insula-GFP.tif"

    uploaded_file = _uploaded_file(file_name)

    # first we get what process_images would return
    figs, data, data_axis = process_images([uploaded_file], pixel_size, is_masked, cache=at.session_state.figure_cache)

    # then we mock the actual call, so even though the file_uploader does have any images, it returns a result
    # this allows us to simulate the upload of an image
    with patch('axoden.gui.streamlit_app.app_utils.process_images') as mock_process_images:
        mock_process_images.return_value = (figs, data, data_axis)
        at.run()

    # assert the dataframes in streamlit exist and have the correct data
    assert len(at.dataframe) == 2
    assert np.all(at.dataframe[0].value == data)
    # cannot directly compare dataframes as they contain np.arrays
    assert np.all(at.dataframe[1].value.columns == data_axis.columns)
    assert at.dataframe[1].value.shape == data_axis.shape

    # assert the pdf control plots to download exists
    assert at.session_state.ctrl_plots_pdf is not None

    # there should be two download buttons
    assert len(at.get("download_button")) == 2

    # there should be two images (from calling pyplot)
    assert len(at.get("imgs")) == 2


def test_process_image_single():
    at = _streamlit_app()

    animal = "2111"
    brain_area = "Insula-GFP"
    file_name = "2111_Insula-GFP.tif"

    pixel_size = 0.75521
    is_masked = True

    uploaded_file = _uploaded_file(file_name)

    assert len(at.session_state.figure_cache) == 0
    fig_pdf, data, data_axis = process_image_single(uploaded_file, 0.75521, True, cache=at.session_state.figure_cache)
    assert isinstance(fig_pdf, pypdf.PdfWriter)
    assert isinstance(data, dict)
    assert isinstance(data_axis, dict)
    assert animal == data['animal']
    assert brain_area == data['brain_area']
    assert animal == data_axis['animal']
    assert brain_area == data_axis['brain_area']

    assert len(at.session_state.figure_cache) == 1
    assert (uploaded_file.file_id, pixel_size, is_masked) in at.session_state.figure_cache


def test_process_images():
    at = _streamlit_app()

    pixel_size = 0.75521
    is_masked = True

    files = [
        {"animal": "2111", "brain_area": "Insula-GFP", "file_name": "2111_Insula-GFP.tif"},
        {"animal": "3518", "brain_area": "dsRed", "file_name": "3518_dsRed.tif"},
    ]
    n_uploades = len(files)

    uploaded_files = []
    for file in files:
        uploaded_file = _uploaded_file(file['file_name'])
        uploaded_files.append(uploaded_file)

    figs, data, data_axis = process_images(uploaded_files, 0.75521, True, cache=at.session_state.figure_cache)

    assert len(figs) == n_uploades

    assert isinstance(data, pd.DataFrame)
    assert isinstance(data_axis, pd.DataFrame)

    for file in files:
        assert file["animal"] in data_axis["animal"].values
        assert file["animal"] in data["animal"].values
        assert file["brain_area"] in data["brain_area"].values
        assert file["brain_area"] in data_axis["brain_area"].values



    