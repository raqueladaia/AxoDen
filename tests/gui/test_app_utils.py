import pytest
import pypdf
import pandas as pd

from axoden.gui.streamlit_app.app_utils import (
    process_image_single, process_images, invalidate_figure_cache, get_brain_regions,
    get_figure_by_brain_region
)

from gui_utils import _streamlit_app, _uploaded_file
from gui_utils import _sample_fig, _sample_pdf_page, _sample_pdf_reader, _sample_pdf_writer

from unittest.mock import patch


def test_invalidate_figure_cache():
    with patch('axoden.gui.streamlit_app.app_utils.st.session_state') as mock_cache:
        invalidate_figure_cache()
        assert mock_cache.figure_cache == {}

def test_get_brain_regions():
    uploaded_file1 = _uploaded_file("2111_Insula-GFP.tif")
    uploaded_file2 = _uploaded_file("3518_dsRed.tif")
    brain_regions = get_brain_regions([uploaded_file1, uploaded_file2])

    assert len(brain_regions) == 2
    assert "Insula-GFP" in brain_regions
    assert "dsRed" in brain_regions

def test_get_figure_by_brain_region():
    pdf_1 = _sample_pdf_writer()
    pdf_2= _sample_pdf_writer()
    brain_regions = ["region1", "region1", "region2"]
    dict_out = get_figure_by_brain_region([pdf_1, pdf_2, pdf_1], brain_regions)
    assert len(dict_out) == 2
    assert "region1" in dict_out
    assert "region2" in dict_out
    assert dict_out["region1"] == [pdf_1, pdf_2]
    assert dict_out["region2"] == [pdf_1]

def test_process_images():
    at = _streamlit_app()

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


def test_process_image_single_cache():
    file_name = "2111_Insula-GFP.tif"
    pixel_size = 0.75521
    is_masked = True
    uploaded_file = _uploaded_file(file_name)

    cache_key = (uploaded_file.file_id, pixel_size, is_masked) 
    cache_value = "fake cache value"
    cache = {cache_key: cache_value}

    output_value = process_image_single(uploaded_file, pixel_size, is_masked, cache=cache)
    assert output_value == cache_value


@patch('axoden.gui.streamlit_app.app_utils.st.warning')
@patch('axoden.gui.streamlit_app.app_utils.st.stop')
def test_process_image_single_error(warning_mock, stop_mock):
    file_name = "2111_Insula-GFP.tif"
    pixel_size = 0.75521
    is_masked = True
    uploaded_file = _uploaded_file(file_name)
    uploaded_file.name = "invalid.tif"

    process_image_single(uploaded_file, pixel_size, is_masked)

    # If we get an invalid image name, the streamlit app should stop and show a warning
    assert warning_mock.called
    assert stop_mock.called

    uploaded_file = _uploaded_file("invalid/not_tif.tif", test_data=True)

    warning_mock.reset_mock()
    stop_mock.reset_mock()
    process_image_single(uploaded_file, pixel_size, is_masked)
    # If we get valid image name, but the file can't be read, we should stop and warn
    assert warning_mock.called
    assert stop_mock.called

