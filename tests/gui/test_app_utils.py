from io import BytesIO
from unittest.mock import patch

import pandas as pd
from gui_utils import (
    _sample_pdf_writer,
    # _streamlit_app,
    _uploaded_file,
)

from axoden.gui.streamlit_app.app_utils import (
    get_brain_regions,
    get_figure_by_brain_region,
    process_image_single_cached,
    process_images,
)


def test_get_brain_regions():
    uploaded_file1 = _uploaded_file("745_TH-PL.tif")
    uploaded_file2 = _uploaded_file("746_TH-CL.tif")
    brain_regions = get_brain_regions([uploaded_file1, uploaded_file2])

    assert len(brain_regions) == 2
    assert "TH-PL" in brain_regions
    assert "TH-CL" in brain_regions


def test_get_figure_by_brain_region():
    pdf_1 = _sample_pdf_writer()
    pdf_2 = _sample_pdf_writer()
    brain_regions = ["region1", "region1", "region2"]
    dict_out = get_figure_by_brain_region([pdf_1, pdf_2, pdf_1], brain_regions)
    assert len(dict_out) == 2
    assert "region1" in dict_out
    assert "region2" in dict_out
    assert dict_out["region1"] == [pdf_1, pdf_2]
    assert dict_out["region2"] == [pdf_1]


def test_process_images():
    files = [
        {"animal": "745", "brain_area": "TH-PL", "file_name": "745_TH-PL.tif"},
        {"animal": "746", "brain_area": "TH-CL", "file_name": "746_TH-CL.tif"},
    ]
    n_uploades = len(files)

    uploaded_files = []
    for file in files:
        uploaded_file = _uploaded_file(file["file_name"])
        uploaded_files.append(uploaded_file)

    figs, data, data_axis = process_images(uploaded_files, 0.75521, True)

    assert len(figs) == n_uploades

    assert isinstance(data, pd.DataFrame)
    assert isinstance(data_axis, pd.DataFrame)

    for file in files:
        assert file["animal"] in data_axis["animal"].values
        assert file["animal"] in data["animal"].values
        assert file["brain_area"] in data["brain_area"].values
        assert file["brain_area"] in data_axis["brain_area"].values


def test_process_image_single_cached():
    animal = "745"
    brain_area = "TH-PL"
    file_name = "745_TH-PL.tif"

    uploaded_file = _uploaded_file(file_name)

    fig_pdf, data, data_axis = process_image_single_cached(uploaded_file, 0.75521, True)
    assert isinstance(fig_pdf, BytesIO)
    assert isinstance(data, dict)
    assert isinstance(data_axis, dict)
    assert animal == data["animal"]
    assert brain_area == data["brain_area"]
    assert animal == data_axis["animal"]
    assert brain_area == data_axis["brain_area"]


@patch("axoden.gui.streamlit_app.app_utils.st.warning")
@patch("axoden.gui.streamlit_app.app_utils.st.stop")
def test_process_image_single_cached_error(warning_mock, stop_mock):
    file_name = "745_TH-PL.tif"
    pixel_size = 0.75521
    is_masked = True
    uploaded_file = _uploaded_file(file_name)
    uploaded_file.name = "invalid.tif"

    process_image_single_cached(uploaded_file, pixel_size, is_masked)

    # If we get an invalid image name, the streamlit app should stop and show a warning
    assert warning_mock.called
    assert stop_mock.called

    uploaded_file = _uploaded_file("invalid/not_tif.tif", test_data=True)

    warning_mock.reset_mock()
    stop_mock.reset_mock()
    process_image_single_cached(uploaded_file, pixel_size, is_masked)
    # If we get valid image name, but the file can't be read, we should stop and warn
    assert warning_mock.called
    assert stop_mock.called
