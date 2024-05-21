from unittest.mock import patch

import numpy as np
from gui_utils import _streamlit_app, _uploaded_file

from axoden.gui.streamlit_app.app_utils import process_images


def test_app_init():
    at = _streamlit_app()

    # Assert default pixel size is set
    assert at.number_input("pixel_size").value > 0.0

    # Assert default is_masked value is set to True
    assert at.checkbox("is_masked").value


def test_app_upload():
    at = _streamlit_app()

    file_uploader = at.get("file_uploader")[0].proto
    assert all([x in [".tiff", ".tif"] for x in file_uploader.type])
    # TODO: once streamlit supports file_uploader, test full file upload functionality

    pixel_size = 0.75521
    is_masked = True
    file_name = "745_TH-PL.tif"

    uploaded_file = _uploaded_file(file_name)

    # first we get what process_images would return
    figs, data, data_axis = process_images([uploaded_file], pixel_size, is_masked)

    # then we mock the actual call, so even though the file_uploader does have any
    # images, it returns a result this allows us to simulate the upload of an image
    with patch(
        "axoden.gui.streamlit_app.app_utils.process_images"
    ) as mock_process_images:
        mock_process_images.return_value = (figs, data, data_axis)
        at.run()

    # assert the dataframes in streamlit exist and have the correct data
    assert len(at.dataframe) == 2
    assert np.all(at.dataframe[0].value == data)
    # cannot directly compare dataframes as they contain np.arrays
    assert np.all(at.dataframe[1].value.columns == data_axis.columns)
    assert at.dataframe[1].value.shape == data_axis.shape

    # there should be two download buttons
    assert len(at.get("download_button")) == 2

    # there should be two images (from calling pyplot)
    assert len(at.get("imgs")) == 2
