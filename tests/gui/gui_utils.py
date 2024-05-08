import os
import pypdf
from uuid import uuid4
import matplotlib.pyplot as plt
from streamlit.testing.v1 import AppTest
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs


def _uploaded_file(file_name, test_data=False):
    base_dir = "tests/data" if test_data else "test_images"
    with open (os.path.join(base_dir, file_name), "rb") as f:
        raw_file = f.read()

    file_id = str(uuid4())

    record = UploadedFileRec(file_id=file_id, name=file_name, type='image/tiff', data=raw_file)
    upload_url = f"/_stcore/upload_file/{uuid4()}/{file_id}"
    file_urls = FileURLs(upload_url=upload_url, delete_url=upload_url)

    return UploadedFile(record, file_urls)


def _streamlit_app():
    return AppTest.from_file("axoden/gui/streamlit_app/1_📊️_Axoden.py").run()


def _sample_pdf_page():
    pdf_reader = _sample_pdf_reader()
    pdf_page = pdf_reader.pages[0]
    return pdf_page


def _sample_pdf_writer():
    pdf_reader = _sample_pdf_reader()
    pdf_writer = pypdf.PdfWriter(pdf_reader)
    return pdf_writer


def _sample_pdf_reader():
    pdf_test_file = "tests/data/sample.pdf"
    pdf_reader = pypdf.PdfReader(pdf_test_file)
    return pdf_reader


def _sample_fig():
    fig, ax = plt.subplots(1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    return fig, ax
