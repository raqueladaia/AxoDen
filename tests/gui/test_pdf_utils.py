from io import BytesIO

import pypdf
from gui_utils import (
    _sample_fig,
    _sample_pdf_page,
    _sample_pdf_stream,
    _sample_pdf_writer,
)

from axoden.gui.streamlit_app.pdf_utils import (
    fig2pdfpage,
    fig2stream,
    join_pdfs,
    pages2pdf,
    pdf2stream,
)


def test_join_pdfs():
    pdf_reader = _sample_pdf_stream()
    pdf_reader2 = _sample_pdf_stream()

    stream = join_pdfs([pdf_reader, pdf_reader2])
    assert isinstance(stream, BytesIO)

    pdf_reader_out = pypdf.PdfReader(stream)
    assert pdf_reader_out.get_num_pages() == 2


def test_pages2pdf():
    pdf_page = _sample_pdf_page()

    writer = pages2pdf([pdf_page])
    assert isinstance(writer, pypdf.PdfWriter)
    assert writer.get_num_pages() == 1

    writer = pages2pdf([pdf_page, pdf_page])
    assert isinstance(writer, pypdf.PdfWriter)
    assert writer.get_num_pages() == 2

    assert pages2pdf([]) is None


def test_pdf2stream():
    pdf_writer = _sample_pdf_writer()

    stream = pdf2stream(pdf_writer)
    assert isinstance(stream, BytesIO)

    # calling pdf2stream without an input should return None
    assert pdf2stream(None) is None


def test_fig2stream():
    fig, ax = _sample_fig()

    stream = fig2stream(fig)
    assert isinstance(stream, BytesIO)


def test_fig2pdfpage():
    fig, ax = _sample_fig()

    page = fig2pdfpage(fig)
    assert isinstance(page, pypdf._page.PageObject)
