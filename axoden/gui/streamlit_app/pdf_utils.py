from io import BytesIO
from typing import List

from matplotlib.figure import Figure
from pypdf import PdfReader, PdfWriter
from pypdf._page import PageObject


def fig2pdfpage(fig: Figure) -> PageObject:
    """Converts a matplotlib figure to a PyPDF PageObject.

    Args:
        fig (Figure): matplotlib figure

    Returns:
        PageObject: PyPDF PageObject containing the input matplotlib figure
    """
    out_pdf = BytesIO()
    fig.savefig(out_pdf, format="pdf")
    out_pdf = PdfReader(out_pdf)
    return out_pdf.pages[0]


def fig2stream(fig: Figure) -> BytesIO:
    """Converts a matplotlib figure to a BytesIO stream.

    The output stream can be directly used as a download in Streamlit.

    Args:
        fig (Figure): matplotlib figure

    Returns:
        BytesIO: BytesIO stream of the input matplotlib figure

    """
    stream = BytesIO()
    fig.savefig(stream, format="pdf")
    return stream


def pdf2stream(pdf: PdfWriter) -> BytesIO:
    """Converts a PyPDF PdfReader to a BytesIO stream.

    The output stream can be directly used as a download in Streamlit.

    Args:
        pdf (PdfWriter): PyPDF PdfWriter object

    Returns:
        BytesIO: BytesIO stream of the input PyPDF PdfReader object
    """
    if not pdf:
        return None

    pdf_stream = BytesIO()
    pdf.write_stream(pdf_stream)
    return pdf_stream


def pages2pdf(pages: List[PageObject]) -> PdfWriter:
    """Converts a list of PyPDF Pages to a single pdf.

    Args:
        pages (List[PageObject]): list of pages

    Returns:
        PdfWriter: PyPDF PdfWriter object containing all input pages
    """
    if not pages:
        return None

    pdf = PdfWriter()
    for page in pages:
        pdf.add_page(page)
    return pdf


def join_pdfs(pdf_streams: List[BytesIO]) -> BytesIO:
    """Joins all first pages of multiple pdfs into a single pdf.

    This function only cares about the first page.
    These types of pdfs are used when figures are converted to pdfs containing only one
    page. The output BytesIO stream can be directly used as a download in Streamlit.

    Args:
        pdfs (List[BytesIO]): list of pdfs as streams

    Returns:
        BytesIO: BytesIO stream of the joined pdf

    """
    pdf = PdfWriter()
    for p_stream in pdf_streams:
        p = PdfReader(p_stream)

        page = p.pages[0]
        pdf.add_page(page)

    out_pdf_stream = BytesIO()
    pdf.write_stream(out_pdf_stream)
    return out_pdf_stream
