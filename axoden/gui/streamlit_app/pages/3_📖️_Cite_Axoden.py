import streamlit as st

st.set_page_config(page_title="Cite Axoden", page_icon="📖️")


def get_citation_ris():
    return """
TY  - JOUR
T1  - AxoDen: An Algorithm for the Automated Quantification of Axonal Density in defined Brain Regions
JF  - bioRxiv
DO  - 10.1101/2024.05.30.596687
SP  - 2024.05.30.596687
AU  - Sandoval Ortega, Raquel Adaia
AU  - Li, Emmy
AU  - Joseph, Oliver
AU  - Dufour, Pascal A.
AU  - Corder, Gregory
Y1  - 2024/01/01
UR  - http://biorxiv.org/content/early/2024/05/31/2024.05.30.596687.abstract
N2  - The rodent brain contains 70,000,000+ neurons interconnected via complex axonal circuits with varying architectures. Neural pathologies are often associated with anatomical changes in these axonal projections and synaptic connections. Notably, axonal density variations of local and long-range projections increase or decrease as a function of the strengthening or weakening, respectively, of the information flow between brain regions. Traditionally, histological quantification of axonal inputs relied on assessing the mean fluorescence intensity within a rectangle placed in the brain region-of-interest. Despite yielding valuable insights, this conventional method is notably susceptible to background fluorescence, post-acquisition adjustments, and inter-researcher variability. Additionally, it fails to account for the non-uniform innervation across brain regions, thus overlooking critical data such as innervation percentages and axonal distribution patterns. In response to these chal-lenges, we introduce AxoDen, an open-source semi-automated platform designed to increase the speed and rigor of axon quantifications for basic neuroscience discovery. AxoDen processes user-defined brain regions-of-interests incorporating dynamic thresholding of grayscales-transformed images to facilitate binarized pixel measurements. Thereby AxoDen segregates the image content into signal and non-signal categories, effectively eliminating background interference and enabling the exclusive measurement of fluorescence from axonal projections. AxoDen provides detailed and accurate representations of axonal density and spatial distribution. AxoDen&#039;s advanced yet user-friendly platform enhances the reliability and efficiency of axonal density analysis and facilitates access to unbiased high-quality data analysis with no technical background or coding experience required. AxoDen is freely available to everyone as a valuable neuroscience tool for dissecting axonal innervation patterns in precisely defined brain regions.Competing Interest StatementThe authors have declared no competing interest.
ER  - 
"""


def get_citation_bib():
    return """
@article {Sandoval Ortega2024.05.30.596687,
	author = {Sandoval Ortega, Raquel Adaia and Li, Emmy and Joseph, Oliver and Dufour, Pascal A. and Corder, Gregory},
	title = {AxoDen: An Algorithm for the Automated Quantification of Axonal Density in defined Brain Regions},
	elocation-id = {2024.05.30.596687},
	year = {2024},
	doi = {10.1101/2024.05.30.596687},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The rodent brain contains 70,000,000+ neurons interconnected via complex axonal circuits with varying architectures. Neural pathologies are often associated with anatomical changes in these axonal projections and synaptic connections. Notably, axonal density variations of local and long-range projections increase or decrease as a function of the strengthening or weakening, respectively, of the information flow between brain regions. Traditionally, histological quantification of axonal inputs relied on assessing the mean fluorescence intensity within a rectangle placed in the brain region-of-interest. Despite yielding valuable insights, this conventional method is notably susceptible to background fluorescence, post-acquisition adjustments, and inter-researcher variability. Additionally, it fails to account for the non-uniform innervation across brain regions, thus overlooking critical data such as innervation percentages and axonal distribution patterns. In response to these chal-lenges, we introduce AxoDen, an open-source semi-automated platform designed to increase the speed and rigor of axon quantifications for basic neuroscience discovery. AxoDen processes user-defined brain regions-of-interests incorporating dynamic thresholding of grayscales-transformed images to facilitate binarized pixel measurements. Thereby AxoDen segregates the image content into signal and non-signal categories, effectively eliminating background interference and enabling the exclusive measurement of fluorescence from axonal projections. AxoDen provides detailed and accurate representations of axonal density and spatial distribution. AxoDen{\\textquoteright}s advanced yet user-friendly platform enhances the reliability and efficiency of axonal density analysis and facilitates access to unbiased high-quality data analysis with no technical background or coding experience required. AxoDen is freely available to everyone as a valuable neuroscience tool for dissecting axonal innervation patterns in precisely defined brain regions.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/05/31/2024.05.30.596687},
	eprint = {https://www.biorxiv.org/content/early/2024/05/31/2024.05.30.596687.full.pdf},
	journal = {bioRxiv}
"""


def cite_app():
    citation_ris = get_citation_ris()
    citation_bib = get_citation_bib()
    st.markdown(
        """
# Citing AxoDen
Our manuscript is available on bioRxiv: https://www.biorxiv.org/content/10.1101/2024.05.30.596687  
pdf: https://www.biorxiv.org/content/10.1101/2024.05.30.596687.full.pdf

If you use this app in your research, cite it as follows:
"""
    )

    with st.container(border=True):
        st.markdown(
            """
##### AxoDen: An Algorithm for the Automated Quantification of Axonal Density in defined Brain Regions
###### Raquel Adaia Sandoval Ortega, Emmy Li, Oliver Joseph, Pascal A. Dufour, Gregory Corder
bioRxiv 2024.05.30.596687; doi: https://doi.org/10.1101/2024.05.30.596687

    """
        )

    st.markdown("#### Download Citation")
    st.write(
        "Download the citation in .ris (Mendeley/Zotero/Ref Manager) or .bib (BibTeX) format below."
        " You can find more citation formats on bioRxiv."
    )
    st.download_button(
        label="Download .ris",
        data=citation_ris,
        file_name="axoden_-an-algorithm-for-the-automated-quantification-of-axonal-density-in-defined-brain-regions.ris",
    )
    st.download_button(
        label="Download .bib",
        data=citation_bib,
        file_name="axoden_-an-algorithm-for-the-automated-quantification-of-axonal-density-in-defined-brain-regions.bib",
    )


cite_app()
