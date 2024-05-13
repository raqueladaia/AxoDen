import streamlit as st

st.set_page_config(
    page_title="Cite Axoden",
    page_icon="📖️"
)

def cite_app():
    st.markdown('''
        # Citing AxoDen
        If you use this app in your research, cite it as follows:
        ```
        AxoDen: An Algorithm for Automated Quantification of Axonal Density in defined Brain Regions.
        Raquel Adaia Sandoval Ortega, Emmy Li, Oliver Joseph, Pascal Dufour, Gregory Corder
        BioRxiv
        May 2024
        doi: TODO
        ```
        TODO: add full citation in different formats
    ''')

cite_app()
