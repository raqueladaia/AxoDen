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
        Some
        Citation
        Here
        ```
        TODO: add full citation in different formats
    ''')

cite_app()
