
import streamlit as st
from streamlit.components.v1 import html



def mainBase():
    my_js = """
    <script type="text/javascript" src="https://base64.ai/scripts/nocode.js?key=YOUR_FORM_ID"></script>
    """

    my_html = f"<script>{my_js}</script>"

    st.title("Base64 Deneme")
    html(my_js)
