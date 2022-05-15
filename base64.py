

import streamlit as st


def mainBase():
    my_js = """
  
    type = "text/javascript"
    src = "https://base64.ai/scripts/nocode.js?key=fe0a21c5-a657-40a8-b0ba-002ada4cf656" > 
    """

    my_html = "<script>{my_js}</script>"

    st.title("Javascript example")
    st.components.v1.html(my_js)
