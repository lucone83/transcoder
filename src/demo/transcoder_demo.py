import json
import os
import streamlit as st
import httpx

TRANSCODER_API_ENDPOINT = f'{os.environ["TRANSCODER_API_HOST"]}/translate'


@st.cache
def get_translation(data):
    try:
        resp = httpx.post(
            TRANSCODER_API_ENDPOINT,
            data=json.dumps(data),
            timeout=30.0
        )
        return json.loads(resp.text)["translation"]
    except Exception as e:
        raise


def set_further_styles():
    st.markdown("""
        <style>
            h1 { text-align: center; }
            textarea { font-family: Courier !important; font-size: 14px !important; }
            .image-container { display: block !important; text-align: center; }
        </style>

    """, unsafe_allow_html=True)


def run():
    set_further_styles()

    # TITLE
    st.title('TransCoder API demo')

    # SIDEBAR
    st.sidebar.header("Optional parameters")
    st.sidebar.markdown("#### Beam size")
    beam_size = st.sidebar.slider(
        "Number of best partial solutions to evaluate",
        min_value=1,
        max_value=10,
        value=1,
    )

    # PARAMETERS
    st.markdown("#### Languages")
    source_lang = st.selectbox("Choose source language", ("cpp", "java", "python"))
    target_lang = st.selectbox("Choose target language", ("cpp", "java", "python"))
    st.markdown("---------------------")
    code_str = st.text_area("Code to translate", height=800, value='')
    data = {
        'code_str': json.dumps(code_str),
        'src_lang': source_lang,
        'tgt_lang': target_lang,
        'beam_size': beam_size
    }

    # BUTTON AND RESULT BOX
    if st.button("Translate code!"):
        translations = get_translation(data)
        for i, translation in enumerate(translations, 1):
            st.markdown(f"Translation **#{i}**")
            st.code(translation, language=target_lang)


if __name__ == "__main__":
    run()
