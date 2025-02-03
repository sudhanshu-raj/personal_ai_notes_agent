import streamlit as st


def confirm_dialog(title, message):
    st.session_state[f"confirm_{title}"] = True
    with st.container():
        st.warning(message)
        col1, col2 = st.columns([1, 2])
        if col1.button("Confirm"):
            st.session_state[f"confirm_{title}"] = False
            return True
        if col2.button("Cancel"):
            st.session_state[f"confirm_{title}"] = False
            return False
    return False