import streamlit as st
import json

placeholder = st.empty()
fp = placeholder.container()

if __name__ == '__main__':
    path = st.text_input(label="path")
    if st.button(label='button'):
        with open(path, 'r') as f:
            data_dicts = json.load(f)
        metadata_path = '.'.join(path.split('.')[:-1]) + '_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata_dicts = json.load(f)
        tabs = st.tabs([i for i in data_dicts.keys()])
        for i, datas in data_dicts.items():
            data_num = int(i[-1])
            with tabs[data_num]:
                for data in datas:
                    with st.chat_message(data['name'][-1]):
                        st.write(data['content'])
                with st.container():
                    st.write(f"Domain: {metadata_dicts['domain'][data_num]}")
                    st.write(f"funclist: {metadata_dicts['funclist']}")
