import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

st.title('Image to Text')


with st.container():
    st.title('test')
    with st.form(key="form"):
        img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
        submit = st.form_submit_button(label = "확인")
        if submit:
            st.image(img_file, caption='Uploaded Image', width=200)
            with st.spinner('로딩 중...'):
                image = Image.open(img_file).convert('RGB')

                processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
                model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
                
                text = "a photography of"
                inputs = processor(image, text, return_tensors="pt")
                outputs = model.generate(**inputs)
            
                with st.container():
                        st.title("결과")
                st.success(processor.decode(outputs[0], skip_special_tokens=True))
    
