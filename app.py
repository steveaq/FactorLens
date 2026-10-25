from pathlib import Path
import streamlit as st
from research import INDUSTRIES, MODELS, load_data, fit_model, performance

ROOT=Path(__file__).parent
st.set_page_config(page_title='FactorLens',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown(f'<style>{(ROOT/"styles.css").read_text()}</style>',unsafe_allow_html=True)
try:
    data,manifest=load_data()
except Exception:
    st.error('The data file could not be opened. Check data/snapshot.csv and data/snapshot.json.');st.stop()
with st.sidebar:
    st.markdown('### ◈ FactorLens')
    st.caption('US industry returns and factor models')
    industry=st.selectbox('Industry',list(INDUSTRIES),index=4,format_func=lambda i:INDUSTRIES[i],key='industry')
    model=st.selectbox('Model',list(MODELS),key='model')
    years=st.slider('Years',int(data.index[0].year),int(data.index[-1].year),(2010,int(data.index[-1].year)),key='years')
    st.caption(f'Monthly data through {data.index[-1]:%B %Y}. Values come from the French Data Library.')
frame=data[(data.index.year>=years[0]) & (data.index.year<=years[1])].copy()
factors=MODELS[model]
try:
    fitted=fit_model(frame,industry,factors)
except ValueError as exc:
    st.warning(str(exc));st.stop()
st.session_state.research_context={'data':data,'manifest':manifest,'frame':frame,'industry':industry,
    'model':model,'factors':factors,'fit':fitted,'metrics':performance(frame,industry)}
st.markdown(f'<div class="app-bar"><b>FactorLens</b><span>{INDUSTRIES[industry]} / {model} / {frame.index[0]:%b %Y} to {frame.index[-1]:%b %Y}</span></div>',unsafe_allow_html=True)
pages=[st.Page('views/overview.py',title='Overview',default=True,url_path='overview'),
       st.Page('views/exposures.py',title='Factor exposures',url_path='exposures'),
       st.Page('views/risk.py',title='Risk and returns',url_path='risk'),
       st.Page('views/diagnostics.py',title='Model diagnostics',url_path='diagnostics'),
       st.Page('views/validation.py',title='Validation',url_path='validation'),
       st.Page('views/methodology.py',title='Data and methodology',url_path='methodology')]
st.navigation(pages).run()
