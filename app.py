from pathlib import Path
import html
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from research import (FACTORS, INDUSTRIES, MODELS, load_data, refresh_snapshot, fit_model,
    exposure_table, rolling_exposures, holdout_test, performance, growth, research_note)

ROOT = Path(__file__).parent
st.set_page_config(page_title='FactorLens · See what moves returns', page_icon='◈', layout='wide', initial_sidebar_state='collapsed')
st.markdown(f'<style>{(ROOT/"styles.css").read_text()}</style>', unsafe_allow_html=True)
COLORS = ['#205749','#b57350','#788f9a','#a39b5f','#735b7d']
NAMES = {'Mkt-RF':'Market', 'SMB':'Size', 'HML':'Value', 'RMW':'Profitability', 'CMA':'Investment'}


def raw(text):
    st.markdown(text, unsafe_allow_html=True)


def chart(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=12,r=16,t=20,b=15), paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Arial, sans-serif',color='#33443e',size=11),
        legend=dict(orientation='h',y=1.12,x=0), hovermode='x unified',
        xaxis=dict(showgrid=False,zeroline=False), yaxis=dict(gridcolor='#e3e3d8',zerolinecolor='#c0c7bc'))
    st.plotly_chart(fig, width='stretch', config={'displayModeBar':False})


@st.cache_data(show_spinner=False)
def rolling(frame, industry, factors, window):
    return rolling_exposures(frame, industry, factors, window)


def main():
    try:
        data, manifest = load_data()
    except Exception:
        st.error('The research snapshot could not be loaded. Restore data/snapshot.csv and snapshot.json, then restart.')
        st.stop()
    raw('<nav><div class="wordmark"><span>◈</span> FactorLens<span class="nav-tag">A RETURN, EXPLAINED.</span></div><a href="#research-desk">OPEN RESEARCH DESK ↗</a></nav>')
    raw(f'''<section class="hero"><div><div class="eyebrow">INDEPENDENT QUANTITATIVE RESEARCH / VOL. 01</div>
    <h1>There’s a story<br> beneath <em>the return.</em></h1><p>Separate market exposure from the rest. Explore the factors<br> behind ten US industry portfolios, one month at a time.</p>
    <div class="hero-note"><span class="dot"></span> PUBLIC DATA. TRANSPARENT MODELS. EXPLICIT ASSUMPTIONS.</div></div>
    <aside class="research-card"><span class="eyebrow">THE RESEARCH UNIVERSE</span><div class="orb" aria-hidden="true"><i></i><b>f</b></div>
    <div class="card-numbers"><span><strong>10</strong>industries</span><span><strong>05</strong>factors</span><span><strong>{len(data)}</strong>months</span></div>
    <small>MONTHLY OBSERVATIONS · THROUGH {data.index[-1]:%b %Y}</small></aside></section>''')
    raw('<div id="research-desk" class="section-title"><span>01 /</span><h2>The research desk</h2><small>CHANGE THE LENS. KEEP THE EVIDENCE.</small></div>')
    with st.container(border=True):
        a,b,c = st.columns([1.15,1,1.65])
        with a:
            industry = st.selectbox('Industry portfolio', list(INDUSTRIES), index=4, format_func=lambda k:INDUSTRIES[k])
        with b:
            model = st.selectbox('Explanatory model', list(MODELS))
        with c:
            years = st.slider('Research window', int(data.index[0].year),int(data.index[-1].year),
                (2010,int(data.index[-1].year)),help='Calendar-year boundaries, monthly observations. Short windows may not support validation.')
        st.caption(f'Value-weighted US research portfolios · monthly total returns · {years[0]}–{years[1]} · source: Kenneth R. French Data Library')
    frame = data[(data.index.year>=years[0]) & (data.index.year<=years[1])].copy()
    factors = MODELS[model]
    try:
        fitted = fit_model(frame, industry, factors)
        metrics = performance(frame, industry)
    except ValueError as exc:
        st.warning(str(exc)); st.stop()
    table = exposure_table(fitted)
    alpha = fitted.params['const']*12
    ci = table.loc['const',['lower_95','upper_95']]*12
    for column, label, value, detail in zip(st.columns(4),
        ['ANNUALISED ALPHA','MODEL FIT','ANNUALISED RETURN','MAXIMUM DRAWDOWN'],
        [f'{alpha:+.2%}',f'{fitted.rsquared_adj:.1%}',f'{metrics["cagr"]:.1%}',f'{metrics["max_drawdown"]:.1%}'],
        [f'95% CI {ci.iloc[0]:+.1%} to {ci.iloc[1]:+.1%} · linear', 'Adjusted R² · in sample', 'CAGR · total portfolio return', 'Peak-to-trough · selected window']):
        with column:
            raw(f'<div class="metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>')
    overview, exposures, validation, room = st.tabs(['The overview','Factor anatomy','Out-of-sample check','Data & method'])
    valid = None
    with overview:
        left,right = st.columns([2.05,1],gap='large')
        with left:
            raw('<div class="panel-heading"><h3>A hundred dollars, through time.</h3><span>GROWTH OF $100</span></div>')
            wealth = growth(frame, industry)
            fig = go.Figure()
            for name,color in zip(wealth.columns,COLORS):
                fig.add_trace(go.Scatter(x=wealth.index,y=wealth[name],name=INDUSTRIES[industry] if name=='Industry' else name,
                    line=dict(color=color,width=2.6 if name=='Industry' else 1.5,dash='solid' if name=='Industry' else 'dot'),
                    hovertemplate='%{y:$.2f}<extra>%{fullData.name}</extra>'))
            fig.update_yaxes(tickprefix='$')
            chart(fig,350)
            st.caption('Historical total returns, with distributions reinvested. US market = Mkt−RF + RF. Before costs; research portfolios are not directly investable.')
        with right:
            strongest = max(factors,key=lambda f:abs(fitted.params[f]))
            significant = ci.iloc[0]>0 or ci.iloc[1]<0
            raw(f'''<aside class="insight"><span class="eyebrow">READING THE RESULTS</span><h3>{NAMES[strongest]} is the<br> largest exposure.</h3>
            <p>A one-percentage-point move in this factor is associated with a {fitted.params[strongest]:+.2f} percentage-point move in monthly excess return, holding other factors fixed.</p>
            <div class="insight-rule"></div><strong>{'Alpha’s interval excludes zero.' if significant else 'Alpha’s interval includes zero.'}</strong>
            <p>{'The fitted intercept is statistically distinguishable from zero under these assumptions; this is not evidence of a tradable edge.' if significant else 'The sample does not clearly distinguish the intercept from zero. A positive point estimate alone is not a demonstrated edge.'}</p></aside>''')
        left,right = st.columns(2,gap='large')
        with left:
            st.subheader('The drawdown beneath the growth')
            drawdown=(wealth/wealth.cummax()-1)*100
            fig=go.Figure(go.Scatter(x=drawdown.index,y=drawdown['Industry'],fill='tozeroy',fillcolor='rgba(181,115,80,.14)',line=dict(color=COLORS[1],width=1.8),name='Drawdown'))
            fig.update_yaxes(ticksuffix='%'); chart(fig,230)
        with right:
            st.subheader('Risk, in context')
            stats=pd.DataFrame({'Metric':['Annualised volatility','Monthly-excess Sharpe × √12','Observations','Risk-free input'],
                'Value':[f'{metrics["volatility"]:.2%}',f'{metrics["sharpe"]:.2f}',str(len(frame)),'French monthly RF']})
            st.dataframe(stats,hide_index=True,width='stretch')
            st.caption('Sharpe uses monthly excess returns and sample standard deviation. The monthly intercept is multiplied by 12; it is not compounded or presented as a trading return.')
    with exposures:
        col1,col2=st.columns([1.5,1],gap='large')
        with col1:
            st.subheader('What you are exposed to')
            coeff=table.loc[factors]
            fig=go.Figure(go.Bar(x=coeff.exposure,y=[NAMES[f] for f in factors],orientation='h',
                marker_color=[COLORS[i] for i in range(len(factors))],
                error_x=dict(type='data',symmetric=False,array=coeff.upper_95-coeff.exposure,arrayminus=coeff.exposure-coeff.lower_95),
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
            chart(fig,310)
            st.caption('Factor slopes with 95% HAC confidence intervals. Slopes are sensitivities, not portfolio weights; they need not sum to one.')
        with col2:
            st.subheader('A factor field guide')
            for f,desc in [('Mkt-RF','Market excess return'),('SMB','Small minus big companies'),('HML','High minus low book-to-market'),('RMW','Robust minus weak profitability'),('CMA','Conservative minus aggressive investment')]:
                raw(f'<div class="factor-definition"><b>{f}</b><span>{desc}</span></div>')
        window=st.select_slider('Trailing fit window (months)',options=[36,60,84],value=60)
        roll=rolling(frame,industry,factors,window)
        if roll.empty:
            st.info('Widen the research window to display trailing exposures.')
        else:
            st.subheader('Exposures evolve')
            fig=go.Figure()
            for f,color in zip(factors,COLORS):
                fig.add_trace(go.Scatter(x=roll.index,y=roll[f],name=NAMES[f],line=dict(color=color,width=1.7)))
            chart(fig,280)
            st.caption('Each point fits the trailing window including that month. This chart is descriptive, not a forward signal.')
        st.subheader('The wider universe')
        universe=pd.DataFrame({INDUSTRIES[i]:fit_model(frame,i,factors).params[factors] for i in INDUSTRIES}).T
        fig=go.Figure(go.Heatmap(z=universe.values,x=[NAMES[f] for f in factors],y=universe.index,
            colorscale=[[0,'#b57350'],[.5,'#f5f3eb'],[1,'#205749']],zmid=0,
            hovertemplate='%{y}<br> %{x}: %{z:.2f}<extra></extra>',colorbar=dict(title='Beta')))
        chart(fig,390)
    with validation:
        st.subheader('Fit in the past. Examine what follows.')
        st.write('Freeze the coefficients before the holdout begins, then check whether they still explain later excess returns. The holdout uses realised factor returns from the same months; it is an attribution stability check, not a forecast.')
        holdout=st.select_slider('Months held out',options=[12,24,36,48],value=24)
        try:
            valid=holdout_test(frame,industry,factors,holdout)
            a,b,c=st.columns(3)
            a.metric('Selected model MAE',f'{valid["mae"]*100:.2f} pp')
            b.metric('Market-only MAE',f'{valid["baseline_mae"]*100:.2f} pp')
            c.metric('Holdout R² vs training mean',f'{valid["oos_r2"]:.3f}')
            raw(f'<div class="split-strip"><span>TRAIN / {frame.index[0]:%b %Y} → {valid["train_end"]:%b %Y}</span><b>HOLDOUT / {valid["test_start"]:%b %Y} → {valid["test_end"]:%b %Y}</b></div>')
            fig=go.Figure()
            for name,color in zip(valid['series'].columns,COLORS):
                fig.add_trace(go.Scatter(x=valid['series'].index,y=valid['series'][name]*100,name=name,line=dict(color=color,width=1.8)))
            fig.update_yaxes(ticksuffix=' pp'); chart(fig,340)
            st.caption('MAE is measured in monthly excess-return percentage points. Negative holdout R² means the model does worse than the training-period mean. The market baseline is fitted on exactly the same training months.')
        except ValueError as exc:
            st.info(str(exc))
    with room:
        st.subheader('The evidence drawer')
        st.write(f'Data spans {data.index[0]:%B %Y} to {data.index[-1]:%B %Y}. Snapshot retrieved {manifest["retrieved_at"][:10]}. The app opens from a bundled public-data snapshot; refresh is explicit.')
        st.dataframe(frame.tail(12).rename(columns=INDUSTRIES),width='stretch')
        st.caption('Values in this table are decimal monthly returns: 0.01 means 1%. Industry data uses the first, value-weighted monthly block. Missing-source sentinels are excluded before alignment.')
        st.markdown('**Method.** OLS on industry excess returns, with an intercept and selected French factors. Confidence intervals use HAC standard errors with three lags. Historical data and factor construction can change; no causal or future-performance claim is made.')
        st.markdown('[Source and methodology: Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)')
        st.json(manifest,expanded=False)
        if st.button('Refresh from the official source ↗'):
            with st.spinner('Downloading and checking both research archives…'):
                try:
                    refresh_snapshot(); rolling.clear(); st.success('New source bundle validated and saved.'); st.rerun()
                except Exception:
                    st.error('The source refresh failed. Your existing snapshot is unchanged; try again later.')
    st.divider()
    a,b,c=st.columns([2,1,1])
    with a:
        st.subheader('Take the research with you.')
        st.caption('A reproducible note and the exact observations behind this view.')
    with b:
        st.download_button('Download research note ↓',research_note(frame,industry,model,fitted,metrics,valid),
            file_name=f'factorlens_{industry}_research.md',mime='text/markdown',width='stretch')
    with c:
        st.download_button('Download monthly data ↓',frame.to_csv(),file_name='factorlens_monthly.csv',mime='text/csv',width='stretch')
    raw('<footer><b>FactorLens</b><span>RESEARCH WITH THE ASSUMPTIONS LEFT IN.</span><span>DESCRIPTIVE ANALYSIS · NO LIVE TRADING</span></footer>')


if __name__=='__main__':
    main()
