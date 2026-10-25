import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from research import (INDUSTRIES,MODELS,FACTORS,fit_model,exposure_table,rolling_exposures,
    holdout_test,growth,research_note,refresh_snapshot)
from analytics import (risk_summary,drawdown_periods,factor_contributions,regression_diagnostics,
    walk_forward,validation_scores)

COLORS=['#205749','#b57350','#678795','#a5994c','#80608a']
NAMES={'Mkt-RF':'Market','SMB':'Size','HML':'Value','RMW':'Profitability','CMA':'Investment'}


def ctx(): return st.session_state.research_context


def title(heading,copy):
    st.title(heading);st.write(copy)


def plot(fig,height=330):
    fig.update_layout(height=height,margin=dict(l=12,r=12,t=25,b=20),paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',font=dict(family='Arial, sans-serif',color='#33443e',size=12),
        legend=dict(orientation='h',y=1.14,x=0),hovermode='x unified',
        xaxis=dict(showgrid=False,zeroline=False,automargin=True),yaxis=dict(gridcolor='#dedfd3',zerolinecolor='#bdc8ba',automargin=True))
    st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})


def lines(frame,percent=False,height=320):
    if frame.empty:
        st.info('Select more years to calculate this rolling series.');return
    fig=go.Figure()
    for name,color in zip(frame.columns,COLORS):
        fig.add_trace(go.Scatter(x=frame.index,y=frame[name]*(100 if percent else 1),name=name,
            line=dict(color=color,width=2),hovertemplate='%{y:.2f}<extra>%{fullData.name}</extra>'))
    if percent: fig.update_yaxes(ticksuffix='%')
    plot(fig,height)


@st.cache_data(show_spinner=False)
def trailing(frame,industry,factors,window): return rolling_exposures(frame,industry,factors,window)


@st.cache_data(show_spinner=False)
def forward(frame,industry,factors,window,expanding): return walk_forward(frame,industry,factors,window,expanding)


def overview():
    c=ctx();frame=c['frame'];industry=c['industry'];fit=c['fit'];metrics=c['metrics']
    title('Returns overview','Choose an industry and date range in the sidebar. The charts use monthly total returns, including reinvested distributions.')
    a,b,d,e=st.columns(4)
    a.metric('Annualised return',f'{metrics["cagr"]:.1%}')
    b.metric('Annualised volatility',f'{metrics["volatility"]:.1%}')
    d.metric('Maximum drawdown',f'{metrics["max_drawdown"]:.1%}')
    e.metric('Adjusted R²',f'{fit.rsquared_adj:.1%}')
    st.subheader('Growth of $100')
    wealth=growth(frame,industry).rename(columns={'Industry':INDUSTRIES[industry]})
    lines(wealth,height=360)
    st.caption('The market series adds RF back to Mkt−RF. These are research portfolios before costs; $100 is a common starting value, not an account balance.')
    left,right=st.columns(2)
    with left:
        st.subheader('Calendar-year returns')
        annual=(1+frame[[industry]]).groupby(frame.index.year).prod()-1
        fig=go.Figure(go.Bar(x=annual.index.astype(str),y=annual[industry]*100,marker_color=[COLORS[0] if x>=0 else COLORS[1] for x in annual[industry]]))
        fig.update_yaxes(ticksuffix='%');plot(fig,260)
        last=frame.index[-1]
        st.caption(f'{last.year} ends in {last:%B} in this dataset. Any incomplete calendar year is a partial-year return.')
    with right:
        st.subheader('Drawdown')
        drawdown=(wealth.iloc[:,0]/wealth.iloc[:,0].cummax()-1)*100
        fig=go.Figure(go.Scatter(x=drawdown.index,y=drawdown,fill='tozeroy',line=dict(color=COLORS[1],width=1.8),fillcolor='rgba(181,115,80,.13)'))
        fig.update_yaxes(ticksuffix='%');plot(fig,260)
        st.caption('Drawdown measures the fall from the previous peak, including the initial $100. Recovery can take longer than the decline.')
    ci=exposure_table(fit).loc['const']*12
    st.subheader('What does the model explain?')
    st.write(f'The selected model explains {fit.rsquared_adj:.1%} of the variation in monthly excess returns after adjusting for the number of regressors. Its annualised intercept is {fit.params["const"]*12:+.2%}, with a 95% interval from {ci.lower_95:+.2%} to {ci.upper_95:+.2%}.')
    if ci.lower_95<=0<=ci.upper_95:
        st.write('That interval includes zero, so this sample does not give a clear estimate of a nonzero intercept.')
    else:
        st.write('That interval excludes zero under the fitted model. It does not account for model selection or show that the intercept can be earned after costs.')
    st.caption('The intercept is multiplied by 12. It is separate from the compounded portfolio return shown above.')


def exposures():
    c=ctx();frame=c['frame'];fit=c['fit'];factors=c['factors'];industry=c['industry']
    title('Factor exposures','A coefficient describes how the industry’s excess return moves with a factor after controlling for the other factors in the model.')
    table=exposure_table(fit).loc[factors]
    left,right=st.columns([1.4,1])
    with left:
        st.subheader('Estimated coefficients')
        fig=go.Figure(go.Bar(x=table.exposure,y=[NAMES[f] for f in factors],orientation='h',marker_color=COLORS[:len(factors)],
            error_x=dict(type='data',symmetric=False,array=table.upper_95-table.exposure,arrayminus=table.exposure-table.lower_95)))
        plot(fig,300)
        st.caption('Bars show slopes and 95% HAC confidence intervals. A market slope of 1.2 corresponds to 1.2 percentage points of industry excess return for a one-point factor move, conditional on the other regressors.')
    with right:
        st.subheader('Factor definitions')
        st.dataframe(pd.DataFrame({'Factor':FACTORS,'Definition':['Market return less RF','Small minus big firms','High minus low book-to-market','Robust minus weak profitability','Conservative minus aggressive investment']}),hide_index=True,width='stretch')
        st.caption('Slopes are not portfolio weights. They can be negative and do not need to add to one.')
    window=st.select_slider('Trailing window, months',options=[36,60,84],value=60,key='exposure_window')
    rolling=trailing(frame,industry,factors,window)
    if not rolling.empty:
        st.subheader('Exposure over time')
        lines(rolling[factors].rename(columns=NAMES))
        st.caption('Each estimate includes the month on its x-axis and the preceding observations in the selected window. These are descriptive fits.')
    else: st.info('Select more years to estimate rolling exposures.')
    st.subheader('Monthly factor attribution')
    parts=factor_contributions(frame,industry,fit,factors)
    recent=parts.tail(24)
    fig=go.Figure()
    for name,color in zip(recent.columns,COLORS+['#b9beb3','#9d6d66']):
        fig.add_trace(go.Bar(x=recent.index,y=recent[name]*100,name=NAMES.get(name,name),marker_color=color))
    fig.update_layout(barmode='relative');fig.update_yaxes(title='Percentage points');plot(fig,350)
    st.caption('For each month, factor contributions plus the intercept and residual equal the observed excess return. Coefficients use the full selected sample. This is an in-sample decomposition, so it must not be read as a forecast.')
    st.subheader('How industries differ')
    universe=pd.DataFrame({INDUSTRIES[i]:fit_model(frame,i,factors).params[factors] for i in INDUSTRIES}).T
    fig=go.Figure(go.Heatmap(z=universe.values,x=[NAMES[f] for f in factors],y=universe.index,
        colorscale=[[0,COLORS[1]],[.5,'#f7f5ee'],[1,COLORS[0]]],zmid=0,hovertemplate='%{y}<br>%{x}: %{z:.2f}<extra></extra>'))
    plot(fig,390)
    st.caption('Every industry uses the same months and specification. Differences in scale or signs can help identify where the model is sensitive to the choice of industry.')
    st.download_button('Download attribution data',parts.to_csv(),'factor_attribution.csv','text/csv')


def risk():
    c=ctx();frame=c['frame'];industry=c['industry'];r=frame[industry]
    title('Risk and returns','This page looks at the distribution of monthly returns, how risk changes over time, and how long losses take to recover.')
    confidence=st.select_slider('Historical loss quantile',options=[.90,.95,.99],value=.95,format_func=lambda x:f'{x:.0%}')
    summary=risk_summary(r,confidence)
    a,b,d,e=st.columns(4)
    a.metric('Monthly VaR',f'{summary["var"]:.1%}')
    b.metric('Monthly expected shortfall',f'{summary["expected_shortfall"]:.1%}')
    d.metric('Positive months',f'{summary["positive_months"]:.1%}')
    e.metric('Worst month',f'{summary["worst"]:.1%}')
    st.caption(f'Loss is defined as minus return. VaR is its {confidence:.0%} sample quantile. Expected shortfall averages losses at or above that threshold ({summary["tail_months"]} months here). These historical estimates are sensitive to the chosen period, particularly at 99%.')
    left,right=st.columns(2)
    with left:
        st.subheader('Monthly return distribution')
        fig=go.Figure(go.Histogram(x=r*100,nbinsx=35,histnorm='probability density',marker_color=COLORS[0],opacity=.7,name='Observed'))
        x=np.linspace(r.min()*100,r.max()*100,180)
        if r.std()>0:fig.add_trace(go.Scatter(x=x,y=stats.norm.pdf(x,r.mean()*100,r.std()*100),name='Normal reference',line=dict(color=COLORS[1],dash='dot')))
        fig.update_xaxes(title='Monthly return (%)');plot(fig,300)
        st.caption(f'Skew: {summary["skew"]:.2f}. Excess kurtosis: {summary["excess_kurtosis"]:.2f}. The dotted curve has the same sample mean and standard deviation; it is a reference, not an assumption used to estimate VaR.')
    with right:
        st.subheader('Returns by month and year')
        matrix=pd.DataFrame({'year':frame.index.year,'month':frame.index.month,'return':r.to_numpy()*100}).pivot(index='year',columns='month',values='return').reindex(columns=range(1,13))
        fig=go.Figure(go.Heatmap(z=matrix.values,x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],y=matrix.index,
            colorscale=[[0,COLORS[1]],[.5,'#f7f5ee'],[1,COLORS[0]]],zmid=0,hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>'))
        plot(fig,300)
        st.caption('Blank cells have no observation. This heatmap is useful for locating large months; it does not establish a seasonal trading pattern.')
    window=st.select_slider('Risk window, months',options=[12,24,36,60],value=36)
    a,b=st.columns(2)
    with a:
        st.subheader('Rolling volatility')
        vol=frame[[industry,'Mkt-RF']].copy();vol['Mkt-RF']+=frame.RF
        vol=vol.rolling(window).std()*np.sqrt(12)
        lines(vol.dropna().rename(columns={industry:INDUSTRIES[industry],'Mkt-RF':'US market'}),percent=True,height=270)
    with b:
        st.subheader('Rolling Sharpe ratio')
        excess=frame[industry]-frame.RF
        sharpe=excess.rolling(window).mean()/excess.rolling(window).std()*np.sqrt(12)
        lines(sharpe.dropna().to_frame('Excess-return Sharpe'),height=270)
        st.caption('Annualisation uses √12. Serial dependence and short windows can make the usual Sharpe interpretation unreliable.')
    st.subheader('Largest drawdown episodes')
    episodes=drawdown_periods(r)
    st.dataframe(episodes.head(10),hide_index=True,width='stretch',column_config={'Depth':st.column_config.NumberColumn(format='percent')})
    st.caption('An empty recovery date means the portfolio had not regained that peak by the end of the selected data. Months underwater run from the prior peak to recovery, or to the last available month.')
    st.subheader('Industry return correlations')
    correlation=frame[list(INDUSTRIES)].rename(columns=INDUSTRIES).corr()
    fig=go.Figure(go.Heatmap(z=correlation.values,x=correlation.columns,y=correlation.index,zmin=-1,zmax=1,zmid=0,
        colorscale=[[0,COLORS[1]],[.5,'#f7f5ee'],[1,COLORS[0]]],hovertemplate='%{x}<br>%{y}: %{z:.2f}<extra></extra>'))
    plot(fig,410)
    st.caption('Pearson correlations use monthly total returns over the selected window. High historical correlation does not tell you what diversification will look like during the next drawdown.')


def diagnostics():
    c=ctx();fit=c['fit'];frame=c['frame'];factors=c['factors']
    title('Model diagnostics','A high R² can hide a poor specification. Check the residuals and the relationships among the factors before interpreting an intercept or an individual slope.')
    result=regression_diagnostics(fit,frame,factors)
    a,b,d=st.columns(3)
    a.metric('Residual standard deviation',f'{np.std(fit.resid,ddof=1):.2%}')
    b.metric('Durbin-Watson',f'{result["dw"]:.2f}')
    d.metric('Design condition number',f'{result["condition_number"]:.1f}')
    st.caption('Durbin-Watson near two is consistent with little first-order residual autocorrelation. The condition number depends on variable scaling as well as collinearity, so it should be read alongside VIF and factor correlations.')
    left,right=st.columns(2)
    with left:
        st.subheader('Residuals against fitted values')
        fig=go.Figure(go.Scatter(x=fit.fittedvalues*100,y=fit.resid*100,mode='markers',marker=dict(color=COLORS[0],size=5,opacity=.6),text=frame.index.strftime('%b %Y'),hovertemplate='%{text}<br>Fitted %{x:.2f} pp<br>Residual %{y:.2f} pp<extra></extra>'))
        fig.add_hline(y=0,line_dash='dot',line_color=COLORS[1]);fig.update_xaxes(title='Fitted excess return (pp)');fig.update_yaxes(title='Residual (pp)');plot(fig,300)
        st.caption('Curvature can suggest omitted nonlinear effects. A widening spread can suggest changing residual variance.')
    with right:
        st.subheader('Residual autocorrelation')
        values=result['acf'][1:]
        fig=go.Figure(go.Bar(x=list(range(1,len(values)+1)),y=values,marker_color=COLORS[0]))
        for y in [-result['acf_band'],result['acf_band']]:fig.add_hline(y=y,line_dash='dot',line_color=COLORS[1])
        fig.update_xaxes(title='Lag in months',dtick=1);plot(fig,300)
        st.caption('Dotted bounds are the approximate ±1.96/√n white-noise reference. They are not adjusted for testing several lags or for coefficient estimation.')
    left,right=st.columns(2)
    with left:
        st.subheader('Normal Q-Q plot of residuals')
        fig=go.Figure(go.Scatter(x=result['qq_x'],y=result['qq_y'],mode='markers',marker=dict(color=COLORS[0],size=5),name='Residuals'))
        fig.add_trace(go.Scatter(x=result['qq_x'],y=result['qq_line'],name='Fitted normal reference',line=dict(color=COLORS[1],dash='dot')))
        fig.update_xaxes(title='Theoretical normal quantile');fig.update_yaxes(title='Ordered residual (pp)');plot(fig,300)
        st.caption('Tail departures are common in returns. HAC intervals relax the constant-variance assumption; they do not make a misspecified model correct.')
    with right:
        st.subheader('Variance inflation factors')
        st.dataframe(result['vif'],hide_index=True,width='stretch')
        st.write('VIF measures how well each regressor can be explained by the others. A value of one means no linear overlap. Large values make individual slopes harder to estimate precisely; there is no universal cutoff that fixes this problem.')
        correlation=frame[factors].corr()
        fig=go.Figure(go.Heatmap(z=correlation.values,x=factors,y=factors,zmin=-1,zmax=1,zmid=0,colorscale=[[0,COLORS[1]],[.5,'#f7f5ee'],[1,COLORS[0]]]))
        plot(fig,230)
    st.subheader('Specification tests')
    st.dataframe(result['tests'],hide_index=True,width='stretch')
    st.write('A small p-value is evidence against the stated null under that test’s assumptions. Several tests on the same sample increase the chance of a false positive. These results help locate problems; they are not a pass/fail score for the model.')
    st.markdown('[Test definitions in the statsmodels documentation](https://www.statsmodels.org/stable/diagnostic.html)')


def validation():
    c=ctx();frame=c['frame'];industry=c['industry'];factors=c['factors']
    title('Validation','Test whether coefficients estimated on earlier months still explain later returns. The factors for each test month are realised values, so these exercises measure attribution stability, not trading forecasts.')
    st.subheader('Fixed chronological holdout')
    months=st.select_slider('Holdout length, months',options=[12,24,36,48],value=24)
    try: result=holdout_test(frame,industry,factors,months)
    except ValueError as exc:st.info(str(exc));return
    a,b,d=st.columns(3)
    a.metric('Model MAE',f'{result["mae"]*100:.2f} pp');b.metric('Market-only MAE',f'{result["baseline_mae"]*100:.2f} pp');d.metric('R² vs training mean',f'{result["oos_r2"]:.3f}')
    st.caption(f'Coefficients are frozen after {result["train_end"]:%B %Y}. The holdout runs from {result["test_start"]:%B %Y} to {result["test_end"]:%B %Y}. Both models use the same training and test months.')
    lines(result['series']*100)
    st.caption('Series show monthly excess returns in percentage points. A negative holdout R² means the training-period mean gave a smaller squared error.')
    st.subheader('Walk-forward refits')
    a,b=st.columns(2)
    with a:window=st.select_slider('Initial training window, months',options=[36,60,120],value=60)
    with b:expanding=st.radio('Training history',['Rolling window','Expanding window'],horizontal=True)=='Expanding window'
    if len(frame)<=window:st.info('Select a longer date range for this training window.');return
    selected=forward(frame,industry,factors,window,expanding)
    rows=[]
    for name,cols in MODELS.items():
        experiment=forward(frame,industry,cols,window,expanding)
        rows.append({'Model':name,**validation_scores(experiment),'Test months':len(experiment)})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.write('For each test month, the fit ends in the preceding month. A rolling fit keeps the training sample the same length; an expanding fit retains all earlier months. The table compares identical test dates across model specifications.')
    error=pd.DataFrame({'Selected model':(selected.actual-selected.estimate).abs()*100,'Market only':(selected.actual-selected.market_only).abs()*100},index=selected.index)
    st.subheader('Trailing 12-month absolute attribution error')
    lines(error.rolling(12).mean().dropna())
    st.caption('Values are mean absolute errors in percentage points. Adding factors can improve the full-sample fit and still make later attribution worse.')
    selected['error_pp']=(selected.actual-selected.estimate)*100
    st.subheader('Months with the largest errors')
    worst=selected.reindex(selected.error_pp.abs().sort_values(ascending=False).index).head(10)
    st.dataframe(worst, width='stretch')
    st.caption('Returns and estimates in this table are decimal fractions; error_pp is in percentage points. train_end records the last month used to fit that estimate.')
    st.download_button('Download walk-forward results',selected.to_csv(),'walk_forward.csv','text/csv')


def methodology():
    c=ctx();frame=c['frame'];meta=c['manifest']
    title('Data and methodology','The app uses the French Data Library’s US factors and ten industry portfolios. The bundled snapshot lets the analysis run without an API key or a live connection.')
    st.write(f'The loaded dataset covers {c["data"].index[0]:%B %Y} to {c["data"].index[-1]:%B %Y}. It was retrieved on {meta["retrieved_at"][:10]}. The research window currently contains {len(frame)} monthly observations.')
    st.subheader('Regression specification')
    st.latex(r'R_{i,t}-R_{f,t}=\alpha+\beta_M(Mkt-RF)_t+\beta_S SMB_t+\beta_H HML_t+\beta_R RMW_t+\beta_C CMA_t+\epsilon_t')
    st.write('The three-factor model omits profitability and investment. The market-only model keeps the intercept and market excess return. All fits use ordinary least squares. Reported intervals use Newey-West HAC standard errors with three monthly lags.')
    st.write('Monthly source percentages are converted to decimals once. Industry returns come from the first value-weighted monthly block. Annual summaries and equal-weighted returns are excluded. Missing-value sentinels are removed before month alignment, and the combined series must be contiguous.')
    st.subheader('Metric conventions')
    st.markdown('CAGR compounds monthly total returns. Volatility uses sample standard deviation times √12. Sharpe uses monthly excess returns and the same annualisation. Alpha is 12 times the monthly intercept. VaR and expected shortfall use empirical monthly losses, with no normal-distribution assumption.')
    st.subheader('Source data for the selected window')
    st.dataframe(frame.tail(24).rename(columns=INDUSTRIES),width='stretch')
    st.caption('0.01 means a 1% monthly return. Historical source data can be revised, and the underlying CRSP file format changed in 2025. Research portfolios omit implementation costs and are not directly investable.')
    st.markdown('[French Data Library and construction notes](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)')
    with st.expander('Download provenance'):st.json(meta)
    if st.button('Refresh source data'):
        try:
            with st.spinner('Downloading and checking the source archives...'):refresh_snapshot()
            st.cache_data.clear();st.rerun()
        except Exception:st.error('The refresh failed. The existing snapshot is still available.')
    a,b=st.columns(2)
    with a:st.download_button('Download monthly observations',frame.to_csv(),'factorlens_monthly.csv','text/csv',width='stretch')
    with b:st.download_button('Download research note',research_note(frame,c['industry'],c['model'],c['fit'],c['metrics']),'factorlens_research.md','text/markdown',width='stretch')
