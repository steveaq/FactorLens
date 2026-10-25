"""Additional risk and model diagnostics for monthly research returns."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import acf
from research import fit_model


def risk_summary(returns, confidence=.95):
    values=pd.Series(returns).dropna()
    if len(values)<12 or not 0<confidence<1:
        raise ValueError('Risk estimates need at least 12 months and a confidence between zero and one.')
    losses=-values
    threshold=float(losses.quantile(confidence))
    tail=losses[losses>=threshold]
    return {'var':threshold,'expected_shortfall':float(tail.mean()),'tail_months':len(tail),
        'positive_months':float(values.gt(0).mean()),'worst':float(values.min()),
        'best':float(values.max()),'skew':float(values.skew()),'excess_kurtosis':float(values.kurt())}


def drawdown_periods(returns):
    values=pd.Series(returns).sort_index()
    wealth=1.; peak=1.; peak_date=values.index[0]-pd.offsets.MonthEnd(1)
    start=None; trough=None; depth=0.; rows=[]
    for month,value in values.items():
        wealth*=1+value
        loss=wealth/peak-1
        if loss>=-1e-12:
            if start is not None:
                rows.append({'Peak':start,'Trough':trough,'Recovery':month,'Depth':depth,
                    'Months underwater':(month.year-start.year)*12+month.month-start.month})
            peak=wealth;peak_date=month;start=None;trough=None;depth=0.
        else:
            if start is None: start=peak_date
            if loss<depth: depth=loss;trough=month
    if start is not None:
        end=values.index[-1]
        rows.append({'Peak':start,'Trough':trough,'Recovery':pd.NaT,'Depth':depth,
            'Months underwater':(end.year-start.year)*12+end.month-start.month})
    return pd.DataFrame(rows,columns=['Peak','Trough','Recovery','Depth','Months underwater']).sort_values('Depth')


def factor_contributions(frame, industry, fitted, factors):
    parts=frame[factors].mul(fitted.params[factors],axis=1)
    parts['Intercept']=fitted.params['const']
    parts['Residual']=(frame[industry]-frame.RF)-parts.sum(axis=1)
    return parts


def regression_diagnostics(fitted, frame, factors):
    residuals=np.asarray(fitted.resid)
    lag=min(12,max(1,len(residuals)//5))
    lb=acorr_ljungbox(residuals,lags=[lag],return_df=True)
    bp=het_breuschpagan(residuals,fitted.model.exog)
    jb=jarque_bera(residuals)
    design=sm.add_constant(frame[factors],has_constant='add').to_numpy()
    vif=pd.DataFrame({'Factor':factors,'VIF':[variance_inflation_factor(design,i+1) for i in range(len(factors))]})
    autocorrelation=acf(residuals,nlags=lag,fft=False)
    quantiles=stats.probplot(residuals*100,dist='norm')
    tests=pd.DataFrame([
        {'Test':f'Ljung-Box ({lag} lags)','Statistic':float(lb.lb_stat.iloc[0]),'p-value':float(lb.lb_pvalue.iloc[0]),'Null hypothesis':'Residual autocorrelations are zero through the chosen lag.'},
        {'Test':'Breusch-Pagan (LM)','Statistic':float(bp[0]),'p-value':float(bp[1]),'Null hypothesis':'Residual variance does not depend on the regressors.'},
        {'Test':'Jarque-Bera','Statistic':float(jb[0]),'p-value':float(jb[1]),'Null hypothesis':'Residual skew and kurtosis match a normal distribution.'},
    ])
    return {'tests':tests,'vif':vif,'acf':autocorrelation,'acf_band':1.96/np.sqrt(len(residuals)),
        'dw':float(durbin_watson(residuals)),'condition_number':float(fitted.condition_number),
        'qq_x':quantiles[0][0],'qq_y':quantiles[0][1],
        'qq_line':quantiles[1][0]*quantiles[0][0]+quantiles[1][1]}


def walk_forward(frame, industry, factors, window=60, expanding=False):
    if window<36 or len(frame)<=window:
        raise ValueError('Walk-forward analysis needs at least 36 training months and one later month.')
    records=[]
    for position in range(window,len(frame)):
        train=frame.iloc[0 if expanding else position-window:position]
        row=frame.iloc[position]
        fitted=fit_model(train,industry,factors)
        baseline=fit_model(train,industry,['Mkt-RF'])
        estimate=float(fitted.params['const']+sum(fitted.params[f]*row[f] for f in factors))
        market=float(baseline.params['const']+baseline.params['Mkt-RF']*row['Mkt-RF'])
        records.append({'month':frame.index[position], 'actual':float(row[industry]-row.RF),
            'estimate':estimate,'market_only':market,'past_mean':float((train[industry]-train.RF).mean()),
            'train_end':train.index[-1]})
    return pd.DataFrame(records).set_index('month')


def validation_scores(result):
    error=result.actual-result.estimate
    denominator=float(((result.actual-result.past_mean)**2).sum())
    return {'MAE (pp)':float(error.abs().mean()*100),'RMSE (pp)':float(np.sqrt((error**2).mean())*100),
        'R² vs past mean':1-float((error**2).sum())/denominator if denominator>0 else np.nan}
