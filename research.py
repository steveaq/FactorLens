"""Monthly factor research. Returns are decimal fractions throughout."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

ROOT = Path(__file__).parent
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
MODELS = {'Five factors': FACTORS, 'Three factors': FACTORS[:3], 'Market only': FACTORS[:1]}
INDUSTRIES = {'NoDur':'Consumer staples', 'Durbl':'Consumer durables', 'Manuf':'Manufacturing',
    'Enrgy':'Energy', 'HiTec':'Technology', 'Telcm':'Telecommunications', 'Shops':'Retail',
    'Hlth':'Healthcare', 'Utils':'Utilities', 'Other':'Other industries'}
SOURCES = {
    'factors':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip',
    'industries':'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_CSV.zip',
}


def parse_monthly(content: bytes) -> pd.DataFrame:
    """Read only the first monthly block (value-weighted returns for industries).

    French archives contain multiple blocks, annual summaries and prose. Never
    concatenate these into the monthly returns or accidentally include equal weights.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        filename = next(n for n in archive.namelist() if n.lower().endswith('.csv'))
        lines = archive.read(filename).decode('utf-8-sig').splitlines()
    first = next((i for i, line in enumerate(lines) if re.match(r'^\s*\d{6}\s*,', line)), None)
    if first is None or first == 0:
        raise ValueError('No monthly return block in the source archive.')
    header = lines[first-1]
    end = first
    while end < len(lines) and re.match(r'^\s*\d{6}\s*,', lines[end]):
        end += 1
    frame = pd.read_csv(io.StringIO('\n'.join([header] + lines[first:end])), index_col=0)
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format='%Y%m') + pd.offsets.MonthEnd(0)
    frame.index.name = 'month'
    frame = frame.apply(pd.to_numeric, errors='raise').replace([-99.99, -999], np.nan) / 100
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError('The source has duplicate or unordered months.')
    return frame


def combine(factors, industries):
    required = FACTORS + ['RF']
    if not set(required).issubset(factors) or not set(INDUSTRIES).issubset(industries):
        raise ValueError('The source schema changed; the previous snapshot is retained.')
    frame = factors[required].join(industries[list(INDUSTRIES)], how='inner').dropna()
    if len(frame) < 120 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Insufficient or invalid monthly data.')
    expected = pd.period_range(frame.index.min(), frame.index.max(), freq='M')
    if len(expected) != len(frame):
        raise ValueError('Source data contains missing months.')
    if (frame[list(INDUSTRIES)] <= -1).any().any():
        raise ValueError('Invalid total return in source data.')
    return frame


def load_snapshot():
    live = ROOT/'data'/'live.csv'
    file = live if live.exists() else ROOT/'data'/'snapshot.csv'
    meta = file.with_suffix('.json')
    return pd.read_csv(file, index_col='month', parse_dates=True), json.loads(meta.read_text())


def refresh_snapshot():
    """Download and validate both inputs before publishing a single combined file."""
    inputs, manifest = {}, {'retrieved_at':datetime.now(timezone.utc).isoformat(), 'sources':[]}
    for name, url in SOURCES.items():
        response = requests.get(url, timeout=(10, 35), headers={'User-Agent':'FactorLens/1.0 research snapshot'})
        response.raise_for_status()
        inputs[name] = parse_monthly(response.content)
        manifest['sources'].append({'name':name,'url':url,'sha256':hashlib.sha256(response.content).hexdigest()})
    frame = combine(inputs['factors'], inputs['industries'])
    manifest.update(first_month=str(frame.index.min().date()), last_month=str(frame.index.max().date()),
                    rows=len(frame), units='decimal monthly returns', weighting='value-weighted industry portfolios')
    # Publish the data and its provenance as a single atomic JSON document, then
    # load it as a unit. The committed CSV remains a portable offline fallback.
    import os, tempfile
    payload = {'manifest':manifest, 'csv':frame.to_csv()}
    destination = ROOT/'data'/'live_bundle.json'
    with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, delete=False) as temporary:
        json.dump(payload, temporary)
        temporary_path = temporary.name
    os.replace(temporary_path, destination)
    return frame, manifest


def load_data():
    live = ROOT/'data'/'live_bundle.json'
    if live.exists():
        payload = json.loads(live.read_text())
        return pd.read_csv(io.StringIO(payload['csv']), index_col='month', parse_dates=True), payload['manifest']
    return load_snapshot()


def fit_model(frame, industry, factors):
    if industry not in INDUSTRIES or not factors or not set(factors).issubset(FACTORS):
        raise ValueError('Unknown industry or factor model.')
    if len(frame) < max(24, len(factors) * 5):
        raise ValueError('Use at least 24 months (25 for the five-factor model).')
    y = frame[industry] - frame['RF']
    x = sm.add_constant(frame[factors], has_constant='add')
    return sm.OLS(y, x, missing='raise').fit(cov_type='HAC', cov_kwds={'maxlags':3})


def exposure_table(fit):
    confidence = fit.conf_int(alpha=.05)
    return pd.DataFrame({'exposure':fit.params, 'lower_95':confidence[0],
                         'upper_95':confidence[1], 'p_value':fit.pvalues})


def rolling_exposures(frame, industry, factors, window=60):
    if window < 25:
        raise ValueError('Rolling windows must contain at least 25 months.')
    records = []
    for end in range(window, len(frame)+1):
        fitted = fit_model(frame.iloc[end-window:end], industry, factors)
        records.append({'month':frame.index[end-1], **fitted.params.to_dict()})
    return pd.DataFrame(records).set_index('month') if records else pd.DataFrame()


def holdout_test(frame, industry, factors, holdout=24):
    if holdout < 6 or len(frame)-holdout < max(36, len(factors)*5):
        raise ValueError('Choose a longer research window: validation needs 36 training months plus the holdout.')
    train, test = frame.iloc[:-holdout], frame.iloc[-holdout:]
    full = fit_model(train, industry, factors)
    baseline = fit_model(train, industry, ['Mkt-RF'])
    actual = test[industry] - test['RF']
    prediction = full.predict(sm.add_constant(test[factors], has_constant='add'))
    market = baseline.predict(sm.add_constant(test[['Mkt-RF']], has_constant='add'))
    denominator = float(((actual - (train[industry]-train['RF']).mean())**2).sum())
    return {'train_end':train.index[-1], 'test_start':test.index[0], 'test_end':test.index[-1],
        'training':train, 'fitted':full, 'mae':float((actual-prediction).abs().mean()),
        'baseline_mae':float((actual-market).abs().mean()),
        'oos_r2':1-float(((actual-prediction)**2).sum())/denominator if denominator else float('nan'),
        'series':pd.DataFrame({'Actual':actual, 'Selected model':prediction, 'Market-only model':market})}


def performance(frame, industry):
    returns = frame[industry]
    wealth = pd.concat([pd.Series([1.], index=[frame.index[0]-pd.offsets.MonthEnd(1)]), (1+returns).cumprod()])
    excess = returns-frame['RF']
    return {'cagr':float(wealth.iloc[-1]**(12/len(returns))-1),
        'volatility':float(returns.std(ddof=1)*np.sqrt(12)),
        'sharpe':float(excess.mean()/excess.std(ddof=1)*np.sqrt(12)) if excess.std(ddof=1)>0 else float('nan'),
        'max_drawdown':float((wealth/wealth.cummax()-1).min())}


def growth(frame, industry):
    base = frame.index[0]-pd.offsets.MonthEnd(1)
    series = pd.DataFrame({'Industry':(1+frame[industry]).cumprod()*100,
                          'US market':(1+frame['Mkt-RF']+frame['RF']).cumprod()*100})
    return pd.concat([pd.DataFrame({'Industry':[100.], 'US market':[100.]}, index=[base]), series])


def research_note(frame, industry, model, fitted, metrics, validation=None):
    table = exposure_table(fitted)
    note = f'''# FactorLens research note

Portfolio: {INDUSTRIES[industry]} ({industry}); value weighted, monthly total returns.
Window: {frame.index[0]:%b %Y} to {frame.index[-1]:%b %Y}; {len(frame)} observations.
Model: {model}. Dependent variable: industry return minus the monthly risk-free return.
Inference: OLS with Newey-West / HAC standard errors, 3 lags; 95% confidence intervals.

Annualised intercept (linear, 12 × monthly): {fitted.params['const']*12:.2%}
Adjusted R²: {fitted.rsquared_adj:.3f}
CAGR: {metrics['cagr']:.2%}; volatility: {metrics['volatility']:.2%}
Maximum drawdown: {metrics['max_drawdown']:.2%}

## Coefficients (monthly intercept; factor slopes are unitless)
{table.to_csv()}

Source: Kenneth R. French Data Library, US five factors and 10 industry portfolios.
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

These are descriptive research portfolios, not investable strategies. Historical
returns may be revised. Factor attribution does not establish causation or future
alpha. No transaction costs are modelled. Alpha is not CAGR.
'''
    if validation:
        note += f'''\n## Chronological holdout
Training ends {validation['train_end']:%b %Y}; test begins {validation['test_start']:%b %Y}.
Holdout MAE: {validation['mae']:.2%}; market-only MAE: {validation['baseline_mae']:.2%}.
Test factor returns are contemporaneous explanatory inputs, not known-ahead signals.
This evaluates stability of attribution, not a tradable forecast.
'''
    return note
