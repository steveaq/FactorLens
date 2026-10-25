# FactorLens

![Application preview](docs/preview.png)

An interactive factor-research desk built with Streamlit. Explore what explains
monthly returns across ten US industry portfolios, with uncertainty and model
limitations visible alongside the charts.

## Run

Python 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens immediately from a committed real-data snapshot. No keys, accounts,
or network calls are required on startup. **Data & method → Refresh** explicitly
downloads the official archives, validates both, and publishes an atomic local
bundle. A failed refresh preserves the existing dataset. The local live bundle is
Git-ignored; the committed snapshot is the deployment fallback.

## Research workflow

- Choose an industry, calendar-year window and five-, three-, or one-factor model.
- Inspect growth of $100, drawdowns, volatility and excess-return Sharpe.
- Read factor slopes with 95% HAC confidence intervals and trailing exposures.
- Compare all ten industries on a factor heatmap.
- Freeze coefficients before a chronological holdout and compare attribution
  error against a market-only model fitted on the same training months.
- Download the research note and exact monthly observations used in the analysis.

## Method and source

[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):
US five research factors and the **value-weighted monthly** block of the ten
industry portfolios. The snapshot starts in January 1990; provenance, retrieval
date and source archive hashes live in `data/snapshot.json`. Data is converted
from percent to decimal returns once, aligned by month, and checked for gaps.
Annual blocks, equal-weighted blocks and missing-data sentinels are excluded.

OLS fits industry return minus RF against an intercept and selected factors.
Confidence intervals use Newey–West/HAC standard errors with three lags.
Annualised alpha is **12 × the monthly intercept**, not compounded return or CAGR.
US market total return is Mkt−RF + RF. Maximum drawdown includes the starting
capital, so a loss in the first month is counted.

The holdout evaluates **explanatory stability**, using realised contemporaneous
factor returns. Those inputs are not known ahead of time; this is not a tradable
forecast or a simulated trading strategy. Research portfolios are not directly
investable, costs are not modelled, and historical data may be revised. The
dashboard is an analytical research tool, not a portfolio allocation service.

## Verify

```bash
python -m unittest test_research.py
```

Tests cover archive block parsing and units, recovery of known coefficients,
holdout isolation, the first-month drawdown edge case, total-market return
construction, snapshot completeness, and Streamlit launch/control changes.
GitHub Actions installs the declared dependencies and runs these checks.

## Deploy

In Streamlit Community Cloud, select this repository, branch `main`, and `app.py`.
Use Python 3.11. No secrets are required. Snapshot refreshes are local to the host
and may be lost when the service restarts. Community Cloud can hibernate after
inactivity; the optional `render.yaml` describes a paid always-on service.
Committing that configuration alone does not provision anything.

Design sources and acceptance criteria are in `DESIGN.md`. Public source datasets
retain their upstream terms and attribution; the code is MIT-licensed.
