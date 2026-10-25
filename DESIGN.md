# FactorLens interface notes

The interface follows the typography, spacing and accessibility foundations of
Bergside/TypeUI's [Editorial skill](https://github.com/bergside/awesome-design-skills/blob/main/skills/editorial/SKILL.md).
Its layout is a research dashboard with six routed pages and persistent sidebar
filters. The opening page starts with results, without a promotional hero.

The copy was edited using the local Humanizer skill. Use plain technical language:
name the calculation, interpret the result, and explain a limitation when it affects
how the result can be used. Avoid slogans, staged questions and dramatic closers.
Do not add personal research claims or imply that an explanatory model is a forecast.

## Visual system

Paper #f7f5ee, ink #263c33, muted #63736a, green #205749, clay #b57350,
and borders #dedfd3. System sans throughout; 38px page titles and readable 12px
captions. Use the spacing and colors in styles.css, with native Streamlit controls.
Keep at least 44px button targets and visible keyboard focus.

## Pages and analysis

Overview provides returns and model context. Exposures separates slopes, rolling
fits and in-sample attribution. Risk covers distributions, tail losses, recovery
and correlation. Diagnostics tests residual and regressor assumptions. Validation
separates fixed holdouts from rolling/expanding refits. Methodology keeps equations,
source records, refresh and exports together.

Every chart needs units and a caption explaining its scope. Data tables and CSV
exports supplement the plots. Alpha intervals, diagnostic null hypotheses and
contemporaneous test-factor inputs must remain explicit.

## Check before release

- Navigate all six pages and change the shared filters.
- Confirm short windows show a useful message rather than a broken plot.
- Check attribution conservation and walk-forward training dates in the tests.
- Inspect desktop and narrow layouts, including sidebar navigation.
- Keep reduced-motion support and visible focus states.
