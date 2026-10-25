# FactorLens design system

## Intent

A quantitative research notebook with the clarity and restraint of an investment
letter. Apply the MIT-licensed [Editorial skill](https://github.com/bergside/awesome-design-skills/blob/main/skills/editorial/SKILL.md)
from TypeUI/Bergside's Awesome Design Skills registry, adapted for analytical work.

## Foundations

Paper #f7f5ee; ink #263c33; muted #63736a; action green #205749;
clay comparison #b57350; borders #dedfd3. System sans for controls, Georgia for
the editorial hero and interpretation card. No font CDN. 8px spacing rhythm,
44px action targets, 3–4px corners. The orbital illustration is CSS.

## Components and interaction

The masthead and two-column hero introduce the research purpose and source scope.
The desk groups industry, model and calendar-year controls. Four metrics expose
alpha uncertainty and model fit before the overview, anatomy, validation and
method tabs. Every chart names its units and reference series. Tooltip details
supplement a visible interpretation and downloadable data.

Selectors have visible labels. Buttons support hover/focus, with a minimum 44px
target. Loading and source failures have explicit messages; a failed refresh
keeps the old data. Short research windows show a useful validation message.
Exports reflect the active filters. All external dynamic text is controlled
source metadata or escaped; arbitrary uploaded HTML is not rendered.

## Accessibility and responsiveness

Maintain AA text contrast and visible 3px keyboard focus. At 640px the hero and
control columns stack, typography scales down, and the footer wraps. Tabs remain
available horizontally. Native tables expose the chart's underlying data.
Honor reduced motion. Colour is reinforced by labels, line dashes and captions.

## Content rules

Use descriptive financial language. Distinguish alpha from CAGR, factor slopes
from portfolio weights, and attribution from a forecast. Show confidence intervals
and explain when they include zero. Avoid buy/sell signals, guarantees, fabricated
live status or charts whose labels imply transaction costs were included.

## QA

- Check desktop and narrow layouts for hero/card overflow and tab clipping.
- Change industry/model/window and verify metrics, plots and exports update.
- Verify short-window and source-refresh failure states.
- Run coefficient, holdout, drawdown and parser regressions.
- Verify every chart has units, source and a readable interpretation.
