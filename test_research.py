import io
import json
from pathlib import Path
import unittest
import zipfile
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from research import FACTORS, parse_monthly, fit_model, holdout_test, performance, growth, load_data
from analytics import risk_summary, drawdown_periods, factor_contributions, walk_forward, regression_diagnostics


def synthetic():
    rng=np.random.default_rng(7)
    frame=pd.DataFrame(rng.normal(0,.035,(180,5)),columns=FACTORS,
        index=pd.period_range('2000-01',periods=180,freq='M').to_timestamp('M'))
    frame['RF']=.001
    frame['HiTec']=frame.RF+.002+frame[FACTORS].to_numpy()@np.array([1.1,.2,-.3,.4,.1])
    return frame


class ResearchTests(unittest.TestCase):
    def test_parser_reads_only_first_monthly_block_and_converts_units(self):
        content='intro\n, A,B\n202001,1.0,-99.99\n202002,2.0,3.0\n\nAnnual Returns\n2020,999,999\n, A,B\n202001,50,50\n'
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as archive: archive.writestr('test.csv',content)
        parsed=parse_monthly(output.getvalue())
        self.assertEqual(len(parsed),2)
        self.assertAlmostEqual(parsed.iloc[0].A,.01)
        self.assertTrue(np.isnan(parsed.iloc[0].B))

    def test_recovers_known_factor_exposures_and_alpha(self):
        fitted=fit_model(synthetic(),'HiTec',FACTORS)
        np.testing.assert_allclose(fitted.params.to_numpy(),[.002,1.1,.2,-.3,.4,.1],atol=1e-10)

    def test_holdout_does_not_leak_test_returns_into_fit(self):
        frame=synthetic(); original=holdout_test(frame,'HiTec',FACTORS)
        altered=frame.copy();altered.loc[altered.index[-24:],'HiTec']+=.1
        changed=holdout_test(altered,'HiTec',FACTORS)
        np.testing.assert_allclose(original['fitted'].params,changed['fitted'].params)
        self.assertGreater(changed['mae'],original['mae']+.09)
        self.assertLess(original['train_end'],original['test_start'])

    def test_initial_loss_is_in_drawdown_and_market_adds_riskfree(self):
        frame=synthetic().iloc[:2].copy()
        frame['HiTec']=[-.1,0.];frame['Mkt-RF']=0.;frame['RF']=.01
        self.assertAlmostEqual(performance(frame,'HiTec')['max_drawdown'],-.1)
        self.assertAlmostEqual(growth(frame,'HiTec')['US market'].iloc[-1],102.01)

    def test_short_windows_rejected(self):
        with self.assertRaises(ValueError): fit_model(synthetic().iloc[:10],'HiTec',FACTORS)
        with self.assertRaises(ValueError): holdout_test(synthetic().iloc[:40],'HiTec',FACTORS,24)

    def test_snapshot_is_real_complete_and_finite(self):
        frame,meta=load_data()
        self.assertGreater(len(frame),300)
        self.assertFalse(frame.isna().any().any())
        self.assertEqual(len(frame),meta['rows'])
        self.assertEqual(len(frame),len(pd.period_range(frame.index[0],frame.index[-1],freq='M')))

    def test_app_launch_and_control_change(self):
        app=AppTest.from_file('app.py',default_timeout=45).run()
        self.assertFalse(app.exception)
        app.selectbox[0].select('Enrgy').run()
        app.selectbox[1].select('Market only').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Annualised' in m.label for m in app.metric))

    def test_historical_tail_loss_convention(self):
        result=risk_summary(pd.Series([-.12,-.08,-.04]+[.01]*9),.95)
        self.assertAlmostEqual(result['var'],.098)
        self.assertAlmostEqual(result['expected_shortfall'],.12)
        self.assertEqual(result['tail_months'],1)

    def test_drawdown_recovery_episodes_include_initial_capital(self):
        returns=pd.Series([-.1,1/9,-.2,.25],index=pd.period_range('2020-01',periods=4,freq='M').to_timestamp('M'))
        episodes=drawdown_periods(returns)
        self.assertEqual(len(episodes),2)
        np.testing.assert_allclose(sorted(episodes.Depth),[-.2,-.1])
        self.assertFalse(episodes.Recovery.isna().any())

    def test_attribution_reconstructs_observed_excess_return(self):
        frame=synthetic();fitted=fit_model(frame,'HiTec',FACTORS)
        parts=factor_contributions(frame,'HiTec',fitted,FACTORS)
        np.testing.assert_allclose(parts.sum(axis=1),frame.HiTec-frame.RF,atol=1e-12)

    def test_walk_forward_uses_only_prior_returns_for_each_fit(self):
        frame=synthetic();original=walk_forward(frame,'HiTec',FACTORS,60)
        changed=frame.copy();changed.loc[changed.index[60]:,'HiTec']+=.1
        altered=walk_forward(changed,'HiTec',FACTORS,60)
        self.assertAlmostEqual(original.estimate.iloc[0],altered.estimate.iloc[0])
        self.assertTrue((original.train_end<original.index).all())

    def test_diagnostic_vif_detects_overlapping_factors(self):
        frame=synthetic();frame['SMB']=frame['Mkt-RF']+.001*frame['SMB']
        fitted=fit_model(frame,'HiTec',FACTORS)
        diagnostic=regression_diagnostics(fitted,frame,FACTORS)
        self.assertGreater(diagnostic['vif'].set_index('Factor').loc['SMB','VIF'],100)
        self.assertEqual(len(diagnostic['tests']),3)

    def test_all_research_pages_render(self):
        app=AppTest.from_file('app.py',default_timeout=60).run()
        for name in ['exposures','risk','diagnostics','validation','methodology']:
            app.switch_page('views/'+name+'.py').run()
            self.assertFalse(app.exception,name)


if __name__=='__main__': unittest.main()
