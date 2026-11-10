"""Software correctness tests. Synthetic examples are not plant backtests."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from wind_forecast.core import (PowerCurve, load_config, read_turbines, summarize,
                                weighted_quantile, aggregate_farm, apply_operations)
from wind_forecast.weather import read_snapshot
from wind_forecast.learning import train_before, fit_weights, persistence_at_issue

ROOT = Path(__file__).resolve().parents[1]


class PhysicalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT/"config.json")
        self.curve = PowerCurve.read(ROOT/"data/power_curve.csv", self.cfg)
        self.turbines = read_turbines(ROOT/"data/turbines.csv")

    def test_curve_values_interpolation_and_shutdown(self):
        v = [2.99, 3, 5, 8.35, 13, 25, 25.001, 25.05, 26, 40]
        np.testing.assert_allclose(self.curve.evaluate(v), [0,22,339,1748,3600,3600,0,0,0,0], atol=1e-6)

    def test_high_wind_hysteresis(self):
        np.testing.assert_allclose(self.curve.trajectory([24,26,24,23,24]), [3600,0,0,3600,3600])
        np.testing.assert_allclose(self.curve.trajectory([26,24], False), [0,3600])

    def test_convert_members_before_averaging(self):
        self.assertNotAlmostEqual(float(self.curve.evaluate([4,10]).mean()), float(self.curve.evaluate([7])[0]))

    def test_family_weight_not_member_count(self):
        d = pd.DataFrame({"time":[0]*4, "family":["gfs","ecmwf","ecmwf","ecmwf"], "p":[0,100,100,100]})
        row = summarize(d, ["time"], "p", {"gfs":.5,"ecmwf":.5}).iloc[0]
        self.assertAlmostEqual(row.mean_power_kw, 50)
        self.assertEqual(row.q10_kw, 0); self.assertEqual(row.q90_kw, 100)

    def test_grid_cap_before_mean(self):
        issue = pd.Timestamp("2025-01-01T00:00Z"); valid = issue+pd.Timedelta(hours=1)
        rows=[]
        for member, p in [(0,0),(1,3600)]:
            for tid in self.turbines.turbine_id:
                rows.append(dict(issue_time_utc=issue,valid_time_utc=valid,family="gfs",member=member,
                                 turbine_id=tid,power_kw=p,potential_power_kw=p))
        f = aggregate_farm(pd.DataFrame(rows), self.turbines, self.cfg)
        self.assertEqual(f.export_power_kw.mean(),18000)
        self.assertEqual(f.gross_power_kw.mean(),23400)
        # A turbine omission must not understate farm output by silently summing 12.
        with self.assertRaises(ValueError): aggregate_farm(pd.DataFrame(rows[:-1]),self.turbines,self.cfg)

    def test_missing_family_is_error(self):
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame({"t":[1],"family":["gfs"],"p":[1]}),["t"],"p",self.cfg["family_weights"])

    def test_quantiles_preserve_mass_at_zero(self):
        self.assertEqual(weighted_quantile([0,0,100], [.5,.49,.01], .9),0)

    def test_outage_plan_and_future_information_guard(self):
        issue=pd.Timestamp("2025-01-01T00:00Z");valid=issue+pd.Timedelta(hours=1)
        w=pd.DataFrame([dict(turbine_id="01",family="gfs",member=0,issue_time_utc=issue,
                             valid_time_utc=valid,wind_speed_ref_ms=12.,wind_direction_deg=0.,reference_height_m=100)])
        t=self.turbines.iloc[:1]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"plan.csv"
            plan=pd.DataFrame([dict(valid_time_utc=valid,turbine_id="01",availability=0,
                                    power_limit_kw=3600,known_at_utc=issue)])
            plan.to_csv(p,index=False)
            d=apply_operations(w,t,self.cfg,self.curve,issue,p)
            self.assertEqual(d.power_kw.iloc[0],0)
            self.assertGreater(d.potential_power_kw.iloc[0],0)
            plan["known_at_utc"]=valid;plan.to_csv(p,index=False)
            with self.assertRaises(ValueError): apply_operations(w,t,self.cfg,self.curve,issue,p)
            plan["known_at_utc"]=pd.NaT;plan.to_csv(p,index=False)
            with self.assertRaises(ValueError): apply_operations(w,t,self.cfg,self.curve,issue,p)


class TemporalTests(unittest.TestCase):
    def test_purge_horizon_and_late_labels(self):
        origin=pd.Timestamp("2025-01-01T00:00Z");cutoff=origin+pd.Timedelta(hours=24)
        d=pd.DataFrame({"issue_time_utc":[origin]*3,
                        "valid_time_utc":[origin+pd.Timedelta(hours=h) for h in [1,25,2]],
                        "observed_at_utc":[origin+pd.Timedelta(hours=h) for h in [2,26,30]]})
        self.assertEqual(train_before(d,cutoff).index.tolist(),[0])

    def test_stacking_uses_constrained_weights(self):
        p=np.array([[0,10],[0,20],[0,30]],float)
        w=fit_weights(p,[10,20,30],[100,100,100])
        self.assertAlmostEqual(w.sum(),1);self.assertTrue((w>=0).all())
        self.assertGreater(w[1],.999)

    def test_persistence_uses_available_observations(self):
        issue=pd.Timestamp("2025-01-01T12:00Z")
        f=pd.DataFrame({"issue_time_utc":[issue],"turbine_id":["01"]})
        s=pd.DataFrame({"valid_time_utc":[issue-pd.Timedelta(hours=2),issue-pd.Timedelta(hours=1)],
                        "observed_at_utc":[issue-pd.Timedelta(hours=1),issue+pd.Timedelta(hours=1)],
                        "turbine_id":["01","01"],"power_kw":[100,500]})
        self.assertEqual(persistence_at_issue(f,s)[0],100)


class SavedSnapshotTests(unittest.TestCase):
    def test_snapshot_has_complete_coherent_ensemble(self):
        t=read_turbines(ROOT/"data/turbines.csv")
        w,grid,audit,m=read_snapshot(ROOT/"examples/weather_snapshot",t)
        for family in ["gfs","ecmwf"]:
            d=w.loc[w.family.eq(family)]
            self.assertEqual(len(d),len(t)*m["hours"]*audit[family]["members_used"])
            self.assertTrue(np.isfinite(d.wind_speed_ref_ms).all())
        self.assertTrue((w.valid_time_utc>pd.Timestamp(m["issue_time_utc"])).all())
        self.assertEqual(len(grid),39)

    def test_changed_coordinates_reject_replay(self):
        t=read_turbines(ROOT/"data/turbines.csv");t.loc[0,"latitude"]+=.1
        with self.assertRaises(ValueError):read_snapshot(ROOT/"examples/weather_snapshot",t)


if __name__ == "__main__": unittest.main()
