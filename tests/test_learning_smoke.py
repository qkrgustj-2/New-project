"""Exercise the complete training/save/predict path using synthetic test fixtures.

No synthetic accuracy metric is evidence of performance at Nakasato.
"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from wind_forecast.learning import NUMERIC, train, predict_saved


class LearningWorkflowTest(unittest.TestCase):
    def test_train_holdout_refit_and_prediction(self):
        rows, observations = [], []
        for origin in pd.date_range("2025-01-01", periods=24, freq="D", tz="UTC"):
            for tid in ["01", "02"]:
                for lead in [1,2,3]:
                    valid = origin+pd.Timedelta(hours=lead)
                    r = {key:1. for key in NUMERIC}
                    physics = 1000+250*np.sin(origin.day/4)+100*lead
                    r.update(issue_time_utc=origin,valid_time_utc=valid,turbine_id=tid,
                             lead_hours=lead,physics_mean_kw=physics,physics_std_kw=100,
                             wind_mean_ms=7,wind_std_ms=1,rated_power_kw=3600,
                             availability=1,power_limit_kw=3600,gfs_power_kw=physics+100,
                             ecmwf_power_kw=physics-100,latitude=41,longitude=140.45,hub_height_m=116.5)
                    rows.append(r)
                    observations.append(dict(valid_time_utc=valid,turbine_id=tid,
                                             observed_at_utc=valid+pd.Timedelta(hours=1),
                                             power_kw=.8*physics+30*int(tid)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=root/"features.csv"; s=root/"scada.csv"
            pd.DataFrame(rows).to_csv(f,index=False);pd.DataFrame(observations).to_csv(s,index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                report=train(str(f),s,root/"trained",min_train_rows=20)
            self.assertAlmostEqual(sum(report["weights"].values()),1)
            self.assertGreater(report["holdout_rows"],0)
            cutoff=pd.Timestamp(report["holdout_first_issue_utc"])
            for fold in report["folds"]:
                if fold["status"]=="used":
                    self.assertLess(pd.Timestamp(fold["max_training_valid_time_utc"]),pd.Timestamp(fold["validation_first_issue_utc"]))
                    self.assertLessEqual(pd.Timestamp(fold["max_training_label_available_utc"]),pd.Timestamp(fold["validation_first_issue_utc"]))
            future=pd.DataFrame(rows[-6:]);future["issue_time_utc"]+=pd.Timedelta(days=40)
            future["valid_time_utc"]+=pd.Timedelta(days=40)
            future.to_csv(root/"future.csv",index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result=predict_saved(root/"trained/model.pkl",root/"future.csv",root/"predictions.csv")
            self.assertTrue(result.ml_ensemble_power_kw.between(0,3600).all())
            with self.assertRaises(ValueError):
                predict_saved(root/"trained/model.pkl",f,root/"leaked.csv")


if __name__=="__main__":unittest.main()
