import unittest
import numpy as np
import pandas as pd
from scipy.special import expit

from probability_calibration import (SymmetricPlatt,SymmetricIsotonic,MarketBlender,
    TemperatureCalibration,select_binary_calibrator,last_event_window,event_boundary)
from outcome_models import duration_distribution


class CalibrationTests(unittest.TestCase):
    def test_calibration_preserves_corner_symmetry(self):
        p = np.linspace(.02,.98,200)
        y = (np.random.default_rng(1).random(200) < p).astype(int)
        for model in (SymmetricPlatt(),SymmetricIsotonic()):
            model.fit(p,y)
            a = model.predict_proba(p.reshape(-1,1))[:,1]
            b = model.predict_proba((1-p).reshape(-1,1))[:,1]
            np.testing.assert_allclose(a+b,1,atol=1e-12)

    def test_calibration_can_retain_identity_without_fit_data(self):
        model,report = select_binary_calibrator(np.array([.5]*10),np.array([0,1]*5),
                                               np.array([.8,.2]),np.array([1,0]))
        self.assertIsNone(model);self.assertEqual(report["selected"],"none")

    def test_event_boundaries_never_split_events(self):
        dates = pd.to_datetime(["2020-01-01"]*3+["2020-01-02"]*4+["2020-01-03"]*3)
        self.assertEqual(event_boundary(dates,5),3)
        self.assertEqual(last_event_window(dates,5),3)

    def test_market_blender_is_unfitted_until_real_samples_are_supplied(self):
        blender = MarketBlender()
        self.assertFalse(blender.is_fitted)
        np.testing.assert_equal(blender.predict([.6],[.9]),[.6])
        with self.assertRaises(ValueError):blender.fit(pd.DataFrame())

    def test_learn_market_weight_and_reject_future_prices(self):
        dates = pd.date_range("2020-01-01",periods=120,tz="UTC")
        labels = np.arange(120)%2
        frame = pd.DataFrame({"event_date":dates,"event_start":dates+pd.Timedelta(hours=20),
            "generated_at":dates+pd.Timedelta(hours=12),"observed_at":dates+pd.Timedelta(hours=11),
            "probability":np.full(120,.5),"market_probability":np.where(labels,.9,.1),"outcome":labels})
        model = MarketBlender().fit(frame)
        self.assertGreater(model.market_weight,.8)
        broken = frame.copy();broken.loc[0,"observed_at"] = dates[0]+pd.Timedelta(hours=13)
        with self.assertRaisesRegex(ValueError,"before the event"):MarketBlender().fit(broken)

    def test_multiclass_temperature_is_normalized(self):
        calibration = TemperatureCalibration();calibration.temperature = 2.
        p = calibration.predict(np.array([[.1,.2,.7]]))
        self.assertAlmostEqual(p.sum(),1.)
        self.assertTrue((p>0).all())


class DurationTests(unittest.TestCase):
    def test_cause_masses_and_duration_reconcile_to_six_outcomes(self):
        hazards = np.tile([.85,.05,.04,.03,.03],(30,1))
        joint = np.array([.2,.25,.15,.1,.2,.1])
        d = duration_distribution(hazards,joint)
        cause = np.asarray(d["cause_probability_by_interval"])
        np.testing.assert_allclose(cause.sum(axis=0),joint[[1,4,2,5]],atol=1e-12)
        self.assertAlmostEqual(sum(d["duration_support_weights"]),1.)
        self.assertAlmostEqual(d["goes_distance_probability"],.3)
        self.assertTrue(0 < d["expected_seconds"] <= 900)
        p = list(d["over_rounds_probabilities"].values())
        self.assertTrue(all(a >= b for a,b in zip(p,p[1:])))

    def test_all_decision_mass_reaches_full_scheduled_duration(self):
        d = duration_distribution(np.tile([.9,.025,.025,.025,.025],(50,1)),
                                  [.6,0,0,.4,0,0],seconds=1500)
        self.assertEqual(d["expected_seconds"],1500)
        self.assertEqual(d["goes_distance_probability"],1.)
        self.assertTrue(all(p == 1. for p in d["over_rounds_probabilities"].values()))


if __name__ == "__main__":unittest.main()
