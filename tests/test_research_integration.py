import json
import unittest

import numpy as np
import pandas as pd

import UFC_ML_Model as core
from outcome_models import JOINT_LABELS
from probability_calibration import event_boundary
from research_pipeline import FittedSystem,TrainingConfig
from temporal_features import build_frame


class ResearchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.read_csv(core.DATA_PATH).iloc[-200:].reset_index(drop=True)
        frame = build_frame(None,core,frame=data)
        valid = frame["meta"].method.map(core._normalize_method_label).isin(core.METHOD_LABELS)
        cls.x = frame["X"].loc[valid].reset_index(drop=True)
        cls.meta = frame["meta"].loc[valid].reset_index(drop=True)
        cls.y = (cls.meta.winner == "Red").astype(int)
        cls.cut = event_boundary(cls.meta.event_date,len(cls.meta)-30)
        config = TrainingConfig(validation_fights=50,minimum_training_fights=60,tree_limit=12,
                                outcome_iterations=12,duration_iterations=12,count_iterations=12)
        cls.system = FittedSystem(core,config).fit(cls.x.iloc[:cls.cut],cls.y.iloc[:cls.cut],
                                                 cls.meta.iloc[:cls.cut],log=lambda _:None)
        cls.tx = cls.x.iloc[cls.cut:].reset_index(drop=True)
        cls.tm = cls.meta.iloc[cls.cut:].reset_index(drop=True)
        cls.ty = cls.y.iloc[cls.cut:].reset_index(drop=True)

    def test_joint_probabilities_match_calibrated_winner_marginals(self):
        winner = self.system.winner.predict_proba_rows(self.tx)
        joint = self.system.methods.joint(self.tx,winner)
        np.testing.assert_allclose(joint.sum(axis=1),1,atol=1e-12)
        np.testing.assert_allclose(joint[:,:3].sum(axis=1),winner,atol=1e-12)
        np.testing.assert_allclose(joint[:,3:].sum(axis=1),1-winner,atol=1e-12)

    def test_corner_order_swaps_joint_labels_and_count_projections(self):
        x,m = self.tx.iloc[:2],self.tm.iloc[:2]
        p,j,d,c = self.system.predict(x,m)
        rp,rj,rd,rc = self.system.predict(core._swap_features(x),m)
        np.testing.assert_allclose(p+rp,1,atol=1e-10)
        np.testing.assert_allclose(j,rj[:,[3,4,5,0,1,2]],atol=1e-10)
        for i in range(len(x)):
            self.assertAlmostEqual(d[i]["expected_seconds"],rd[i]["expected_seconds"],places=8)
            for metric in ("td","sig_str"):
                self.assertAlmostEqual(c[i][f"a_{metric}"]["expected_count"],rc[i][f"b_{metric}"]["expected_count"],places=8)

    def test_full_evaluation_reports_joint_duration_and_count_metrics(self):
        score = self.system.evaluate(self.tx,self.ty,self.tm)
        self.assertEqual(score["rows"],len(self.tx))
        self.assertEqual(sum(score["joint_class_support"].values()),len(self.tx))
        self.assertTrue(np.isfinite(score["joint_log_loss"]))
        self.assertIn("direct_joint_challenger",score)
        self.assertIn("a_td_nll",score["count_scores"])
        self.assertIn("duration_thresholds",score)
        json.dumps(score,allow_nan=False)

    def test_duration_count_distributions_are_supported_and_monotonic(self):
        _,joint,durations,counts = self.system.predict(self.tx.iloc[:2],self.tm.iloc[:2])
        for i,d in enumerate(durations):
            self.assertAlmostEqual(d["goes_distance_probability"],joint[i,0]+joint[i,3])
            self.assertAlmostEqual(sum(d["duration_support_weights"]),1)
            for key in ("a_td","b_td","a_sig_str","b_sig_str"):
                ps = list(counts[i][key]["over_probabilities"].values())
                self.assertTrue(all(0 <= value <= 1 for value in ps))
                self.assertTrue(all(a >= b for a,b in zip(ps,ps[1:])))


if __name__ == "__main__":unittest.main()
