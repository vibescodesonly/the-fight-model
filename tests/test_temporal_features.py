import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import UFC_ML_Model as core
from temporal_features import ContextPriors, TemporalState, build_frame


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.read_csv(core.DATA_PATH)
        dates = pd.to_datetime(data.event_date,format="%m/%d/%Y")
        start = sorted(dates.unique())[-7]
        cls.data = data.loc[dates >= start].reset_index(drop=True)

    def test_future_outcomes_and_normalization_cannot_change_past_features(self):
        dates = pd.to_datetime(self.data.event_date,format="%m/%d/%Y")
        boundary = sorted(dates.unique())[3]
        prefix = self.data.loc[dates < boundary]
        past = build_frame(None,core,frame=prefix)
        future = self.data.copy()
        future.loc[dates >= boundary,"method"] = "Submission"
        future.loc[dates >= boundary,"r_kd"] = 10000
        for column in ("r_sig_str","r_td","r_sub_att","event_altitude","r_training_altitude_ft"):
            future.loc[dates >= boundary,column] = 10000
        full = build_frame(None,core,frame=future)
        assert_frame_equal(past["X"],full["X"].iloc[:len(past["X"])].reset_index(drop=True),check_exact=True)

    def test_same_date_features_are_frozen_before_all_results(self):
        first_date = self.data.event_date.iloc[0]
        event = self.data[self.data.event_date == first_date]
        before = build_frame(None,core,frame=event)
        changed = event.copy()
        changed["winner"] = changed.winner.map({"Red":"Blue","Blue":"Red"}).fillna("Draw")
        changed["r_sig_str"] = 10000
        after = build_frame(None,core,frame=changed)
        assert_frame_equal(before["X"],after["X"],check_exact=True)

    def test_live_and_historical_rows_use_identical_state_and_context(self):
        dates = pd.to_datetime(self.data.event_date,format="%m/%d/%Y")
        target_date = sorted(dates.unique())[3]
        prefix = self.data.loc[dates < target_date]
        state = build_frame(None,core,frame=prefix)["state"]
        event = self.data.loc[dates == target_date]
        row = event[event.winner.isin(["Red","Blue"])].iloc[0].to_dict()
        full = build_frame(None,core,frame=pd.concat([prefix,event]))
        expected = full["X"].loc[full["meta"].fight_url == row["fight_url"]].reset_index(drop=True)
        features,*_ = state.snapshot(row["r_name"],row["b_name"],target_date,row["weight_class"],
            row["gender"],row["total_rounds"],bool(row["is_title_bout"]),row["event_location"],row["event_altitude"],
            (core._extract_profile_from_row(row,"r"),core._extract_profile_from_row(row,"b")),
            (row["r_training_altitude_ft"],row["b_training_altitude_ft"]))
        assert_frame_equal(pd.DataFrame([features]),expected,check_exact=True)

    def test_birth_date_age_and_activity_advance_with_time(self):
        record = core._make_synthetic_fight_record(pd.Timestamp("2020-01-01"),
                    {"age":30,"date_of_birth":"1990-01-01"})
        a = core.compute_fighter_features([record],(core.MU_0,core.PHI_0,core.SIGMA_0),[],pd.Timestamp("2026-01-01"))
        self.assertAlmostEqual(a["age"],36.,delta=.01)
        self.assertEqual(a["days_inactive"],2192)
        self.assertEqual(a["fights_last_year"],0)
        self.assertAlmostEqual(a["age_squared"],a["age"]**2)

    def test_missing_birth_date_is_dated_age_estimate(self):
        record = core._make_synthetic_fight_record(pd.Timestamp("2020-01-01"),{"age":30})
        a = core.compute_fighter_features([record],(core.MU_0,core.PHI_0,core.SIGMA_0),[],pd.Timestamp("2026-01-01"))
        self.assertAlmostEqual(a["age"],36.,delta=.01)

    def test_gender_does_not_depend_on_prefix(self):
        features = core.compute_fighter_features([], (core.MU_0,core.PHI_0,core.SIGMA_0),[],pd.Timestamp("2026-01-01"))
        matchup = core.compute_matchup_features(features,features,weight_class="Bantamweight",gender="Women")
        self.assertEqual(matchup["gender_flag"],1.)

    def test_context_prior_does_not_read_future_and_matches_live(self):
        frame = pd.DataFrame({"event_date":pd.to_datetime(["2020-01-01","2021-01-01"]),
                              "method":["Decision","KO/TKO"]})
        before = core._precompute_context_finish_priors(frame.iloc[:1])[0][0]
        after = core._precompute_context_finish_priors(frame)[0][0]
        self.assertEqual(before,after)
        self.assertEqual(before,.55)
        prior = ContextPriors()
        row = dict(event_date=pd.Timestamp("2020-01-01"),weight_class="Bantamweight",
                   gender="Women",total_rounds=3,is_title_bout=0)
        prior.observe(row,"Submission")
        a = prior.predict("Bantamweight","Women",3,False,pd.Timestamp("2021-01-01"))
        b = prior.predict("Women's Bantamweight","female",3,False,pd.Timestamp("2021-01-01"))
        self.assertEqual(a,b)

    def test_old_fitted_model_cannot_silently_use_corrected_features(self):
        pipeline = core.UFCSuperModelPipeline()
        del pipeline._feature_version
        pipeline.model = object()
        pipeline.fighter_history = {};pipeline.glicko_ratings = {};pipeline.opp_glicko_list = {}
        with self.assertRaisesRegex(RuntimeError,"predates causal"):
            pipeline.predict_matchup("A","B")


if __name__ == "__main__":unittest.main()
