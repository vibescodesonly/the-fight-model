import sqlite3
import tempfile
from pathlib import Path
import unittest

from market_store import MarketStore,decimal_price


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = MarketStore(Path(self.directory.name)/"research.sqlite")
        self.store.register_event("event","Example","2020-01-02","2020-01-02T12:00:00Z", "2019-12-01T00:00:00Z")
        self.store.register_fight("fight","event","Alpha","Beta")
        self.store.register_model("model",{"version":"test"},"2019-12-31")

    def tearDown(self):self.store.close();self.directory.cleanup()

    def forecast(self,p=.6):
        return self.store.append_prediction("fight","model",{"prob_a_model":p},
                                            "2020-01-01T13:00:00Z",mode="retrospective")

    def quote(self,side,price,time="2020-01-01T12:00:00Z"):
        return self.store.append_quote("fight","Book",side,time,"fixture",decimal_odds=price)

    def test_prices_have_finite_valid_payouts(self):
        self.assertAlmostEqual(decimal_price(american=-150),1+100/150)
        self.assertEqual(decimal_price(american=150),2.5)
        for price in (1.,float("inf"),float("nan")):
            with self.assertRaises(ValueError):decimal_price(decimal=price)
        with self.assertRaisesRegex(ValueError,"Conflicting price"):
            decimal_price(decimal=2.4,american=-150)

    def test_history_is_append_only_including_underlying_sql(self):
        self.forecast()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute("UPDATE predictions SET generated_at='2020'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute("DELETE FROM predictions")

    def test_future_prices_do_not_influence_bet_selection_or_clv_entry(self):
        self.forecast();self.quote("a",2.4);self.quote("b",1.65)
        self.quote("a",4.,"2020-01-01T14:00:00Z")
        self.quote("b",1.3,"2020-01-01T14:00:00Z")
        self.quote("a",1.8,"2020-01-02T11:55:00Z")
        self.quote("b",2.1,"2020-01-02T11:55:00Z")
        self.store.append_result("fight","a","2020-01-02T16:00:00Z","fixture")
        report = self.store.backtest("model",as_of="2020-01-03T00:00:00Z")
        self.assertEqual(report["settled_bets"],1)
        self.assertEqual(report["bets"][0]["decimal_odds"],2.4)
        self.assertAlmostEqual(report["profit_units"],1.4)
        self.assertGreater(report["bets"][0]["no_vig_clv"],0)
        self.assertEqual(report["probability_benchmarks"]["Book"]["market"]["rows"],1)

    def test_in_play_quotes_and_future_trained_models_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"Post-start"):
            self.quote("a",2.,"2020-01-02T13:00:00Z")
        self.store.register_model("future",{},"2020-01-01")
        with self.assertRaisesRegex(ValueError,"cutoff"):
            self.store.append_prediction("fight","future",{"prob_a_model":.6},
                "2020-01-01T13:00:00Z",mode="retrospective")

    def test_draw_and_no_contest_return_stake_without_a_loss(self):
        self.forecast();self.quote("a",2.4);self.quote("b",1.65)
        self.store.append_result("fight","draw","2020-01-02T16:00:00Z","fixture")
        report = self.store.backtest("model",as_of="2020-01-03T00:00:00Z")
        self.assertEqual(report["void_bets"],1)
        self.assertEqual(report["profit_units"],0)
        self.assertIsNone(report["roi"])

    def test_missing_odds_do_not_produce_fake_market_performance(self):
        self.forecast()
        report = self.store.backtest("model")
        self.assertEqual(report["settled_bets"],0)
        self.assertEqual(report["skipped"]["no_available_price"],1)
        self.assertIsNone(report["roi"])

    def test_old_quotes_are_not_mislabelled_as_closing_prices(self):
        self.forecast();self.quote("a",2.4);self.quote("b",1.65)
        self.store.append_result("fight","a","2020-01-02T16:00:00Z","fixture")
        report = self.store.backtest("model",as_of="2020-01-03T00:00:00Z")
        self.assertIsNone(report["bets"][0]["no_vig_clv"])

    def test_conflicting_quote_is_not_silently_overwritten(self):
        first = self.quote("a",2.4)
        self.assertEqual(first,self.quote("a",2.4))
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            self.quote("a",2.5)

    def test_duplicate_predictions_never_create_multiple_bets_per_fight(self):
        self.forecast();self.forecast();self.quote("a",2.4);self.quote("b",1.65)
        self.store.append_result("fight","a","2020-01-02T16:00:00Z","fixture")
        report = self.store.backtest("model",as_of="2020-01-03T00:00:00Z")
        self.assertEqual(len(report["bets"]),1)
        self.assertEqual(report["probability_benchmarks"]["Book"]["independent"]["rows"],1)

    def test_date_only_events_can_be_archived_but_not_priced(self):
        self.store.register_event("date","Future","2030-01-01")
        self.store.register_fight("future-fight","date","Gamma","Delta")
        self.store.append_prediction("future-fight","model",{"prob_a_model":.6},
                                    mode="prospective")
        with self.assertRaisesRegex(ValueError,"Exact event start"):
            self.store.append_quote("future-fight","Book","a","2029-12-26T12:00:00Z","fixture",decimal_odds=2.)

    def test_csv_can_add_exact_schedule_and_quotes_to_archived_date_only_fight(self):
        import pandas as pd
        self.store.register_event("date","Example","2020-01-01")
        self.store.register_fight("new","date","Gamma","Delta")
        path = Path(self.directory.name)/"quotes.csv"
        pd.DataFrame([dict(fight_id="new",bookmaker="Book",selection="a",decimal_odds=2.4,
            observed_at="2019-12-28T12:00:00Z",source="fixture",event_start="2020-01-01T12:00:00Z")]).to_csv(path,index=False)
        self.assertEqual(self.store.import_quotes(path),1)
        self.assertEqual(self.store._start("new"),"2020-01-01T12:00:00.000000+00:00")

    def test_csv_result_ingestion_preserves_voids_and_known_time(self):
        import pandas as pd
        path = Path(self.directory.name)/"results.csv"
        pd.DataFrame([dict(fight_id="fight",winner="no_contest",known_at="2020-01-02T16:00:00Z",source="fixture")]).to_csv(path,index=False)
        self.assertEqual(self.store.import_results(path),1)
        self.assertEqual(self.store.connection.execute("SELECT winner FROM results").fetchone()[0],"no_contest")

    def test_future_generation_and_future_results_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"future generation"):
            self.store.append_prediction("fight","model",{"prob_a_model":.6},"2099-01-01T00:00:00Z",mode="retrospective")
        with self.assertRaisesRegex(ValueError,"known in the future"):
            self.store.append_result("fight","a","2099-01-01T00:00:00Z","fixture")


if __name__ == "__main__":unittest.main()
