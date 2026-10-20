"""Append-only SQLite forecasts/quotes and pre-fight moneyline paper backtests."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pandas as pd

from probability_calibration import probability_metrics


def utc(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Timestamps must include a timezone.")
    return timestamp.tz_convert("UTC").isoformat(timespec="microseconds")


def now():
    return utc(datetime.now(timezone.utc))


def decimal_price(decimal=None, american=None):
    if decimal is not None and pd.notna(decimal):
        value = float(decimal)
    elif american is not None and pd.notna(american):
        number = float(american)
        if number == 0:
            raise ValueError("American odds cannot be zero.")
        value = 1+number/100 if number > 0 else 1+100/abs(number)
    else:
        raise ValueError("A price is required.")
    if not math.isfinite(value) or value <= 1:
        raise ValueError("Decimal odds must be finite and greater than one.")
    if decimal is not None and pd.notna(decimal) and american is not None and pd.notna(american):
        number = float(american)
        if number == 0 or not math.isfinite(number):
            raise ValueError("Invalid American odds.")
        converted = 1+number/100 if number > 0 else 1+100/abs(number)
        if not math.isclose(value,converted,abs_tol=1e-6):
            raise ValueError("Conflicting price formats; supply one format or consistent values.")
    return value


class MarketStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS fighters(id TEXT PRIMARY KEY, name TEXT NOT NULL, source_id TEXT UNIQUE);
        CREATE INDEX IF NOT EXISTS fighter_names ON fighters(name COLLATE NOCASE);
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, name TEXT NOT NULL, event_date TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS schedules(id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(id),
            starts_at TEXT NOT NULL, known_at TEXT NOT NULL, source TEXT NOT NULL, recorded_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fights(id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(id),
            a_id TEXT NOT NULL REFERENCES fighters(id), b_id TEXT NOT NULL REFERENCES fighters(id));
        CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, metadata TEXT NOT NULL, training_cutoff TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions(id TEXT PRIMARY KEY, fight_id TEXT NOT NULL REFERENCES fights(id),
            model_id TEXT NOT NULL REFERENCES models(id), generated_at TEXT NOT NULL,
            mode TEXT NOT NULL CHECK(mode IN ('prospective','retrospective')), payload TEXT NOT NULL, recorded_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS prediction_lookup ON predictions(model_id, fight_id, generated_at);
        CREATE TABLE IF NOT EXISTS quotes(id TEXT PRIMARY KEY, fight_id TEXT NOT NULL REFERENCES fights(id),
            bookmaker TEXT NOT NULL, market TEXT NOT NULL, selection TEXT NOT NULL, line REAL,
            decimal_odds REAL NOT NULL CHECK(decimal_odds > 1), observed_at TEXT NOT NULL,
            source TEXT NOT NULL, recorded_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS quote_lookup ON quotes(fight_id, market, bookmaker, observed_at);
        CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY, fight_id TEXT NOT NULL REFERENCES fights(id),
            winner TEXT NOT NULL CHECK(winner IN ('a','b','draw','no_contest')),
            known_at TEXT NOT NULL, source TEXT NOT NULL, recorded_at TEXT NOT NULL);
        """)
        for table in ("fighters", "events", "schedules", "fights", "models", "predictions", "quotes", "results"):
            self.connection.executescript(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
                BEGIN SELECT RAISE(ABORT, 'Append-only research history'); END;
                CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
                BEGIN SELECT RAISE(ABORT, 'Append-only research history'); END;
            """)
        self.connection.commit()

    def close(self):
        self.connection.close()

    def fighter(self, name, source_id=None):
        if source_id:
            row = self.connection.execute("SELECT id FROM fighters WHERE source_id=?", (source_id,)).fetchone()
            if row:
                return row["id"]
        rows = self.connection.execute("SELECT id,source_id FROM fighters WHERE name=? COLLATE NOCASE", (name,)).fetchall()
        if source_id is None:
            if len(rows) > 1:
                raise ValueError("Ambiguous fighter name; provide a stable source ID.")
            if rows:
                return rows[0]["id"]
        identifier = source_id or "local:"+uuid.uuid4().hex
        self.connection.execute("INSERT INTO fighters VALUES (?,?,?)", (identifier, name, source_id))
        self.connection.commit()
        return identifier

    def register_event(self, event_id, name, event_date, starts_at=None, known_at=None, source="user"):
        event_date = pd.Timestamp(event_date).date().isoformat()
        self.connection.execute("INSERT OR IGNORE INTO events VALUES (?,?,?)", (event_id, name, event_date))
        old = self.connection.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if old["event_date"] != event_date:
            raise ValueError("Changed event date requires an explicit new event/schedule version.")
        if starts_at is not None:
            start, known = utc(starts_at), utc(known_at or now())
            if known >= start:
                raise ValueError("A pre-fight schedule must have been known before its start.")
            identifier = hashlib.sha256(f"{event_id}|{start}|{known}|{source}".encode()).hexdigest()
            self.connection.execute("INSERT OR IGNORE INTO schedules VALUES (?,?,?,?,?,?)",
                                    (identifier, event_id, start, known, source, now()))
        self.connection.commit()

    def register_fight(self, fight_id, event_id, a, b, a_source_id=None, b_source_id=None):
        ai, bi = self.fighter(a, a_source_id), self.fighter(b, b_source_id)
        if ai == bi:
            raise ValueError("A fighter cannot face themselves.")
        self.connection.execute("INSERT OR IGNORE INTO fights VALUES (?,?,?,?)", (fight_id,event_id,ai,bi))
        row = self.connection.execute("SELECT * FROM fights WHERE id=?", (fight_id,)).fetchone()
        if (row["event_id"], row["a_id"], row["b_id"]) != (event_id, ai, bi):
            raise ValueError("Fight identity or corner order conflicts with existing history.")
        self.connection.commit()

    def register_model(self, identifier, metadata, training_cutoff):
        encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
        old = self.connection.execute("SELECT * FROM models WHERE id=?", (identifier,)).fetchone()
        if old and (old["metadata"] != encoded or old["training_cutoff"] != training_cutoff):
            raise ValueError("A model ID cannot be reused for different weights or training metadata.")
        self.connection.execute("INSERT OR IGNORE INTO models VALUES (?,?,?)", (identifier,encoded,training_cutoff))
        self.connection.commit()

    def _event(self, fight_id):
        row = self.connection.execute("SELECT e.* FROM events e JOIN fights f ON f.event_id=e.id WHERE f.id=?", (fight_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown fight ID.")
        return row

    def _start(self, fight_id, as_of=None):
        event = self._event(fight_id)
        query = "SELECT starts_at FROM schedules WHERE event_id=?"
        args = [event["id"]]
        if as_of:
            query += " AND known_at<=?"
            args.append(utc(as_of))
        query += " ORDER BY known_at DESC, rowid DESC LIMIT 1"
        row = self.connection.execute(query, args).fetchone()
        return row["starts_at"] if row else None

    def append_prediction(self, fight_id, model_id, payload, generated_at=None, mode="prospective"):
        timestamp = utc(generated_at or now())
        if pd.Timestamp(timestamp) > pd.Timestamp(now())+pd.Timedelta(minutes=5):
            raise ValueError("A forecast cannot have a future generation timestamp.")
        event = self._event(fight_id)
        start = self._start(fight_id, timestamp)
        # A date-only schedule gets a conservative earliest possible local-day
        # start (UTC+14); it can be archived, but cannot support odds backtests.
        bound = start or utc(pd.Timestamp(event["event_date"], tz="UTC")-pd.Timedelta(hours=14))
        if timestamp >= bound:
            raise ValueError("Forecast timestamp must precede the known event start/date bound.")
        model = self.connection.execute("SELECT training_cutoff FROM models WHERE id=?", (model_id,)).fetchone()
        if model is None or pd.Timestamp(model["training_cutoff"]).date() >= min(pd.Timestamp(bound).date(), pd.Timestamp(timestamp).date()):
            raise ValueError("Model training/history cutoff must precede the forecast and event dates.")
        p = payload.get("prob_a_model")
        if not isinstance(p, (int,float)) or not math.isfinite(p) or not 0 < p < 1:
            raise ValueError("Forecast requires a finite probability strictly between zero and one.")
        if mode == "prospective" and bound <= now():
            raise ValueError("Past-event imports must be explicitly marked retrospective.")
        identifier = uuid.uuid4().hex
        self.connection.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)", (
            identifier, fight_id, model_id, timestamp, mode,
            json.dumps(payload, allow_nan=False, sort_keys=True), now()))
        self.connection.commit()
        return identifier

    def append_quote(self, fight_id, bookmaker, selection, observed_at, source,
                     decimal_odds=None, american_odds=None, market="moneyline", line=None):
        timestamp = utc(observed_at)
        start = self._start(fight_id)
        if not start:
            raise ValueError("Exact event start is required before importing pre-fight quotes.")
        if pd.Timestamp(timestamp) > pd.Timestamp(now())+pd.Timedelta(minutes=5):
            raise ValueError("A quote cannot have been observed in the future.")
        if timestamp >= start:
            raise ValueError("Post-start/in-play prices are excluded.")
        if market == "moneyline" and selection not in ("a", "b"):
            raise ValueError("Moneyline selections must be a or b in the registered corner order.")
        price = decimal_price(decimal_odds, american_odds)
        line = None if line is None or pd.isna(line) else float(line)
        if line is not None and not math.isfinite(line):
            raise ValueError("Market lines must be finite.")
        identity = [fight_id,bookmaker,market,selection,line,timestamp,source]
        identifier = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        previous = self.connection.execute("SELECT decimal_odds FROM quotes WHERE id=?", (identifier,)).fetchone()
        if previous and not math.isclose(previous["decimal_odds"],price,abs_tol=1e-8):
            raise ValueError("Conflicting historical price; provide a distinct correction/source revision.")
        self.connection.execute("INSERT OR IGNORE INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?)", (
            identifier,fight_id,bookmaker,market,selection,line,price,timestamp,source,now()))
        self.connection.commit()
        return identifier

    def import_quotes(self, csv_path):
        frame = pd.read_csv(csv_path)
        required = {"fight_id","bookmaker","selection","observed_at","source"}
        if not required <= set(frame):
            raise ValueError("Odds CSV requires fight_id, bookmaker, selection, observed_at, source and a price column.")
        imported = 0
        for row in frame.to_dict("records"):
            if pd.notna(row.get("event_start")):
                existing = self.connection.execute("SELECT event_id FROM fights WHERE id=?",(row["fight_id"],)).fetchone()
                event_id = existing["event_id"] if existing else row.get("event_id")
                if not event_id:
                    raise ValueError("Unknown fight: supply event_id, event_name, event_date, fighter_a and fighter_b.")
                event = self.connection.execute("SELECT * FROM events WHERE id=?",(event_id,)).fetchone()
                self.register_event(event_id,event["name"] if event else row["event_name"],
                    event["event_date"] if event else row["event_date"],row["event_start"],
                    row.get("schedule_known_at") if pd.notna(row.get("schedule_known_at")) else row["observed_at"],row["source"])
                if not existing or (pd.notna(row.get("fighter_a")) and pd.notna(row.get("fighter_b"))):
                    self.register_fight(row["fight_id"],event_id,row["fighter_a"],row["fighter_b"])
            self.append_quote(**{key:value for key,value in row.items()
                if key in required | {"decimal_odds","american_odds","market","line"}})
            imported += 1
        return imported

    def append_result(self, fight_id, winner, known_at, source):
        timestamp = utc(known_at)
        if pd.Timestamp(timestamp) > pd.Timestamp(now())+pd.Timedelta(minutes=5):
            raise ValueError("A result cannot be known in the future.")
        start = self._start(fight_id)
        if not start or timestamp < start:
            raise ValueError("A result must be known after the event starts.")
        identifier = uuid.uuid4().hex
        self.connection.execute("INSERT INTO results VALUES (?,?,?,?,?,?)",
            (identifier,fight_id,winner,timestamp,source,now()))
        self.connection.commit()

    def import_results(self,csv_path):
        frame = pd.read_csv(csv_path)
        required = {"fight_id","winner","known_at","source"}
        if not required <= set(frame):
            raise ValueError("Results CSV requires fight_id, winner (a/b/draw/no_contest), known_at and source.")
        for row in frame.to_dict("records"):
            self.append_result(**{key:row[key] for key in required})
        return len(frame)

    def _prices(self, fight_id, when, bookmaker=None, max_age_hours=24):
        query = "SELECT * FROM quotes WHERE fight_id=? AND market='moneyline' AND observed_at<=?"
        args = [fight_id,utc(when)]
        if bookmaker:
            query += " AND bookmaker=?"
            args.append(bookmaker)
        query += " ORDER BY observed_at DESC, rowid DESC"
        prices = {}
        minimum = pd.Timestamp(when)-pd.Timedelta(hours=max_age_hours)
        for row in self.connection.execute(query,args):
            if pd.Timestamp(row["observed_at"]) < minimum:
                continue
            prices.setdefault((row["bookmaker"],row["selection"]),dict(row))
        return prices

    @staticmethod
    def _no_vig(prices, bookmaker, selection, max_pair_minutes=15):
        a, b = prices.get((bookmaker,"a")), prices.get((bookmaker,"b"))
        if not a or not b or abs((pd.Timestamp(a["observed_at"])-pd.Timestamp(b["observed_at"])).total_seconds()) > max_pair_minutes*60:
            return None
        pa,pb = 1/a["decimal_odds"],1/b["decimal_odds"]
        return pa/(pa+pb) if selection == "a" else pb/(pa+pb)

    def backtest(self, model_id, min_ev=.03, max_quote_age_hours=24, commission=0., as_of=None,
                 closing_max_age_minutes=30):
        """One unit, at most one bet per fight, earliest qualifying forecast.

        The rule is fixed before evaluation. Closing prices only enter CLV,
        never the offer selection or the simulated decision.
        """
        if not 0 <= commission < 1 or not math.isfinite(min_ev) or max_quote_age_hours <= 0 or closing_max_age_minutes <= 0:
            raise ValueError("Invalid strategy parameters.")
        as_of = utc(as_of or now())
        rows = self.connection.execute("SELECT * FROM predictions WHERE model_id=? AND generated_at<=? ORDER BY generated_at,rowid",
                                       (model_id,as_of)).fetchall()
        seen, bets, skipped, probabilities, scored = set(), [], {}, [], set()
        skip = lambda reason: skipped.__setitem__(reason, skipped.get(reason,0)+1)
        for row in rows:
            fight = row["fight_id"]
            if fight in seen:
                continue
            start = self._start(fight)
            if not start:
                skip("missing_exact_event_start"); continue
            if row["generated_at"] >= start:
                skip("forecast_after_start"); continue
            prices = self._prices(fight,row["generated_at"],max_age_hours=max_quote_age_hours)
            if not prices:
                skip("no_available_price"); continue
            payload = json.loads(row["payload"])
            pa = payload["prob_a_model"]
            result = self.connection.execute("SELECT winner,known_at FROM results WHERE fight_id=? AND known_at<=? ORDER BY known_at DESC,rowid DESC LIMIT 1",(fight,as_of)).fetchone()
            for book in sorted({key[0] for key in prices}):
                market_a = self._no_vig(prices,book,"a")
                if result and result["winner"] in ("a","b") and market_a is not None and (fight,book) not in scored:
                    probabilities.append((book,int(result["winner"] == "a"),pa,market_a))
                    scored.add((fight,book))
            options = []
            for (_,selection),quote in prices.items():
                p = pa if selection == "a" else 1-pa
                net_win = (quote["decimal_odds"]-1)*(1-commission)
                ev = p*net_win-(1-p)
                if ev >= min_ev:
                    options.append((ev,quote))
            if not options:
                skip("below_ev_threshold"); continue
            ev, quote = max(options,key=lambda item:(item[0],item[1]["decimal_odds"]))
            seen.add(fight)
            winner = result["winner"] if result else None
            profit = (None if winner is None else 0. if winner in ("draw","no_contest") else
                      (quote["decimal_odds"]-1)*(1-commission) if winner == quote["selection"] else -1.)
            placed_market = self._no_vig(prices,quote["bookmaker"],quote["selection"])
            closing = (self._prices(fight,pd.Timestamp(start)-pd.Timedelta(microseconds=1),
                       bookmaker=quote["bookmaker"],max_age_hours=closing_max_age_minutes/60) if start <= as_of else {})
            close_market = self._no_vig(closing,quote["bookmaker"],quote["selection"])
            bets.append({"fight_id":fight,"event_id":self._event(fight)["id"],
                "prediction_id":row["id"],"mode":row["mode"],"quote_id":quote["id"],
                "bookmaker":quote["bookmaker"],"selection":quote["selection"],
                "decimal_odds":quote["decimal_odds"],"model_ev":float(ev),"profit":profit,
                "settled_at":result["known_at"] if result else None,
                "settlement":"open" if winner is None else "void" if winner in ("draw","no_contest") else "settled",
                "no_vig_clv":close_market-placed_market if close_market is not None and placed_market is not None else None})
        settled = sorted([bet for bet in bets if bet["settlement"] == "settled"],
                         key=lambda bet:(bet["settled_at"],bet["fight_id"]))
        profit = sum(bet["profit"] for bet in settled)
        equity = np.r_[0,np.cumsum([bet["profit"] for bet in settled])]
        metrics = {"model_id":model_id,"strategy":{"min_ev":min_ev,"stake":"one unit",
            "max_bets_per_fight":1,"commission":commission,"max_quote_age_hours":max_quote_age_hours,
            "closing_max_age_minutes":closing_max_age_minutes},
            "bets":bets,"skipped":skipped,"settled_bets":len(settled),
            "void_bets":sum(b["settlement"] == "void" for b in bets),
            "open_bets":sum(b["settlement"] == "open" for b in bets),
            "profit_units":float(profit),"roi":float(profit/len(settled)) if settled else None,
            "maximum_drawdown_units":float(np.max(np.maximum.accumulate(equity)-equity)),
            "roi_interval":None,"mode":"paper simulation; no real wagers",
            "as_of":as_of,"evaluated_at":now(),
            "engine_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settlement_policy":"Two-way moneyline; draws and no-contests void. Other markets are stored but not backtested."}
        event_profits = {}
        for bet in settled:
            event_profits.setdefault(bet["event_id"],[]).append(bet["profit"])
        if len(event_profits) >= 20:
            groups = list(event_profits.values()); rng = np.random.default_rng(42); samples=[]
            for _ in range(1000):
                selected = [groups[i] for i in rng.integers(0,len(groups),len(groups))]
                samples.append(sum(map(sum,selected))/sum(map(len,selected)))
            metrics["roi_interval"] = {"lower":float(np.quantile(samples,.025)),
                "upper":float(np.quantile(samples,.975)),"method":"event-block bootstrap, 1000 samples",
                "events":len(groups)}
        else:
            metrics["roi_interval_reason"] = "Fewer than 20 settled events; no interval reported."
        if probabilities:
            metrics["probability_benchmarks"] = {}
            for bookmaker in sorted({row[0] for row in probabilities}):
                truth,model,market = map(np.array,zip(*[row[1:] for row in probabilities if row[0] == bookmaker]))
                metrics["probability_benchmarks"][bookmaker] = {"independent":probability_metrics(truth,model),
                                                               "market":probability_metrics(truth,market)}
        return metrics

    def market_training_samples(self, model_id, bookmaker):
        rows = self.connection.execute("SELECT * FROM predictions WHERE model_id=? ORDER BY generated_at,rowid",(model_id,)).fetchall()
        result,seen = [],set()
        for prediction in rows:
            fight = prediction["fight_id"]
            if fight in seen:
                continue
            outcome = self.connection.execute("SELECT winner FROM results WHERE fight_id=? ORDER BY known_at DESC,rowid DESC LIMIT 1",(fight,)).fetchone()
            start = self._start(fight)
            if not outcome or outcome["winner"] not in ("a","b") or not start:
                continue
            prices = self._prices(fight,prediction["generated_at"],bookmaker)
            market = self._no_vig(prices,bookmaker,"a")
            if market is None:
                continue
            seen.add(fight)
            result.append({"event_date":self._event(fight)["event_date"],
                "generated_at":prediction["generated_at"],"event_start":start,
                "observed_at":max(quote["observed_at"] for quote in prices.values()),
                "probability":json.loads(prediction["payload"])["prob_a_model"],
                "market_probability":market,"outcome":int(outcome["winner"] == "a")})
        return pd.DataFrame(result)
