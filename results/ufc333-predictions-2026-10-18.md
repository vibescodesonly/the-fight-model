# UFC 333 forecasts — October 18, 2026

Scheduled event: October 24, 2026, Abu Dhabi, United Arab Emirates.

Trained on 9,389 completed fights through October 17, 2026. These are
model probabilities without betting-odds blending. The two title fights
use five rounds and title context; the other nine use three rounds.

| Matchup | Predicted winner | Win probability | Predicted method |
|---|---|---:|---|
| Alexander Volkanovski vs. Movsar Evloev | Movsar Evloev | 55.6% | Decision |
| Petr Yan vs. Merab Dvalishvili | Merab Dvalishvili | 59.8% | Decision |
| Lone'er Kavanagh vs. Ramazan Temirov | Ramazan Temirov | 56.9% | KO/TKO |
| Alexander Volkov vs. Rizvan Kuniev | Alexander Volkov | 55.6% | Decision |
| Aaron Pico vs. Losene Keita | Aaron Pico | 52.7% | Decision |
| Azamat Murzakanov vs. Dominick Reyes | Dominick Reyes | 66.0% | KO/TKO |
| Nikita Krylov vs. Abdul Rakhman Yakhyaev | Abdul Rakhman Yakhyaev | 79.4% | Submission |
| Abus Magomedov vs. Cam Rowston | Cam Rowston | 57.1% | KO/TKO |
| Grant Dawson vs. Nurullo Aliev | Grant Dawson | 53.7% | Submission |
| Abubakar Vagaev vs. Jacob Malkoun | Jacob Malkoun | 73.8% | Decision |
| Colby Thicknesse vs. Asaf Chopurov | Colby Thicknesse† | 58.2% | Decision |

† Asaf Chopurov has no completed UFC history in the dataset. This bout
explicitly opts into the native generic debutant priors. Abubakar Vagaev
has only one recorded UFC fight, so his estimate also has limited history.

Method probabilities are conditional on the predicted winner. Full win
and method distributions, missing-history flags, and source fingerprints
are in [the JSON export](ufc333-predictions-2026-10-18.json).

The [card manifest](ufc333-card.json) records canonical dataset names,
round counts, title status, and location. Matchups were provided by the
user; the manifest links the UFCStats source for event date and location.

The core inference API uses the current date for recency features.
These probabilities were generated on October 18, 2026; a later run can
produce different values even with the same dataset.

The [dataset audit](data-update-2026-10-18.json) records all 15 added
events, checksums, metadata normalization, and validation. The serialized
production model is retained in the cloud environment. To rebuild models
from a fresh checkout, follow the [training instructions](../README.md#reproducing-results).
