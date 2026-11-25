import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scoring import PredictionScorer
from storage import FixtureStorage, PredictionStorage, ScoreStorage


class PredictionScoringTests(unittest.TestCase):
    def test_derives_match_outcomes(self) -> None:
        scorer = PredictionScorer()
        self.assertEqual(scorer.actual_outcome("Home", "Away", 2, 0), "Home win")
        self.assertEqual(scorer.actual_outcome("Home", "Away", 1, 1), "Draw")
        self.assertEqual(scorer.actual_outcome("Home", "Away", 0, 3), "Away win")
        self.assertIsNone(scorer.actual_outcome("Home", "Away", "", 3))

    def test_scores_completed_predictions_once_and_builds_leaderboard(self) -> None:
        fixtures = pd.DataFrame(
            [
                {
                    "match_id": "1",
                    "match_date": "2026-11-24",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "home_score": "2",
                    "away_score": "1",
                    "status": "complete",
                },
                {
                    "match_id": "2",
                    "match_date": "2026-11-24",
                    "home_team": "Liverpool",
                    "away_team": "Everton",
                    "home_score": "0",
                    "away_score": "0",
                    "status": "finished",
                },
                {
                    "match_id": "3",
                    "match_date": "2026-11-24",
                    "home_team": "Tottenham",
                    "away_team": "Fulham",
                    "home_score": "",
                    "away_score": "",
                    "status": "scheduled",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            prediction_storage = PredictionStorage(Path(directory) / "predictions.csv")
            score_storage = ScoreStorage(Path(directory) / "scores.csv")
            prediction_storage.save_predictions(
                [
                    {
                        "match_id": "1",
                        "model": "gemma-id",
                        "prediction": "Arsenal win",
                        "prompt_version": "2",
                    },
                    {
                        "match_id": "2",
                        "model": "gemma-id",
                        "prediction": "Liverpool win",
                        "prompt_version": "2",
                    },
                    {
                        "match_id": "3",
                        "model": "gemma-id",
                        "prediction": "Fulham win",
                        "prompt_version": "2",
                    },
                ]
            )

            scorer = PredictionScorer(prediction_storage, score_storage)
            first = scorer.score_completed_fixtures(fixtures)
            second = scorer.score_completed_fixtures(fixtures)

            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            self.assertEqual(set(first["points"]), {0, 1})

            leaderboard = scorer.leaderboard()
            self.assertEqual(len(leaderboard), 1)
            self.assertEqual(int(leaderboard.iloc[0]["predictions"]), 2)
            self.assertEqual(int(leaderboard.iloc[0]["correct"]), 1)
            self.assertEqual(int(leaderboard.iloc[0]["points"]), 1)

    def test_storage_loads_zero_byte_and_malformed_csv_as_empty_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            fixture_storage = FixtureStorage(root / "fixtures.csv")
            prediction_storage = PredictionStorage(root / "predictions.csv")
            score_storage = ScoreStorage(root / "scores.csv")

            fixture_storage.path.write_bytes(b"")
            prediction_storage.path.write_bytes(b"")
            score_storage.path.write_bytes(b"")

            self.assertTrue(fixture_storage.load_matchday().empty)
            self.assertTrue(prediction_storage.load_predictions().empty)
            self.assertTrue(score_storage.load_scores().empty)

            prediction_storage.path.write_bytes(b"\xff\xfe\x00")
            score_storage.path.write_bytes(b"not,a,valid,score,csv\n")

            self.assertTrue(prediction_storage.load_predictions().empty)
            self.assertTrue(score_storage.load_scores().empty)


if __name__ == "__main__":
    unittest.main()
