import unittest

import pandas as pd

from paper2.restart.restricted_ranking_runner import rank


class RestrictedRankingTests(unittest.TestCase):
    def test_ranks_observed_packages_and_preserves_ties(self):
        packages = pd.DataFrame([
            {"block_id": "b", "management": "a", "system_species": "pea", "audit_total_seed_dm_t_ha": 2., "candidate_scope": "observed_management_package_only", "ranking_approved": False},
            {"block_id": "b", "management": "c", "system_species": "wheat", "audit_total_seed_dm_t_ha": 2., "candidate_scope": "observed_management_package_only", "ranking_approved": False},
        ])
        result = rank(packages, pd.DataFrame([{"block_id": "b", "gate_status": "APPROVED_INCLUDE"}]))
        self.assertEqual(result.descriptive_rank.tolist(), [1, 1])
        self.assertTrue(result.best_observed_package.all())

    def test_unapproved_blocks_are_rejected(self):
        packages = pd.DataFrame(columns=["block_id", "management", "system_species", "audit_total_seed_dm_t_ha", "candidate_scope", "ranking_approved"])
        with self.assertRaisesRegex(ValueError, "approved"):
            rank(packages, pd.DataFrame([{"block_id": "b", "gate_status": "PENDING_REVIEW"}]))


if __name__ == "__main__":
    unittest.main()
