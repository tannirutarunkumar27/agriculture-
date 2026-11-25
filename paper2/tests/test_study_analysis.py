import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper2.study_analysis import aggregate_complete_groups,mean_sd,summarize_study


class TestStudyAnalysis(unittest.TestCase):
    def setUp(self):
        self.protocol = {"seeds":[1,2,3],"neural_algorithms":["fedavg","fedprox","fednova"],
                         "reference_models":["centralized"],
                         "scenarios":[{"id":"iid","datasets":["toy"]}]}
        self.results = []
        for algorithm,offset in [("fedavg",0.0),("fedprox",0.01),("fednova",0.02),("centralized",0.03)]:
            for seed in self.protocol["seeds"]:
                value = 0.7+seed*0.01+offset
                self.results.append({"dataset":"toy","seed":seed,"scenario":"centralized_reference" if algorithm=="centralized" else "iid",
                                     "algorithm":algorithm,"test":{name:value for name in ["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","nll"]},
                                     "resources":{"total_local_sgd_steps":None if algorithm=="centralized" else 10,
                                                  "training_seconds":1.0,"aggregation_seconds":0.1,
                                                  "upload_tensor_bytes":100,"download_tensor_bytes":100}})

    def test_mean_sample_sd(self):
        mean,sd = mean_sd([1,2,3])
        self.assertEqual(mean,2)
        self.assertEqual(sd,1)

    def test_complete_groups_and_paired_units(self):
        groups,excluded,pairs,summary = aggregate_complete_groups(self.results,self.protocol)
        self.assertEqual(len(groups),4)
        self.assertFalse(excluded)
        self.assertEqual(len(pairs),15)
        prox = next(row for row in summary if row["target"]=="fedprox" and row["reference"]=="fedavg")
        self.assertAlmostEqual(prox["macro_f1_difference_mean_pp"],1.0)
        self.assertEqual(prox["seeds"],3)

    def test_partial_group_excluded_not_diluted(self):
        results = [r for r in self.results if not (r["algorithm"]=="fedprox" and r["seed"]==3)]
        groups,excluded,_,summary = aggregate_complete_groups(results,self.protocol)
        self.assertEqual(len(groups),3)
        self.assertEqual(excluded[0]["missing_seeds"],[3])
        self.assertFalse(any(row["target"]=="fedprox" for row in summary))

    def test_duplicate_seed_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_complete_groups(self.results+[self.results[0]],self.protocol)

    def test_unknown_seed_rejected(self):
        result = copy.deepcopy(self.results[0])
        result["seed"] = 999
        with self.assertRaises(ValueError):
            aggregate_complete_groups([result],self.protocol)

    def test_unknown_result_group_rejected(self):
        result = copy.deepcopy(self.results[0])
        result["algorithm"] = "unplanned_model"
        with self.assertRaises(ValueError):
            aggregate_complete_groups([result],self.protocol)

    def test_zero_observation_groups_report_all_missing_seeds(self):
        groups,excluded,_,_ = aggregate_complete_groups([],self.protocol)
        self.assertFalse(groups)
        self.assertEqual(len(excluded),4)
        self.assertTrue(all(row["missing_seeds"]==[1,2,3] for row in excluded))

    def test_conventional_unavailable_sgd_is_not_zero(self):
        groups,_,_,_ = aggregate_complete_groups(self.results,self.protocol)
        reference = next(row for row in groups if row["algorithm"]=="centralized")
        self.assertIsNone(reference["total_local_sgd_steps_mean"])


class TestAnalysisPolicies(unittest.TestCase):
    """Synthetic cached-result fixtures only; no fabricated research results."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frozen,self.study = self.root/"frozen",self.root/"study"
        self.frozen.mkdir()
        (self.study/"runs/fixture_1").mkdir(parents=True)
        self.identity = {"test_fixture":True}
        self.protocol = {"seeds":[1,2,3],"neural_algorithms":["fedavg","fedprox","fednova"],
                         "reference_models":["centralized"],"scenarios":[{"id":"iid","datasets":["toy"]}],
                         "statistics":"descriptive_test_fixture"}
        self.plan = [{"id":"fixture_1","dataset":"toy","seed":1,"partition":"partition.json"},
                     {"id":"fixture_2","dataset":"toy","seed":2,"partition":"partition.json"}]
        (self.frozen/"protocol.json").write_text(json.dumps(self.protocol),encoding="utf-8")
        (self.frozen/"run_plan.json").write_text(json.dumps(self.plan),encoding="utf-8")
        (self.frozen/"partition.json").write_text(json.dumps({"splits":{"train":[0],"validation":[1],"test":[2]}}),encoding="utf-8")
        (self.study/"study_binding.json").write_text(json.dumps({"identity":self.identity}),encoding="utf-8")
        for filename in ["completion.json","summary.json"]:
            (self.study/"runs/fixture_1"/filename).write_text("{}",encoding="utf-8")
        self.result = {"run_id":"fixture_1","dataset":"toy","seed":1,"scenario":"iid","algorithm":"fedavg", "best_round":1,
                       "test":{name:0.5 for name in ["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","nll"]},
                       "resources":{"total_local_sgd_steps":1,"training_seconds":1,"aggregation_seconds":0,
                                    "upload_tensor_bytes":1,"download_tensor_bytes":1},"regional_test":{}}
        self.result["test"] |= {"samples":1,"per_class":{}}
        self.output = self.root/"analysis"

    def tearDown(self):
        self.temp.cleanup()

    def summarize(self,**kwargs):
        with patch("paper2.study_analysis.verify_frozen_protocol",return_value={"status":"PASS"}), \
             patch("paper2.study_analysis.runner_identity",return_value=self.identity), \
             patch("paper2.study_analysis.Execution") as execution:
            execution.return_value.folder.side_effect = lambda run:self.study/"runs"/run["id"]
            execution.return_value.completed.return_value = copy.deepcopy(self.result)
            return summarize_study(self.frozen,self.study,self.output,**kwargs)

    def test_incomplete_study_default_refuses_analysis(self):
        with self.assertRaises(ValueError):
            self.summarize()
        self.assertFalse(self.output.exists())

    def test_partial_analysis_explicit_excludes_incomplete_groups(self):
        report = self.summarize(allow_partial=True)
        self.assertEqual(report["status"],"partial")
        self.assertEqual(report["complete_groups"],0)
        verification = json.loads((self.output/"verification.json").read_text())
        self.assertFalse(verification["test_inference_performed"])
        group = next(row for row in verification["excluded_incomplete_groups"] if row["algorithm"]=="fedavg")
        self.assertEqual(group["missing_seeds"],[2,3])
        self.assertEqual(len(verification["excluded_incomplete_groups"]),4)
