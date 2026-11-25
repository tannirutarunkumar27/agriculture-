"""Reduced-budget orchestration fixtures; real frozen-plan smoke is separate.

Only the expensive frozen-package verifier is stubbed for toy run-plan fixtures.
Actual training/evaluation/receipt/recovery code executes with real dataset rows.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper2.data import file_hash, load_dataset, make_partition
from paper2.study_protocol import BASELINE_SOURCES
from paper2.study_runner import AmbiguousEvaluationError, Execution, execute_study, runner_identity, single_writer
from paper2.train import checked_config, run as actual_train


class TestStudyRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frozen = self.root/"fixture_frozen"
        (self.frozen/"partitions").mkdir(parents=True)
        frame,schema,source = load_dataset("canonical")
        partition = make_partition(frame,schema,file_hash(source),"canonical",seed=123,clients=3)
        self.partition = self.frozen/"partitions/toy.json"
        self.partition.write_text(json.dumps(partition),encoding="utf-8")
        config = checked_config({"seed":123,"rounds":2,"hidden":8,"batch_size":128})
        self.plan = [{"id":"canonical_toy_fedavg","dataset":"canonical","seed":123,"scenario":"toy_iid",
                      "algorithm":"fedavg","partition":"partitions/toy.json","config":config,
                      "evaluate_test":True,"test_scope":"development_holdout"},
                     {"id":"canonical_toy_lr","dataset":"canonical","seed":123,"scenario":"centralized_reference",
                      "algorithm":"logistic_regression","partition":"partitions/toy.json",
                      "config":{"seed":123,"parameters":{"C":1,"max_iter":1000}},
                      "evaluate_test":True,"test_scope":"development_holdout"}]
        (self.frozen/"run_plan.json").write_text(json.dumps(self.plan),encoding="utf-8")
        sources = Path(__file__).resolve().parents[1]
        (self.frozen/"manifest.json").write_text(json.dumps({"code_hashes":{name:file_hash(sources/name) for name in BASELINE_SOURCES}}),encoding="utf-8")
        self.output = self.root/"study"

    def tearDown(self):
        self.temp.cleanup()

    def execute(self,**kwargs):
        with patch("paper2.study_runner.verify_frozen_protocol",return_value={"status":"PASS"}):
            return execute_study(self.frozen,self.output,**kwargs)

    def folder(self,index=0):
        return self.output/"runs"/self.plan[index]["id"]

    def test_dry_run_does_not_create_output(self):
        result = self.execute(dry_run=True)
        self.assertEqual(result["selected_runs"],2)
        self.assertFalse(self.output.exists())

    def test_bounded_run_and_resume_completed_without_test_inference(self):
        result = self.execute(max_runs=1)
        self.assertEqual(result["completed_runs"],1)
        self.assertEqual(result["status"],"partial")
        with patch.object(Execution,"predict_test",side_effect=AssertionError("Test must not run twice")):
            result = self.execute(run_ids=[self.plan[0]["id"]],max_runs=1,resume=True)
        self.assertEqual(result["executed_this_invocation"],0)
        self.assertEqual(result["completed_runs"],1)
        self.assertTrue((self.folder()/"completion.json").exists())

    def test_conventional_run(self):
        result = self.execute(run_ids=[self.plan[1]["id"]])
        self.assertEqual(result["completed_runs"],1)
        summary = json.loads((self.folder(1)/"summary.json").read_text())
        self.assertEqual(summary["test"]["samples"],330)
        self.assertEqual(summary["test_evaluation_attempts"],1)

    def test_completed_tampering_stops_without_retraining(self):
        self.execute()
        path = self.folder()/"summary.json"
        path.write_text("{}",encoding="utf-8")
        with patch("paper2.study_runner.train_neural",side_effect=AssertionError("No retraining")):
            with self.assertRaises(ValueError):
                self.execute(resume=True)
        state = json.loads((self.output/"study_progress.json").read_text())
        self.assertEqual(state["status"],"blocked_integrity")
        self.assertTrue(list((self.output/"study_errors").glob("error_*.json")))

    def test_unknown_run_rejected(self):
        with self.assertRaises(ValueError):
            self.execute(run_ids=["unknown"])
        self.assertFalse(self.output.exists())

    def test_no_overwrite_without_resume(self):
        self.execute()
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_training_failure_recorded_and_explicit_retry(self):
        with patch("paper2.study_runner.train_neural",side_effect=RuntimeError("injected pre-evaluation failure")):
            with self.assertRaises(RuntimeError):
                self.execute()
        self.assertTrue(list((self.folder()/"errors").glob("error_*.json")))
        self.assertFalse((self.folder()/"evaluation_journal.json").exists())
        with self.assertRaises(RuntimeError):
            self.execute(resume=True)
        result = self.execute(resume=True,retry_failed=True)
        self.assertEqual(result["completed_runs"],1)

    def test_incomplete_checkpoint_resumes_to_frozen_budget(self):
        def interrupt_after_one(path,output,algorithm,config,resume=False,evaluate_test=False):
            actual_train(path,output,algorithm,config|{"rounds":1},evaluate_test=False)
            raise RuntimeError("injected interruption after round one")
        with patch("paper2.study_runner.train_neural",side_effect=interrupt_after_one):
            with self.assertRaises(RuntimeError):
                self.execute()
        result = self.execute(resume=True,retry_failed=True)
        self.assertEqual(result["completed_runs"],1)
        checkpoint = torch.load(self.folder()/"attempts/attempt_0001/checkpoint.pt",weights_only=True)
        self.assertEqual(checkpoint["completed_round"],2)

    def test_final_round_checkpoint_without_summary_does_not_retrain(self):
        def interrupted_finalization(path,output,algorithm,config,resume=False,evaluate_test=False):
            actual_train(path,output,algorithm,config,evaluate_test=False)
            (Path(output)/"summary.json").unlink()  # temporary fixture only
            raise RuntimeError("injected post-training interruption")
        with patch("paper2.study_runner.train_neural",side_effect=interrupted_finalization):
            with self.assertRaises(RuntimeError):
                self.execute()
        with patch("paper2.study_runner.train_neural",side_effect=AssertionError("No retraining completed rounds")):
            result = self.execute(resume=True,retry_failed=True)
        self.assertEqual(result["completed_runs"],1)

    def test_committed_prediction_bundle_recovers_without_test_repeat(self):
        with patch.object(Execution,"cached_evaluation",side_effect=RuntimeError("injected export interruption")):
            with self.assertRaises(RuntimeError):
                self.execute()
        self.assertTrue((self.folder()/"test_bundle.npz").exists())
        self.assertFalse((self.folder()/"completion.json").exists())
        with patch.object(Execution,"predict_test",side_effect=AssertionError("No test repeat")):
            result = self.execute(resume=True,retry_failed=True)
        self.assertEqual(result["completed_runs"],1)

    def test_ambiguous_evaluation_blocks_even_with_retry(self):
        with patch.object(Execution,"predict_test",side_effect=RuntimeError("injected evaluation interruption")):
            with self.assertRaises(RuntimeError):
                self.execute()
        self.assertTrue((self.folder()/"evaluation_journal.json").exists())
        self.assertFalse((self.folder()/"test_bundle.npz").exists())
        with patch.object(Execution,"predict_test",side_effect=AssertionError("No automatic repeat")):
            with self.assertRaises(AmbiguousEvaluationError):
                self.execute(resume=True,retry_failed=True)

    def test_runner_identity_change_rejected(self):
        self.execute()
        identity = runner_identity(self.frozen)|{"runner_sha256":"changed"}
        with patch("paper2.study_runner.runner_identity",return_value=identity):
            with self.assertRaises(ValueError):
                self.execute(resume=True)

    def test_output_cannot_overwrite_frozen_package(self):
        with patch("paper2.study_runner.verify_frozen_protocol",return_value={"status":"PASS"}):
            with self.assertRaises(ValueError):
                execute_study(self.frozen,self.frozen/"study")

    def test_os_lock_excludes_second_writer(self):
        self.output.mkdir()
        with single_writer(self.output):
            with self.assertRaises(RuntimeError):
                with single_writer(self.output):
                    self.fail("Second writer entered")
