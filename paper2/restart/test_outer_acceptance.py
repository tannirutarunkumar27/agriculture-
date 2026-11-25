"""Remaining immutable-package and output-recovery acceptance checks."""
import argparse
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from paper2.restart.validation_search import ROOT,atomic_json, runtime, sha
from paper2.restart.outer_runner import CODE, verify_runner,verify_plan,execute


class AcceptanceTests(unittest.TestCase):
    def fresh_execution_package(self, directory):
        """Make a current isolated fixture without rewriting historical studies."""
        package = Path(directory) / "package"
        package.mkdir()
        snapshots = package / "source_snapshot"
        snapshots.mkdir()
        for source in CODE:
            shutil.copyfile(source, snapshots / source.name)
        plan = ROOT / "paper2/results/restart_outer_protocol_v1"
        record = {
            "plan_package": str(plan), "plan_manifest_sha256": sha(plan / "manifest.json"),
            "runtime": runtime(), "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in CODE},
        }
        atomic_json(package / "manifest.json", record)
        return package, record

    def test_execution_snapshot_changed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'paper2/results') as temp:
            copied, _ = self.fresh_execution_package(temp)
            (copied/'source_snapshot/outer_training.py').write_text('changed')
            with patch('paper2.restart.outer_runner.verify_plan', return_value=({}, {}, [])):
                with self.assertRaises(ValueError): verify_runner(copied)

    def test_plan_and_selection_changed(self):
        original=ROOT/'paper2/results/restart_outer_protocol_v1'
        for artifact in ('plan.json','selected.json','config.json'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temp:
                copied=Path(temp)/'package';shutil.copytree(original,copied)
                (copied/artifact).write_text('{}')
                with self.assertRaises(ValueError):verify_plan(copied)

    def test_output_binding_and_explicit_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'paper2/results') as temp:
            package, record = self.fresh_execution_package(temp)
            manifest,config,plan={"data": str(ROOT / "paper2/results/restart_public_data_v1")}, {}, []
            output=Path(temp)/'out';output.mkdir()
            atomic_json(output/'binding.json',{'identity':'changed'})
            with patch('paper2.restart.outer_runner.verify_runner',return_value=(record,manifest,config,plan)),patch('paper2.restart.outer_runner.coordinate',side_effect=AssertionError('must not train')):
                with self.assertRaises(ValueError):execute(package,output,dry_run=False)
                with self.assertRaises(ValueError):execute(package,output,resume=True,dry_run=False)

    def test_unbound_recovery_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'paper2/results') as temp:
            package, record = self.fresh_execution_package(temp)
            manifest,config,plan={"data": str(ROOT / "paper2/results/restart_public_data_v1")}, {}, []
            output=Path(temp)/'out';output.mkdir()
            with patch('paper2.restart.outer_runner.verify_runner',return_value=(record,manifest,config,plan)),patch('paper2.restart.outer_runner.coordinate',side_effect=AssertionError('must not train')):
                with self.assertRaises(ValueError):execute(package,output,resume=True,dry_run=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
        ('paper2.restart.test_outer_protocol','paper2.restart.test_outer_training',
         'paper2.restart.test_outer_runner','paper2.restart.test_outer_audit',
         'paper2.restart.test_outer_summary','paper2.restart.test_outer_acceptance'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
    atomic_json(args.output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'scope':'synthetic data plus read-only real frozen-package acceptance; no real model inference'})
    print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
