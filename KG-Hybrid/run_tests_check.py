import unittest
import sys
from io import StringIO

loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_adaptive_fusion_gate.py')
stream = StringIO()
runner = unittest.TextTestRunner(stream=stream, verbosity=2)
result = runner.run(suite)

output = stream.getvalue()
print(output)
print(f"Was successful: {result.wasSuccessful()}")
print(f"Ran: {result.testsRun}, Errors: {len(result.errors)}, Failures: {len(result.failures)}")

with open('test_gate_results.txt', 'w', encoding='utf-8') as f:
    f.write(output)
    f.write(f"\nSUCCESS={result.wasSuccessful()}\n")
