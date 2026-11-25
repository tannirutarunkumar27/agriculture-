# Next remote-GPU research step

The next implementation task is now complete: an independent comparison tool
has been added at `restart/compare_matched_neural_studies.py`. After a remote
GPU run, it will compare the CPU audit and GPU audit by candidate ID, verify
identical optimizer-step/example accounting for paired runs, preserve distinct
execution-manifest hashes, and prohibit merging or selection.

Tests pass for paired comparison and work-mismatch rejection. No local training
is running. The only remaining external prerequisite for actual GPU execution
is a supplied SSH GPU server; the launcher and preflight are ready.
