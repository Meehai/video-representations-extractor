# Flaky: test_RepresentationMetadata_add_time_in_threads (concurrent metadata.json writes)

**Created**: 2026-11-19
**Priority**: 2
**Closed**: n/a

**Labels**: bug, core

## Description

`test/vre/unit/test_representation_metadata.py::test_RepresentationMetadata_add_time_in_threads` fails
intermittently in CI (`vre_unit_integration`, pipeline #3088923351 on !254, failed after 3 reruns; passed on retry).

4 threads, each with its own `RepresentationMetadata` on the same `metadata.json`, `add_time(..., sync=True)`.
Log shows repeated `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`: one writer reads the file
while another has truncated it but not finished writing. Then a run_stats entry gets lost -> assert mismatch.

The test is right (N writers on one file is the real use: N representations / processes); the sync in
`RepresentationMetadata` is not. Fix in `vre/`: atomic write (tmp file + `os.replace`) and/or a file lock around
read-merge-write, then drop the non-strict xfail on the test.
