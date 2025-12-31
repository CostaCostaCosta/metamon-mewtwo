"""
Test to see if using a stale Result object causes legal actions to become empty.

Hypothesis: The metamon wrapper extracts legal masks BEFORE stepping, but uses
the Result from the PREVIOUS step, which might be causing issues.
