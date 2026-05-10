from __future__ import annotations

from scripts.audit_artifacts import audit_doc_policy


def test_doc_policy_is_currently_satisfied():
    assert audit_doc_policy() == []
