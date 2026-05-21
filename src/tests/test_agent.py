"""
src/tests/test_agent.py
-----------------------
Unit tests for the Agent system (Week 7).
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.remediator import RemediationAgent, RemediationPlan
from src.core.exceptions import RAGError
from src.rag.retriever import RetrievalResult
from src.rag.document_loader import PolicyChunk


@pytest.fixture
def agent() -> RemediationAgent:
    """Provide a fresh RemediationAgent instance for each test."""
    return RemediationAgent()


def test_agent_initialization(agent: RemediationAgent) -> None:
    """Test that the agent initializes correctly."""
    assert isinstance(agent, RemediationAgent)


def test_generate_plan_no_flags(agent: RemediationAgent) -> None:
    """Test that an empty flags list returns a clean claim message."""
    claim = {"claim_id": "CLM-CLEAN-01"}
    
    plan = agent.generate_plan(claim, flags=[])
    
    assert isinstance(plan, RemediationPlan)
    assert plan.claim_id == "CLM-CLEAN-01"
    assert plan.flags == []
    assert plan.sources == []
    assert "No validation flags present" in plan.full_report


@patch("src.agent.remediator.retrieve_multiple")
def test_generate_plan_single_flag(mock_retrieve_multiple: MagicMock, agent: RemediationAgent) -> None:
    """Test generating a plan for a single flag with successful RAG retrieval."""
    # Setup mock RAG response
    mock_result = RetrievalResult(
        query="test query",
        flag="WARN_HIGH_BILLING",
        top_text="If billed amount is high, review fee schedule.",
        source_file="billing_guidelines.txt",
        found=True
    )
    mock_retrieve_multiple.return_value = {"WARN_HIGH_BILLING": mock_result}
    
    claim = {"claim_id": "CLM-HIGH-02"}
    flags = ["WARN_HIGH_BILLING"]
    
    plan = agent.generate_plan(claim, flags)
    
    # Assertions
    mock_retrieve_multiple.assert_called_once_with(flags, top_k=1)
    assert plan.claim_id == "CLM-HIGH-02"
    assert "WARN_HIGH_BILLING" in plan.flags
    assert "billing_guidelines.txt" in plan.sources
    
    # Check that the report contains expected substrings
    assert "REMEDIATION PLAN FOR CLAIM: CLM-HIGH-02" in plan.full_report
    assert "WARN_HIGH_BILLING" in plan.full_report
    assert "billing_guidelines.txt" in plan.full_report
    assert "If billed amount is high" in plan.full_report
    assert "REVIEW billed amount" in plan.full_report  # from heuristic


@patch("src.agent.remediator.retrieve_multiple")
def test_generate_plan_multiple_flags(mock_retrieve_multiple: MagicMock, agent: RemediationAgent) -> None:
    """Test generating a plan for multiple flags."""
    # Setup mock RAG response
    mock_result_1 = RetrievalResult(
        query="test query 1",
        flag="WARN_HIGH_BILLING",
        top_text="High billing rule text.",
        source_file="billing_guidelines.txt",
        found=True
    )
    mock_result_2 = RetrievalResult(
        query="test query 2",
        flag="ERR_INCOMPLETE_CLAIM",
        top_text="Incomplete claim rule text.",
        source_file="incomplete_claim_policy.txt",
        found=True
    )
    mock_retrieve_multiple.return_value = {
        "WARN_HIGH_BILLING": mock_result_1,
        "ERR_INCOMPLETE_CLAIM": mock_result_2,
    }
    
    claim = {"claim_id": "CLM-MULTI-03"}
    flags = ["WARN_HIGH_BILLING", "ERR_INCOMPLETE_CLAIM"]
    
    plan = agent.generate_plan(claim, flags)
    
    # Assertions
    assert plan.claim_id == "CLM-MULTI-03"
    assert set(plan.flags) == {"WARN_HIGH_BILLING", "ERR_INCOMPLETE_CLAIM"}
    assert set(plan.sources) == {"billing_guidelines.txt", "incomplete_claim_policy.txt"}
    
    # Check that both flags and heuristics are in the report
    assert "High billing rule text." in plan.full_report
    assert "Incomplete claim rule text." in plan.full_report
    assert "FIX missing fields" in plan.full_report


@patch("src.agent.remediator.retrieve_multiple")
def test_generate_plan_rag_failure_fallback(mock_retrieve_multiple: MagicMock, agent: RemediationAgent) -> None:
    """Test that the agent gracefully falls back to heuristics if RAG fails."""
    # Simulate a missing FAISS index or embedding error
    mock_retrieve_multiple.side_effect = RAGError(1, "Index missing")
    
    claim = {"claim_id": "CLM-FALLBACK-04"}
    flags = ["WARN_MISSING_DIAGNOSIS"]
    
    plan = agent.generate_plan(claim, flags)
    
    # Assertions
    assert plan.claim_id == "CLM-FALLBACK-04"
    assert "WARN_MISSING_DIAGNOSIS" in plan.flags
    assert plan.sources == []  # No sources because RAG failed
    
    # Report should indicate RAG failure but still provide heuristic steps
    assert "[RAG SYSTEM OFFLINE]" in plan.full_report
    assert "ADD primary ICD-10 diagnosis code" in plan.full_report  # heuristic still there


@patch("src.agent.remediator.retrieve_multiple")
def test_generate_plan_rag_not_found(mock_retrieve_multiple: MagicMock, agent: RemediationAgent) -> None:
    """Test behavior when RAG succeeds but finds no matching policy for a flag."""
    mock_result = RetrievalResult(
        query="weird flag",
        flag="UNKNOWN_FLAG",
        found=False
    )
    mock_retrieve_multiple.return_value = {"UNKNOWN_FLAG": mock_result}
    
    claim = {"claim_id": "CLM-UNK-05"}
    
    plan = agent.generate_plan(claim, flags=["UNKNOWN_FLAG"])
    
    assert plan.sources == []
    assert "No specific policy document found." in plan.full_report
    assert "REVIEW the claim against the referenced policy documents" in plan.full_report  # default heuristic
