"""
src/agent/remediator.py
-----------------------
Week 7 — Agent System: Remediation Logic

WHAT THIS FILE DOES
-------------------
This is the core Agent logic. It consumes the outputs from Week 5 (ML Model)
and Week 6 (RAG System) to produce a final, human-readable Remediation Plan.

Workflow:
  1. Receive a denied claim (dict) and its triggered validation flags (list[str]).
  2. Call `src.rag.retriever.retrieve_multiple(flags)` to get policy context.
  3. Format the RAG results and heuristic action steps into a clean text report.
  4. Return a structured `RemediationPlan` dataclass.

HIPAA & ARCHITECTURE
--------------------
By implementing this as a "Local Rule-Based Agent", we maintain strict HIPAA
data sovereignty (no data leaves the machine) and bypass corporate network
restrictions (no external API calls to AWS Bedrock or OpenAI required).

Usage:
    from src.agent.remediator import RemediationAgent
    
    agent = RemediationAgent()
    plan = agent.generate_plan(
        claim={"claim_id": "CLM-999"}, 
        flags=["WARN_HIGH_BILLING", "WARN_MISSING_DIAGNOSIS"]
    )
    print(plan.full_report)
"""

from dataclasses import dataclass
from typing import Any

from src.core.logger import get_logger
from src.core.exceptions import RAGError
from src.rag.retriever import retrieve_multiple, RetrievalResult
from src.agent.prompts import (
    REMEDIATION_PLAN_TEMPLATE,
    POLICY_CHUNK_TEMPLATE,
    FLAG_REMEDIATION_HEURISTICS,
    DEFAULT_REMEDIATION_HEURISTIC,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Data Structures                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class RemediationPlan:
    """
    Structured output from the RemediationAgent.
    
    Attributes:
        claim_id:    The ID of the claim being remediated.
        flags:       The list of triggered rules.
        sources:     List of source files pulled from RAG.
        full_report: The complete Markdown string ready for rendering.
        rag_evidence: Structured RAG results for building data tables.
    """
    claim_id: str
    flags: list[str]
    sources: list[str]
    full_report: str
    rag_evidence: list[dict[str, Any]] | None = None


# --------------------------------------------------------------------------- #
#  Agent Logic                                                                  #
# --------------------------------------------------------------------------- #

class RemediationAgent:
    """
    Local Rule-Based Agent for claim denial remediation.
    
    This agent coordinates the RAG retrieval and template generation to
    produce actionable steps for the billing analyst.
    """

    def __init__(self) -> None:
        """Initialize the agent."""
        logger.info("RemediationAgent initialized (Local Template Mode).")

    def generate_plan(self, claim: dict[str, Any], flags: list[str]) -> RemediationPlan:
        """
        Generate a comprehensive remediation plan for a denied claim.

        Args:
            claim: Dictionary containing the claim data (must include 'claim_id').
            flags: List of validation flag strings (e.g., ["WARN_HIGH_BILLING"]).

        Returns:
            A populated RemediationPlan dataclass.
        """
        claim_id = str(claim.get("claim_id", "UNKNOWN_CLAIM"))
        
        logger.info(
            "Generating remediation plan. claim_id=%s flag_count=%d",
            claim_id, len(flags),
        )

        if not flags:
            logger.warning("Agent called with empty flags for claim %s", claim_id)
            return RemediationPlan(
                claim_id=claim_id,
                flags=[],
                sources=[],
                full_report=f"CLAIM {claim_id}: No validation flags present. Claim is clean.",
            )

        # 1. Retrieve policy context from RAG layer
        try:
            rag_results = retrieve_multiple(flags, top_k=1)
        except RAGError as e:
            logger.error("RAG retrieval failed during agent generation: %s", e)
            # Graceful fallback: return a plan with heuristic steps only
            return self._build_fallback_plan(claim_id, flags)

        # 2. Format the policy analysis section and collect structured evidence
        policy_blocks = []
        sources_set = set()
        rag_evidence = []

        for flag, result in rag_results.items():
            if result.found:
                policy_blocks.append(
                    POLICY_CHUNK_TEMPLATE.format(
                        flag=flag,
                        source_file=result.source_file,
                        policy_text=result.top_text.strip(),
                    )
                )
                sources_set.add(result.source_file)
                score = result.chunks_with_scores[0][1] if result.chunks_with_scores else 0.0
                rag_evidence.append({
                    "reason_title": flag,
                    "source_name": result.source_file,
                    "similarity_score": round(score, 4),
                    "policy_preview": result.top_text.strip()[:100] + "..." if result.top_text else ""
                })
            else:
                policy_blocks.append(f"[Flag: {flag}] No specific policy document found.")

        policy_analysis_str = "\n".join(policy_blocks)

        # 3. Format the required action steps section
        action_blocks = []
        for flag in flags:
            heuristic = FLAG_REMEDIATION_HEURISTICS.get(flag, DEFAULT_REMEDIATION_HEURISTIC)
            action_blocks.append(f"For {flag}:\n{heuristic}\n")
            
        action_steps_str = "\n".join(action_blocks)

        # 4. Assemble the final report
        flags_list_str = "\n".join(f"- {f}" for f in flags)
        
        full_report = REMEDIATION_PLAN_TEMPLATE.format(
            claim_id=claim_id,
            flags_list=flags_list_str,
            policy_analysis=policy_analysis_str,
            action_steps=action_steps_str,
        )

        logger.info(
            "Remediation plan generated successfully. claim_id=%s sources=%d",
            claim_id, len(sources_set),
        )

        return RemediationPlan(
            claim_id=claim_id,
            flags=flags,
            sources=list(sources_set),
            full_report=full_report,
            rag_evidence=rag_evidence
        )

    def _build_fallback_plan(self, claim_id: str, flags: list[str]) -> RemediationPlan:
        """
        Build a plan when the RAG system is unavailable (e.g. index missing).
        Uses heuristics only, without policy excerpts.
        """
        flags_list_str = "\n".join(f"- {f}" for f in flags)
        
        action_blocks = []
        for flag in flags:
            heuristic = FLAG_REMEDIATION_HEURISTICS.get(flag, DEFAULT_REMEDIATION_HEURISTIC)
            action_blocks.append(f"For {flag}:\n{heuristic}\n")
            
        full_report = REMEDIATION_PLAN_TEMPLATE.format(
            claim_id=claim_id,
            flags_list=flags_list_str,
            policy_analysis="[RAG SYSTEM OFFLINE] Could not retrieve policy documents.",
            action_steps="\n".join(action_blocks),
        )

        return RemediationPlan(
            claim_id=claim_id,
            flags=flags,
            sources=[],
            full_report=full_report,
        )
