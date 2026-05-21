"""
src/agent/__init__.py
---------------------
Week 7 — Agent System

This module bridges the Machine Learning predictions (Week 5) and the 
RAG policy retrieval (Week 6) to generate actionable, human-readable 
remediation plans for denied claims.

The Agent layer is designed to be 100% local, utilizing Rule-based
templates rather than external cloud APIs to maintain strict HIPAA
compliance and ensure fast, reliable execution in restricted networks.

Usage:
    from src.agent.remediator import RemediationAgent, RemediationPlan
"""

from src.agent.remediator import RemediationAgent, RemediationPlan

__all__ = ["RemediationAgent", "RemediationPlan"]
