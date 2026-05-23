"""
src/agent/prompts.py
--------------------
Week 7 — Agent System: Remediation Templates (Local Prompts)

WHAT THIS FILE DOES
-------------------
This file defines the structured text templates used by the RemediationAgent.
Because we are using a "Local Rule-Based Agent" approach to ensure HIPAA
compliance and avoid corporate firewall issues (like HuggingFace blocks),
we use sophisticated string templates rather than a local generative LLM.

These templates act as deterministic "prompts" that stitch together the
claim details and the policy chunks retrieved from the RAG layer.

WHY USE TEMPLATES INSTEAD OF A LOCAL LLM?
-----------------------------------------
1. Speed: String formatting is instant. Local LLMs (like Llama-3) are slow.
2. Consistency: Rule-based generation guarantees exactly the same format every time.
3. Security: Zero risk of hallucinations (e.g., an LLM fabricating policy rules).
4. Architecture: This keeps the exact same architectural boundary as an LLM agent,
   making it trivial to swap in AWS Bedrock in the future if required.
"""

# The main remediation plan structure
REMEDIATION_PLAN_TEMPLATE = """
### 📋 Remediation Plan: `{claim_id}`

#### 1. Summary
This claim has been flagged by the AI prediction model as having a high 
probability of denial. The following validation flags were triggered 
during the Silver layer data enrichment:
{flags_list}

#### 2. Root Cause Analysis (Policy Retrieval)
Based on the healthcare billing policies, here are the underlying rules 
that apply to these flags:

{policy_analysis}

#### 3. Required Actions
To remediate this claim before final submission to the payer, please 
complete the following steps:

{action_steps}
"""

# Template for formatting a single policy chunk retrieved from FAISS
POLICY_CHUNK_TEMPLATE = """
<details>
<summary>🚨 <b>Flag: {flag}</b> <i>(Source: {source_file})</i></summary>
<div style="padding: 12px; background-color: #1E293B; border-left: 4px solid #3B82F6; margin-top: 8px; margin-bottom: 8px; border-radius: 4px; font-size: 0.9em; font-family: monospace; white-space: pre-wrap;">
{policy_text}
</div>
</details>
"""

# Template for generating action steps based on specific flags
# This acts as our "heuristic intelligence", mimicking what an LLM would infer
FLAG_REMEDIATION_HEURISTICS: dict[str, str] = {
    "WARN_HIGH_BILLING": (
        "- REVIEW billed amount: Ensure it matches the hospital fee schedule.\n"
        "- ATTACH itemized receipts justifying the high cost (e.g., prolonged OR time, expensive implants).\n"
        "- ADD clinical notes confirming the medical necessity of the complex procedure."
    ),
    "ERR_INCOMPLETE_CLAIM": (
        "- FIX missing fields: Both diagnosis_code and procedure_code MUST be present.\n"
        "- VERIFY patient demographics and provider NPI."
    ),
    "WARN_MISSING_DIAGNOSIS": (
        "- ADD primary ICD-10 diagnosis code from the physician's clinical documentation.\n"
        "- ENSURE the diagnosis supports the medical necessity of the billed procedure."
    ),
    "WARN_MISSING_PROCEDURE": (
        "- ADD primary CPT/HCPCS procedure code.\n"
        "- VERIFY the operative report matches the selected code."
    ),
    "WARN_MISSING_AMOUNT": (
        "- ADD billed_amount.\n"
        "- CHECK the charge master/fee schedule for the correct pricing."
    ),
    "WARN_INVALID_DIAGNOSIS": (
        "- CORRECT the diagnosis_code: The current code is not a valid ICD-10 string.\n"
        "- CHECK for typographical errors (e.g., 'O' instead of '0')."
    ),
}

# Fallback heuristic if a new flag is encountered
DEFAULT_REMEDIATION_HEURISTIC = (
    "- REVIEW the claim against the referenced policy documents.\n"
    "- CORRECT any data entry errors and ensure all coding guidelines are met."
)
