"""Claim verification module for validating paper claims."""

from typing import Tuple, Literal, Optional
from .reviewer_memory import Claim
from utils.helpers.llm import call_llm
from utils.helpers.logger import logger


# Verification prompt template
VERIFICATION_PROMPT_TEMPLATE = """You are an expert peer reviewer tasked with verifying a specific claim made in a research paper.

**Paper Context:**
{paper_context}

**Claim to Verify:**
ID: {claim_id}
Text: {claim_text}
Type: {claim_type}
Source Section: {claim_section}

**Your Task:**
Carefully analyze the claim against the provided paper context. Determine whether the claim is:
1. **supported**: The claim is well-supported by evidence in the paper
2. **weak**: The claim has partial support but lacks sufficient evidence or has minor issues
3. **invalid**: The claim is contradicted by the paper or lacks any supporting evidence

**Instructions:**
1. Look for direct evidence supporting or contradicting the claim
2. Consider whether the claim overstates the results
3. Check if the claim is consistent with the methodology and experiments
4. Evaluate whether the claim is adequately justified

**Additional Context (if any):**
{additional_context}

Provide your assessment in the following format:
Status: [supported/weak/invalid]
Reasoning: [Your detailed reasoning for this assessment, citing specific evidence from the paper]

Be thorough but concise. Focus on factual accuracy and evidence."""


def verify_claim(
    claim: Claim,
    paper_context: str,
    model: str,
    additional_context: Optional[str] = None
) -> Tuple[Literal["supported", "weak", "invalid"], str]:
    """Verify a claim using an LLM verifier.

    Args:
        claim: The Claim object to verify
        paper_context: Full or partial paper text for context
        model: Model identifier for the verifier LLM
        additional_context: Optional additional context to help verification

    Returns:
        Tuple of (status, reasoning) where status is one of "supported", "weak", "invalid"
    """
    logger.info(f"Verifying claim {claim.id}: {claim.text[:50]}...")

    # Build verification prompt
    prompt = VERIFICATION_PROMPT_TEMPLATE.format(
        paper_context=paper_context,
        claim_id=claim.id,
        claim_text=claim.text,
        claim_type=claim.type,
        claim_section=claim.section,
        additional_context=additional_context or "None"
    )

    # Call verifier LLM
    messages = [
        {"role": "system", "content": "You are an expert peer reviewer specializing in claim verification."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = call_llm(model=model, messages=messages, temperature=0.3)
        response_text = response.choices[0].message.content

        # Parse response
        status, reasoning = _parse_verification_response(response_text)

        logger.info(f"Verification result for {claim.id}: {status}")
        return status, reasoning

    except Exception as e:
        logger.error(f"Error during claim verification: {e}")
        # Fallback: mark as weak with error reasoning
        return "weak", f"Verification failed due to error: {str(e)}"


def _parse_verification_response(response_text: str) -> Tuple[Literal["supported", "weak", "invalid"], str]:
    """Parse the verifier's response to extract status and reasoning.

    Args:
        response_text: Raw response from the verifier LLM

    Returns:
        Tuple of (status, reasoning)
    """
    lines = response_text.strip().split('\n')

    status = None
    reasoning_lines = []
    in_reasoning = False

    for line in lines:
        line_lower = line.lower().strip()

        # Extract status
        if line_lower.startswith('status:'):
            status_text = line_lower.split('status:', 1)[1].strip()
            if 'supported' in status_text and 'weak' not in status_text:
                status = "supported"
            elif 'weak' in status_text:
                status = "weak"
            elif 'invalid' in status_text:
                status = "invalid"

        # Extract reasoning
        elif line_lower.startswith('reasoning:'):
            in_reasoning = True
            reasoning_text = line.split('reasoning:', 1)[1].strip()
            if reasoning_text:
                reasoning_lines.append(reasoning_text)
        elif in_reasoning and line.strip():
            reasoning_lines.append(line.strip())

    # Fallback if parsing fails
    if status is None:
        logger.warning("Could not parse status from verification response, defaulting to 'weak'")
        if 'supported' in response_text.lower() and 'weak' not in response_text.lower() and 'invalid' not in response_text.lower():
            status = "supported"
        elif 'invalid' in response_text.lower():
            status = "invalid"
        else:
            status = "weak"

    reasoning = " ".join(reasoning_lines) if reasoning_lines else response_text

    return status, reasoning


def batch_verify_claims(
    claims: list[Claim],
    paper_context: str,
    model: str
) -> list[Tuple[str, Literal["supported", "weak", "invalid"], str]]:
    """Verify multiple claims in batch.

    Args:
        claims: List of Claim objects to verify
        paper_context: Full or partial paper text for context
        model: Model identifier for the verifier LLM

    Returns:
        List of tuples (claim_id, status, reasoning) for each claim
    """
    results = []

    for claim in claims:
        status, reasoning = verify_claim(claim, paper_context, model)
        results.append((claim.id, status, reasoning))

    return results
