"""
Reflection module — self-checks agent responses for compliance, accuracy, and helpfulness
before returning to the user.
"""

import re
from dataclasses import dataclass, field
from openai import OpenAI


COMPLIANCE_CHECK_PROMPT = """You are a quality-assurance reviewer for an ecommerce customer service chatbot.
Review the following agent response for compliance, accuracy, and helpfulness.

Check for these issues:
1. **Unsupported claims**: Does the response promise anything not supported by the context? (refunds not in policy, price guarantees, etc.)
2. **Wrong information**: Does the response contradict the provided context or make up facts?
3. **Tone issues**: Is the response polite, professional, and helpful?
4. **Unsafe advice**: Does the response suggest anything risky (sharing passwords, unsafe actions)?
5. **Language match**: Does the response language match the user's question language?

Context/Knowledge:
{context}

User question: {user_message}

Agent response to review: {agent_response}

Return a JSON object with:
- "pass": true/false (overall, does it pass QA?)
- "issues": [list of specific issues found, empty if pass]
- "fixed_response": (if pass=false, provide a corrected version; if pass=true, return the original)
"""


@dataclass
class ReflectionResult:
    """Result of the reflection check."""
    passed: bool
    issues: list[str] = field(default_factory=list)
    fixed_response: str = ""


def reflect(
    user_message: str,
    agent_response: str,
    context: str = "",
    client: OpenAI | None = None,
) -> ReflectionResult:
    """
    Run a reflection (self-check) on the agent's response.

    Args:
        user_message: The original user question
        agent_response: The agent's draft response
        context: Retrieved context / knowledge used
        client: OpenAI client (optional; creates new if not provided)

    Returns:
        ReflectionResult with pass/fail and any corrections
    """
    if client is None:
        client = OpenAI()

    prompt = COMPLIANCE_CHECK_PROMPT.format(
        context=context[:2000],
        user_message=user_message,
        agent_response=agent_response,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000,
        )
        result = response.choices[0].message.content
        import json
        data = json.loads(result)

        return ReflectionResult(
            passed=data.get("pass", True),
            issues=data.get("issues", []),
            fixed_response=data.get("fixed_response", agent_response),
        )
    except Exception as e:
        # If reflection fails, pass through the original response
        return ReflectionResult(
            passed=True,
            issues=[f"Reflection check skipped (error: {str(e)[:100]})"],
            fixed_response=agent_response,
        )


# Simple fallback check that doesn't require an API call
def quick_reflect(agent_response: str) -> ReflectionResult:
    """
    Fast local checks without calling the LLM.
    Useful as a pre-filter before the full reflect() call.
    """
    issues = []

    # Check for empty or very short responses
    if len(agent_response.strip()) < 10:
        issues.append("Response is too short")

    # Check for common placeholder patterns
    if re.search(r"\[TODO\]|\[FIXME\]|\[INSERT|As an AI.*I (cannot|don't|am unable)", agent_response):
        issues.append("Response contains placeholder or refusal language")

    # Check for missing information markers
    if re.search(r"I don't know|I'm not sure|I cannot find", agent_response, re.IGNORECASE):
        issues.append("Response expresses uncertainty — may need escalation")

    return ReflectionResult(
        passed=len(issues) == 0,
        issues=issues,
        fixed_response=agent_response,
    )
