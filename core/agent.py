"""Smolagents pipeline for protein candidate design.

Implements a split workflow where one agent gathers literature and database
context, another agent reasons over that context to propose a candidate, and
the orchestration layer verifies the proposal against prior-art databases.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smolagents import CodeAgent, MCPClient
from mcp import StdioServerParameters
from core.smolagent_tools import (
    check_lipinski,
    #check_synthetic_accessibility,
    fasta_to_smiles,
    passes_pains_filter,
    pipeline_filter,
    run_remote_blastp_search,
    query_iedb,
    search_sabdab,
    web_search,
    visit_webpage,
)

FASTA_HEADER_RE = re.compile(r"^>[^\n]+", re.MULTILINE)
ROOT_DIR = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT_DIR / "traces"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgentConfig:
    """Configuration for the protein candidate agent."""

    target_fasta: str
    target_name: str | None = None
    identity_threshold: float = 0.80
    model_id: str = "o3-mini"
    research_model_id: str | None = None
    reasoning_model_id: str | None = None


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def validate_fasta(fasta_text: str) -> None:
    """Validate that a string looks like a single FASTA record."""
    if not isinstance(fasta_text, str) or not fasta_text.strip():
        raise ValueError("FASTA input must be a non-empty string.")
    if not FASTA_HEADER_RE.search(fasta_text):
        raise ValueError("FASTA input must contain a header line beginning with '>'.")


def normalize_fasta(fasta_text: str) -> str:
    """Normalize whitespace while preserving sequence content."""
    validate_fasta(fasta_text)
    lines = [line.rstrip() for line in fasta_text.strip().splitlines() if line.strip()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """\
You are the research agent in a two-agent protein design pipeline.

IMPORTANT: You are a CodeAgent. You MUST write Python code in every step.
Never output raw JSON. Always use the provided tool functions in your code.
Do ONE step per code block - do NOT combine all steps into a single block.

═══ MISSION ═══
Gather literature and database context for the target protein so a second
agent can reason about a novel binder design.

You MUST:
    - Search SAbDab for known binders.
    - Query IEDB for relevant epitopes.
    - Search the web for recent literature or annotations.
    - Use any MCP tools that are available to gather additional target context.
    - Summarize the most relevant findings for downstream reasoning.

Do NOT design a candidate sequence in this stage.

═══ TOOL RETURN FORMAT ═══
All tool functions return JSON strings, not Python dicts, when applicable.
Always parse tool output before using it.

═══ OUTPUT ═══
At the end of your code, assemble a Python dict named research_summary with
these keys:
    - target_name
    - target_overview
    - sabdab_hits
    - iedb_hits
    - web_findings
    - design_constraints
    - notable_epitopes

Print json.dumps(research_summary, indent=2, default=str).
"""


REASONING_SYSTEM_PROMPT = """\
You are the reasoning agent in a two-agent protein design pipeline.

IMPORTANT: You are a CodeAgent. You MUST write Python code in every step.
Never output raw JSON. Do not call external tools in this stage.
You may use the local structural screening tools after you draft a candidate
sequence.
Do ONE step per code block - do NOT combine all steps into a single block.

═══ MISSION ═══
Given the target FASTA and the research summary produced by the research
agent, design a novel peptide or mini-protein binder from scratch.

Hard constraints:
    - The candidate must be a genuinely novel binder, not a mutated target.
    - Do not copy, rearrange, or shuffle segments of the target sequence.
    - Use standard amino acid codes only.
    - Aim for a length between 40 and 120 residues.
    - Make the candidate plausibly compatible with the research findings.
    - After proposing a candidate_sequence, immediately run the local
      structural checks on the candidate FASTA before calling final_answer.
At the end of your code, assemble a Python dict named result with these keys:
    - candidate_sequence
    - design_rationale
    - design_notes

The design_notes should include the structural screening outcome.

If you have found a suitable candidate, call the final_answer tool with the
result JSON instead of continuing to reason. If you are not ready to finish
yet, print json.dumps(result, indent=2, default=str).
"""


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _create_model(model_id: str):
    """Create the LLM model for the CodeAgent.

    Tries OpenAIServerModel first (native), falls back to LiteLLMModel.
    Requires OPENAI_API_KEY in the environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it with: export OPENAI_API_KEY=sk-..."
        )

    try:
        from smolagents import OpenAIServerModel
        return OpenAIServerModel(model_id=model_id, api_key=api_key)
    except (ImportError, TypeError):
        from smolagents import LiteLLMModel
        return LiteLLMModel(
            model_id=f"openai/{model_id}",
            api_key=api_key,
        )


def _agent_prompt_templates(system_prompt: str) -> dict[str, dict[str, str] | str]:
    return {
        "system_prompt": system_prompt,
        "planning": {
            "initial_plan": "",
            "update_plan_pre_messages": "",
            "update_plan_post_messages": "",
        },
        "managed_agent": {"task": "", "report": ""},
        "final_answer": {"pre_messages": "", "post_messages": ""},
    }


def _safe_label(text: str | None) -> str:
    value = (text or "unknown").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _render_candidate_fasta(target_label: str, candidate_sequence: str) -> str:
    return f">candidate_{target_label}\n{candidate_sequence}\n"


def _normalise_candidate_sequence(candidate_sequence: str) -> str:
    if not isinstance(candidate_sequence, str):
        return ""
    return re.sub(r"\s+", "", candidate_sequence).upper().strip()


def _is_standard_amino_acid_sequence(candidate_sequence: str) -> bool:
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", candidate_sequence))


def _parse_agent_output(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw_output": output}
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_output": result}
    return {"raw_output": str(result)}


def _serialize_agent_steps(agent: CodeAgent) -> list[dict[str, Any]]:
    steps_serialized: list[dict[str, Any]] = []
    try:
        for step in agent.memory.steps:
            step_type = type(step).__name__
            try:
                step_dict = step.dict()
            except Exception:
                step_dict = {"raw": str(step)}
            step_dict["_step_type"] = step_type
            step_dict.pop("observations_images", None)
            if "model_input_messages" in step_dict:
                step_dict.pop("model_input_messages", None)
            steps_serialized.append(step_dict)
    except Exception as exc:
        steps_serialized.append({"error": str(exc)})
    return steps_serialized


def _build_reasoning_task(
    *,
    target_name: str,
    target_fasta: str,
    research_summary: dict[str, Any],
    attempt: int,
    previous_attempts: list[dict[str, Any]],
) -> str:
    context = json.dumps(research_summary, indent=2, default=str)
    rejection_notes = json.dumps(
        [
            {
                "attempt": entry.get("attempt"),
                "decision": entry.get("decision"),
                "verification": entry.get("verification"),
            }
            for entry in previous_attempts
        ],
        indent=2,
        default=str,
    )
    #40-120 residues for alphafold to remain viable, and to fit within token limits for reasoning
    return (
        f"Design a novel binder for the target protein using the research\n"
        f"summary below. This is attempt {attempt} of 3.\n\n"
        f"Target name: {target_name}\n"
        f"Target FASTA:\n```\n{target_fasta}```\n\n"
        f"Research summary:\n```json\n{context}\n```\n\n"
        f"Previous attempts and rejections (if any):\n"
        f"```json\n{rejection_notes}\n```\n\n"
        f"Return a JSON object with keys candidate_sequence, design_rationale,"
        f" and design_notes. The candidate_sequence must be a plain amino acid"
        f" string, 40-120 residues long, using standard amino acid codes only."
        f" After proposing the candidate_sequence, convert it to FASTA and run"
        f" the local structural checks before calling final_answer. If the"
        f" candidate is peptide-like, use the peptide-aware screening path."
        f" Include the screening outcome in design_notes. If you have a suitable"
        f" candidate, call the final_answer tool with the completed JSON object."
    )


def _verify_candidate(candidate_fasta: str, identity_threshold: float) -> dict[str, Any]:
    sabdab_result = json.loads(run_remote_blastp_search(candidate_fasta, "SAbDab"))
    iedb_result = json.loads(run_remote_blastp_search(candidate_fasta, "IEDB"))
    return {
        "identity_threshold": identity_threshold,
        "sabdab_max_identity": float(sabdab_result.get("max_identity", 0.0)),
        "iedb_max_identity": float(iedb_result.get("max_identity", 0.0)),
        "sabdab": sabdab_result,
        "iedb": iedb_result,
        "rejected": False,
    }


def _save_run_trace(trace: dict[str, Any], target_label: str) -> str:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    ts_label = timestamp.strftime("%Y%m%d_%H%M%S")
    log_path = TRACES_DIR / f"reasoning_trace_{target_label}_{ts_label}.json"
    log_path.write_text(
        json.dumps(trace, indent=2, default=str, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n[agent] Reasoning trace saved to {log_path}")
    return str(log_path)


def create_research_agent(model_id: str, mcp_tools: list | None = None) -> CodeAgent:
    """Instantiate the research agent with database and web tools."""
    model = _create_model(model_id)
    agent_tools = [search_sabdab, query_iedb, web_search, visit_webpage]
    if mcp_tools:
        agent_tools.extend(mcp_tools)

    return CodeAgent(
        name="research_agent",
        tools=agent_tools,
        model=model,
        planning_interval=4,
        max_steps=8,
        additional_authorized_imports=["json", "textwrap", "os", "sys", "subprocess", "requests"],
        executor_kwargs={"timeout_seconds": 300},
        prompt_templates=_agent_prompt_templates(RESEARCH_SYSTEM_PROMPT),
    )


def create_reasoning_agent(model_id: str) -> CodeAgent:
    """Instantiate the reasoning agent that designs a candidate from inputs."""
    model = _create_model(model_id)
    return CodeAgent(
        name="reasoning_agent",
        tools=[fasta_to_smiles, check_lipinski, passes_pains_filter, pipeline_filter],
        model=model,
        planning_interval=1,
        max_steps=16,
        additional_authorized_imports=["json", "textwrap", "os", "sys", "subprocess", "requests"],
        executor_kwargs={"timeout_seconds": 180},
        prompt_templates=_agent_prompt_templates(REASONING_SYSTEM_PROMPT),
    )


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

def run_agent(config: AgentConfig) -> str:
    """Run the full research -> reasoning -> verify pipeline.

    Returns the agent's JSON output as a string.
    """
    validate_fasta(config.target_fasta)
    target_fasta = normalize_fasta(config.target_fasta)
    target_name = config.target_name or "unknown_target"
    target_label = _safe_label(target_name)
    research_model_id = config.research_model_id or config.model_id
    reasoning_model_id = config.reasoning_model_id or config.model_id
    trace: dict[str, Any] = {
        "kind": "reasoning_trace",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": reasoning_model_id,
        "research_model_id": research_model_id,
        "reasoning_model_id": reasoning_model_id,
        "target_name": target_name,
        "identity_threshold": config.identity_threshold,
    }
    final_output: dict[str, Any] = {
        "target_seq_fasta": target_fasta,
        "candidate_seq_fasta": "",
        "decision": "error",
        "verification": {
            "identity_threshold": config.identity_threshold,
            "sabdab_max_identity": 0.0,
            "iedb_max_identity": 0.0,
            "rejected": True,
        },
    }
    trace_path = ""
    run_error: str | None = None

    server_parameters = StdioServerParameters(command="biomcp", args=["mcp"])
    
    with MCPClient(server_parameters) as mcp_tools:
        research_agent = create_research_agent(research_model_id, mcp_tools)
        reasoning_agent = create_reasoning_agent(reasoning_model_id)

        print(f"[agent] Starting protein candidate design for '{target_name}'...")
        print(f"[agent] Research model:  {research_model_id}")
        print(f"[agent] Reasoning model: {reasoning_model_id}")
        print(f"[agent] Identity threshold: {config.identity_threshold:.0%}")
        print()

        research_task = (
            f"Gather context for the target protein below. Use the available"
            f" research tools to search SAbDab, IEDB, the web, and any MCP"
            f" resources that help characterise the target and known binders.\n\n"
            f"Target name: {target_name}\n"
            f"Target FASTA:\n```\n{target_fasta}```\n\n"
            f"Return a compact JSON summary that can be used by a separate"
            f" reasoning agent. Do not design a candidate sequence in this"
            f" stage."
        )
        research_summary: dict[str, Any] = {}
        reasoning_attempts: list[dict[str, Any]] = []

        try:
            research_result = research_agent.run(research_task)
            research_summary = _parse_agent_output(research_result)
            trace["research_summary"] = research_summary

            final_candidate_sequence = ""
            final_verification: dict[str, Any] = {}
            final_decision = "rejected"

            for attempt in range(1, 4):
                reasoning_task = _build_reasoning_task(
                    target_name=target_name,
                    target_fasta=target_fasta,
                    research_summary=research_summary,
                    attempt=attempt,
                    previous_attempts=reasoning_attempts,
                )
                reasoning_result = reasoning_agent.run(reasoning_task)
                reasoning_payload = _parse_agent_output(reasoning_result)

                candidate_sequence = _normalise_candidate_sequence(
                    reasoning_payload.get("candidate_sequence", "")
                )
                reasoning_entry: dict[str, Any] = {
                    "attempt": attempt,
                    "raw_result": reasoning_payload,
                }

                if not candidate_sequence:
                    reasoning_entry["verification"] = {
                        "rejected": True,
                        "reason": "Reasoning agent did not return a valid candidate sequence.",
                    }
                    reasoning_attempts.append(reasoning_entry)
                    continue

                if not _is_standard_amino_acid_sequence(candidate_sequence):
                    reasoning_entry["verification"] = {
                        "rejected": True,
                        "reason": "Candidate contains non-standard amino acid codes.",
                    }
                    reasoning_attempts.append(reasoning_entry)
                    continue

                if not 40 <= len(candidate_sequence) <= 120:
                    reasoning_entry["verification"] = {
                        "rejected": True,
                        "reason": "Candidate length is outside the 40-120 residue window.",
                    }
                    reasoning_attempts.append(reasoning_entry)
                    continue

                candidate_fasta = _render_candidate_fasta(target_label, candidate_sequence)
                verification = _verify_candidate(candidate_fasta, config.identity_threshold)
                rejected = (
                    verification["sabdab_max_identity"] > config.identity_threshold * 100
                    or verification["iedb_max_identity"] > config.identity_threshold * 100
                )

                reasoning_entry["candidate_sequence"] = candidate_sequence
                reasoning_entry["candidate_seq_fasta"] = candidate_fasta
                reasoning_entry["verification"] = verification
                reasoning_entry["decision"] = "rejected" if rejected else "accepted"
                reasoning_attempts.append(reasoning_entry)

                if rejected:
                    continue

                final_candidate_sequence = candidate_sequence
                final_verification = verification
                final_decision = "accepted"
                break

            if not final_candidate_sequence and reasoning_attempts:
                last_successful = next(
                    (
                        attempt
                        for attempt in reversed(reasoning_attempts)
                        if attempt.get("candidate_sequence")
                    ),
                    None,
                )
                if last_successful:
                    final_candidate_sequence = last_successful["candidate_sequence"]
                    final_verification = last_successful.get("verification", {})
                    final_decision = last_successful.get("decision", "rejected")

            final_candidate_fasta = (
                _render_candidate_fasta(target_label, final_candidate_sequence)
                if final_candidate_sequence
                else ""
            )

            final_output = {
                "target_seq_fasta": target_fasta,
                "candidate_seq_fasta": final_candidate_fasta,
                "decision": final_decision,
                "verification": {
                    "identity_threshold": config.identity_threshold,
                    "sabdab_max_identity": final_verification.get("sabdab_max_identity", 0.0),
                    "iedb_max_identity": final_verification.get("iedb_max_identity", 0.0),
                    "rejected": final_decision != "accepted",
                },
            }
            trace["final_output"] = final_output
        except Exception as exc:
            run_error = str(exc)
            trace["run_error"] = run_error
            final_output["decision"] = "error"
            final_output["verification"]["error"] = run_error
        finally:
            trace["reasoning_attempts"] = reasoning_attempts
            trace["agent_logs"] = [
                {
                    "stage": "research",
                    "model_id": research_model_id,
                    "steps": _serialize_agent_steps(research_agent),
                    "output": research_summary,
                },
                {
                    "stage": "reasoning",
                    "model_id": reasoning_model_id,
                    "steps": _serialize_agent_steps(reasoning_agent),
                    "output": reasoning_attempts[-1] if reasoning_attempts else {},
                },
            ]
            trace_path = _save_run_trace(trace, target_label)

    final_output["trace_uri"] = trace_path
    return json.dumps(final_output, indent=2)


def load_fasta_file(path: str) -> str:
    """Load a FASTA record from disk."""
    return Path(path).read_text(encoding="utf-8")
