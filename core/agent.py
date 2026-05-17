"""Smolagents CodeAgent for protein candidate design.

Implements a Plan-Execute-Verify loop where the agent:
1. Analyses the target protein FASTA sequence.
2. Searches prior-art databases (SAbDab, IEDB) for context.
3. Generates a novel candidate protein sequence.
4. Verifies the candidate against local MMseqs2 databases.
5. Rejects any candidate with >80% identity to known binders.
6. Saves reasoning traces for later analysis.
7. Outputs strict JSON with target and candidate FASTA sequences.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smolagents import CodeAgent

from core.smolagent_tools import (
    run_remote_blastp_search,
    save_reasoning_trace,
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

SYSTEM_PROMPT = """\
You are a protein design assistant. You write Python code to accomplish tasks.

IMPORTANT: You are a CodeAgent. You MUST write Python code in every step. Never
output raw JSON. Always use the provided tool functions in your code.
Do ONE step per code block — do NOT combine all steps into a single block.

═══ MISSION ═══
Design a novel peptide or mini-protein that could plausibly BIND TO a given
target protein. The candidate is a BINDER — it is a completely new protein
designed to interact with the target, NOT a variant or mutated copy of the
target itself.

The candidate must NOT be a rediscovery of an existing known binder. All
database queries hit live remote APIs (SAbDab, IEDB, NCBI BLASTp) to ensure
the most up-to-date prior-art data.

═══ CRITICAL DESIGN CONSTRAINT ═══
The candidate sequence MUST be a genuinely novel protein designed FROM SCRATCH.
  - DO NOT start from the target sequence and introduce point mutations.
  - DO NOT copy, rearrange, or shuffle segments of the target sequence.
  - The target protein is what you are designing a binder FOR, not what you
    are designing a variant OF.
  - Think about what structural features, motifs, and residue compositions
    would allow a peptide to interact with the target's known binding sites
    or epitopes. Design based on binding principles, not sequence similarity.

═══ TOOL RETURN FORMAT ═══
All tool functions return JSON STRINGS, not Python dicts. You MUST parse them:
    result_str = run_remote_blastp_search(fasta, "SAbDab")
    result = json.loads(result_str)
    max_id = result["max_identity"]
NEVER use placeholder or fallback values for max_identity. If parsing fails,
print the raw result for debugging and treat it as an error.

═══ WORKFLOW (one code block per step) ═══

STEP 1 — Gather context from live databases AND the web:
  Call search_sabdab(target_name) and query_iedb(endpoint="epitope_search",
  search_field="parent_source_antigen_names::text",
  search_value=f"ilike.*{{target_name}}.*",
  select_fields="structure_id,linear_sequence,parent_source_antigen_names",
  limit=20).
  These query the real SAbDab and IEDB REST APIs for the latest data.
  Additionally, call web_search(f"{{target_name}} antibody binding epitope") to
  retrieve recent literature and supplementary context not covered by the
  structured databases. If a search result URL looks useful, call
  visit_webpage(url) to read the full page.
  Print the results so you can reason about known binders and epitopes.
  Pay attention to binding interfaces, key contact residues, and epitope
  structures — you will use this to inform your binder design.

STEP 2 — Design a candidate binder:
  Based on what you learned about the target's structure, epitopes, and known
  interactions, design a NOVEL peptide or mini-protein BINDER (40-120 residues)
  that could interact with the target protein.
  - Design the sequence FROM SCRATCH. Do NOT copy or mutate the target.
  - Consider complementary surface properties for binding.
  - Use helical or loop scaffolds suited for protein-protein interaction.
  - Avoid verbatim motifs found in existing known binders.
  - Use standard amino acid codes.
  - For every design decision, reason about your design choice and justify it.
  Store as a FASTA string variable with header ">candidate_{{target_name}}".
  Print the candidate.

STEP 3 — Verify against SAbDab (remote NCBI BLASTp):
  Call run_remote_blastp_search(candidate_fasta, "SAbDab").
  This runs a live NCBI BLASTp query against the PDB database.
  The result is a JSON STRING — parse it with json.loads().
  Extract max_identity from the parsed dict. Print the parsed result.

STEP 4 — Verify against IEDB (remote NCBI BLASTp):
  Call run_remote_blastp_search(candidate_fasta, "IEDB").
  This runs a live NCBI BLASTp query against the NR database.
  The result is a JSON STRING — parse it with json.loads().
  Extract max_identity from the parsed dict. Print the parsed result.

STEP 5 — Decide, retry if needed, and save trace:
  If EITHER max_identity > {identity_threshold_pct}%, the candidate is
  REJECTED because it is too similar to an existing known sequence.
  In that case, GO BACK TO STEP 2 and design a completely DIFFERENT
  candidate. You may retry up to 3 times total.

  Once you have an accepted candidate (or after 3 failed attempts), build
  a trace dict and call save_reasoning_trace(json.dumps(trace)).
  Then call final_answer(result) with a Python dict:

    result = {{
        "target_seq_fasta": target_fasta_string,
        "candidate_seq_fasta": candidate_fasta_string,
        "decision": "accepted",  # or "rejected"
        "verification": {{
            "identity_threshold": {identity_threshold},
            "sabdab_max_identity": sabdab_max_id,
            "iedb_max_identity": iedb_max_id,
            "rejected": False  # Python booleans, not JSON
        }}
    }}

CRITICAL RULES:
  - DO NOT import tools. All tool functions (search_sabdab, query_iedb,
    run_remote_blastp_search, web_search, visit_webpage, save_reasoning_trace,
    final_answer) are already available in scope. Just call them directly.
    For example, write `result = web_search("query")` NOT
    `from tools import web_search`.
  - The ONLY modules you may import are: json, math, re, time, datetime,
    statistics, collections, itertools, random, unicodedata, queue, stat.
  - ONE step per code block. Do NOT combine steps.
  - Use Python True/False, not JSON true/false.
  - Call final_answer() with a Python dict as the very last step.
  - Tool functions return JSON STRINGS. Always json.loads() the result.
  - NEVER use placeholder/fallback values. Parse the actual data.
  - The candidate must be a novel BINDER, NOT a mutated copy of the target.
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


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

def create_agent(config: AgentConfig) -> CodeAgent:
    """Instantiate a CodeAgent with the protein design tools and prompt."""
    model = _create_model(config.model_id)

    identity_pct = config.identity_threshold * 100
    prompt = SYSTEM_PROMPT.format(
        identity_threshold=config.identity_threshold,
        identity_threshold_pct=identity_pct,
        target_name=config.target_name,
    )

    agent = CodeAgent(
        name="agent",
        tools=[
            run_remote_blastp_search,
            search_sabdab,
            query_iedb,
            web_search,
            visit_webpage,
            save_reasoning_trace,
        ],
        model=model,
        planning_interval=1,
        max_steps=20,  # Allow retries — each cycle is ~4 steps
        additional_authorized_imports=["json"],
        executor_kwargs={"timeout_seconds": 300},  # Remote BLAST can take 30-120s
        prompt_templates={
            "system_prompt": prompt,
            "planning": {
                "initial_plan": "",
                "update_plan_pre_messages": "",
                "update_plan_post_messages": "",
            },
            "managed_agent": {"task": "", "report": ""},
            "final_answer": {"pre_messages": "", "post_messages": ""},
        },
    )
    return agent


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

def run_agent(config: AgentConfig) -> str:
    """Run the full Plan-Execute-Verify pipeline.

    Returns the agent's JSON output as a string.
    """
    validate_fasta(config.target_fasta)
    target_fasta = normalize_fasta(config.target_fasta)
    target_name = config.target_name or "unknown_target"

    agent = create_agent(config)

    task = (
        f"Design a novel peptide or mini-protein BINDER for the following "
        f"target protein. The candidate must be a completely new sequence "
        f"designed to BIND TO the target — NOT a mutated copy of the target.\n\n"
        f"Target name: {target_name}\n"
        f"Identity threshold: {config.identity_threshold:.0%}\n\n"
        f"Target FASTA:\n```\n{target_fasta}```\n\n"
        f"Follow the Plan-Execute-Verify loop. "
        f"Remember: tool functions return JSON strings — use json.loads(). "
        f"Return ONLY the JSON output as described in your instructions."
    )

    print(f"[agent] Starting protein candidate design for '{target_name}'...")
    print(f"[agent] Model: {config.model_id}")
    print(f"[agent] Identity threshold: {config.identity_threshold:.0%}")
    print()

    result = agent.run(task)

    # Save the full reasoning trace (all steps + final output)
    formatted = _format_result(result, target_fasta, config)
    trace_path = _save_agent_log(agent, config, result=formatted)

    # Inject the trace path into the output JSON
    try:
        output = json.loads(formatted)
        output["trace_uri"] = trace_path
        return json.dumps(output, indent=2)
    except (json.JSONDecodeError, TypeError):
        return formatted


def _save_agent_log(agent: CodeAgent, config: AgentConfig, result: Any = None) -> str:
    """Persist the agent's full reasoning trace as JSON in the traces/ folder.

    Captures every step from ``agent.memory.steps`` — including the task,
    planning thoughts, code actions, tool calls, observations, and the
    final answer — so that the complete reasoning chain can be analysed
    after the run.

    Returns the path to the saved trace file.
    """
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    ts_label = timestamp.strftime("%Y%m%d_%H%M%S")
    target_label = (config.target_name or "unknown").replace(" ", "_").lower()
    log_path = TRACES_DIR / f"reasoning_trace_{target_label}_{ts_label}.json"

    trace: dict[str, Any] = {
        "kind": "reasoning_trace",
        "created_at": timestamp.isoformat(),
        "config": {
            "model_id": config.model_id,
            "target_name": config.target_name,
            "identity_threshold": config.identity_threshold,
        },
    }

    # ── Capture every memory step ────────────────────────────────────
    steps_serialized: list[dict[str, Any]] = []
    try:
        for step in agent.memory.steps:
            step_type = type(step).__name__
            try:
                step_dict = step.dict()
            except Exception:
                step_dict = {"raw": str(step)}
            step_dict["_step_type"] = step_type

            # Strip binary image data to keep traces readable
            step_dict.pop("observations_images", None)
            if "model_input_messages" in step_dict:
                step_dict.pop("model_input_messages", None)

            steps_serialized.append(step_dict)
    except Exception as exc:
        trace["step_capture_error"] = str(exc)

    trace["steps"] = steps_serialized
    trace["num_steps"] = len(steps_serialized)

    # ── Include the final agent output ───────────────────────────────
    if result is not None:
        try:
            trace["final_output"] = json.loads(result) if isinstance(result, str) else result
        except (json.JSONDecodeError, TypeError):
            trace["final_output"] = str(result)

    # ── Write ────────────────────────────────────────────────────────
    log_path.write_text(
        json.dumps(trace, indent=2, default=str, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n[agent] Reasoning trace saved to {log_path}")
    return str(log_path)


def _format_result(result: Any, target_fasta: str, config: AgentConfig) -> str:
    """Ensure the result is valid JSON with the required schema."""
    # If the agent returned valid JSON, use it
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if "target_seq_fasta" in parsed and "candidate_seq_fasta" in parsed:
                return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass

    # If the agent returned a dict
    if isinstance(result, dict):
        if "target_seq_fasta" in result and "candidate_seq_fasta" in result:
            return json.dumps(result, indent=2, default=str)

    # Fallback: wrap whatever we got
    payload = {
        "target_seq_fasta": target_fasta,
        "candidate_seq_fasta": str(result) if result else "",
        "decision": "error",
        "verification": {
            "note": "Agent did not return structured output; raw result included.",
        },
        "raw_agent_output": str(result),
    }
    return json.dumps(payload, indent=2, default=str)


def load_fasta_file(path: str) -> str:
    """Load a FASTA record from disk."""
    return Path(path).read_text(encoding="utf-8")
