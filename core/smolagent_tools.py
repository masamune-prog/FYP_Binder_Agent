"""Smolagents @tool-decorated functions for the protein candidate agent.

Each function here is a standalone tool that the CodeAgent can invoke.
They wrap local MMseqs2 searches, database lookups, and trace persistence.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import requests
from smolagents import tool, DuckDuckGoSearchTool, VisitWebpageTool

# Lazily instantiated so the import itself doesn't make a network call
_ddg_tool: "DuckDuckGoSearchTool | None" = None
_visit_tool: "VisitWebpageTool | None" = None


def _get_ddg() -> "DuckDuckGoSearchTool":
    global _ddg_tool
    if _ddg_tool is None:
        _ddg_tool = DuckDuckGoSearchTool(max_results=10)
    return _ddg_tool


def _get_visit() -> "VisitWebpageTool":
    global _visit_tool
    if _visit_tool is None:
        _visit_tool = VisitWebpageTool()
    return _visit_tool


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return a summary of the top results.

    Use this to look up recent scientific literature, clinical data, protein
    function descriptions, or any other information that may not be in the
    specialised databases (SAbDab / IEDB).

    Args:
        query: A natural-language or keyword search query, e.g.
            'EGFR antibody binding epitope site', 'PD-L1 crystal structure overview'.
    """
    return _get_ddg()(query)


@tool
def visit_webpage(url: str) -> str:
    """Fetch and return the text content of a web page.

    Use this after web_search to read the full content of a relevant page
    (e.g. a PubMed abstract, UniProt entry, or PDB summary page).

    Args:
        url: The full URL of the page to visit, e.g.
            'https://www.uniprot.org/uniprot/P00533'.
    """
    return _get_visit()(url)


import io
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASES_DIR = ROOT_DIR / "databases"
TRACES_DIR = ROOT_DIR / "traces"

SABDAB_DB = DATABASES_DIR / "sabdab_db" / "sabdab"
IEDB_DB = DATABASES_DIR / "iedb_db" / "iedb"


@tool
def run_remote_blastp_search(query_fasta: str, database: str) -> str:
    """Run a remote NCBI BLASTp sequence alignment search against a prior-art database.

    Use this to check whether a candidate protein sequence is too similar to
    known sequences. A sequence with greater than 80 percent identity to a
    known binder should be rejected.

    Args:
        query_fasta: The query protein sequence in FASTA format, including a
            header line starting with '>'.
        database: Which conceptual database to search. Must be one of 'SAbDab' or 'IEDB'.
            (Will map to NCBI 'pdb' or 'nr' respectively).
    """
    from Bio.Blast import NCBIWWW, NCBIXML

    # Map our conceptual databases to NCBI databases
    db_map = {
        "SAbDab": "pdb",
        "sabdab": "pdb",
        "IEDB": "nr",
        "iedb": "nr",
    }
    ncbi_db = db_map.get(database)
    if not ncbi_db:
        return json.dumps({"error": f"Unknown database '{database}'. Use 'SAbDab' or 'IEDB'."})

    # Extract raw sequence from FASTA
    lines = query_fasta.strip().splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    if not seq:
        return json.dumps({"error": "Empty sequence provided."})

    try:
        # Run BLASTp remotely
        result_handle = NCBIWWW.qblast("blastp", ncbi_db, seq, hitlist_size=10)
        blast_records = NCBIXML.parse(result_handle)

        hits = []
        max_identity = 0.0

        for record in blast_records:
            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    ident = (hsp.identities / hsp.align_length) * 100 if hsp.align_length > 0 else 0.0
                    hit = {
                        "target": alignment.title[:100],  # Truncate long titles
                        "percent_identity": round(ident, 2),
                        "alignment_length": hsp.align_length,
                        "evalue": hsp.expect,
                    }
                    hits.append(hit)
                    max_identity = max(max_identity, ident)

        # Sort hits by identity
        hits.sort(key=lambda h: h["percent_identity"], reverse=True)

        return json.dumps({
            "database": database,
            "ncbi_db_used": ncbi_db,
            "max_identity": round(max_identity, 2),
            "num_hits": len(hits),
            "top_hits": hits[:10],
        }, indent=2)

    except Exception as exc:
        return json.dumps({
            "error": f"Remote BLASTp search failed: {exc}",
            "database": database,
            "max_identity": 0.0,
            "num_hits": 0,
            "top_hits": [],
        })


@tool
def search_sabdab(target_name: str) -> list:
    """Search the SAbDab structural antibody database for known binders to a target antigen.

    Returns information about known antibody structures that bind the given
    target. Use this during the planning phase to understand what antibodies
    already exist for a target before designing a novel candidate.

    Args:
        target_name: Name of the target protein or antigen, e.g. 'EGFR', 'HER2', 'PD-L1'.
    """
    url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"

    params = {
        "antigen_name": target_name,
        "format": "tsv",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    if not response.text.strip():
        return []

    try:
        df = pd.read_csv(io.StringIO(response.text), sep="\t")
    except Exception as e:
        raise ValueError(f"Failed to parse SAbDab response: {e}\nRaw response:\n{response.text[:500]}")

    # Sort by resolution ascending so best-quality structures come first
    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    df = df.sort_values("resolution", ascending=True, na_position="last")

    results = []
    for _, row in df.head(10).iterrows():
        results.append({
            "pdb_id":        row.get("pdb"),
            "antigen_name":  row.get("antigen_name"),
            "antigen_type":  row.get("antigen_type"),
            "heavy_chain":   row.get("Hchain"),
            "light_chain":   row.get("Lchain"),
            "antigen_chain": row.get("antigen_chain"),
            "resolution":    row.get("resolution"),
            "species":       row.get("heavy_species"),
            "scfv":          row.get("scfv"),
        })

    return results


@tool
def query_iedb(endpoint: str, search_field: str, search_value: str, select_fields: str, limit: int) -> str:
    """Query the IEDB (Immune Epitope Database) API for immunological data.

    Args:
        endpoint: IEDB table to query. One of: 'epitope_search', 'tcell_search', 'bcell_search', 'mhc_search', 'antigen_search', 'receptor_search'.
        search_field: Field to filter on, e.g. 'parent_source_antigen_names::text', 'linear_sequence', 'source_organism_names::text', 'host_organism_names::text'. Use '::text' suffix to cast array fields for partial matching.
        search_value: PostgREST filter expression for the field, e.g. 'ilike.*EGFR*', 'eq.SIINFEKL', 'ilike.*human*'.
        select_fields: Comma-separated fields to return, e.g. 'structure_id,linear_sequence,parent_source_antigen_names'. Pass empty string for all fields.
        limit: Maximum number of deduplicated results to return. Use 20-50 for a summary, up to 200 for exhaustive search.
    """
    import requests as _requests

    VALID_ENDPOINTS = {
        "epitope_search",
        "tcell_search",
        "bcell_search",
        "mhc_search",
        "antigen_search",
        "receptor_search",
    }

    if endpoint not in VALID_ENDPOINTS:
        return json.dumps({
            "error": f"Invalid endpoint '{endpoint}'. Choose from: {sorted(VALID_ENDPOINTS)}"
        }, indent=2)

    params = {"limit": min(max(limit, 1), 500)}
    if search_field and search_value:
        params[search_field] = search_value
    if select_fields:
        params["select"] = select_fields

    try:
        resp = _requests.get(
            f"https://query-api.iedb.org/{endpoint}",
            params=params,
            timeout=30,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        # Deduplicate on linear_sequence if present, otherwise on full row
        seen = set()
        unique = []
        for entry in data:
            key = entry.get("linear_sequence") or json.dumps(entry, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique.append(entry)

        return json.dumps({
            "endpoint": endpoint,
            "filters": {search_field: search_value} if search_field else {},
            "total_returned": len(data),
            "unique_count": len(unique),
            "results": unique[:limit],
        }, indent=2, default=str)

    except _requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("message", "")
        except Exception:
            pass
        return json.dumps({"error": f"HTTP {exc.response.status_code}: {detail or str(exc)}"}, indent=2)
    except _requests.RequestException as exc:
        return json.dumps({"error": f"Network error: {exc}"}, indent=2)
    except ValueError:
        return json.dumps({"error": "IEDB did not return valid JSON."}, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Unexpected error: {exc}"}, indent=2)

@tool
def save_reasoning_trace(trace_json: str) -> str:
    """Persist a reasoning trace as a JSON file for later analysis.

    Call this at the end of each Plan-Execute-Verify cycle to record your
    decisions, tool outputs, and verification results.

    Args:
        trace_json: A JSON string containing the reasoning trace. Must include
            at minimum: plan, actions taken, verification outcome, and decision.
    """
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    filename = f"trace_{timestamp.strftime('%Y%m%d_%H%M%S')}_{int(timestamp.timestamp())}.json"
    trace_path = TRACES_DIR / filename

    try:
        payload = json.loads(trace_json)
    except json.JSONDecodeError:
        payload = {"raw": trace_json}

    payload["saved_at"] = timestamp.isoformat()
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return f"Trace saved to {trace_path}"
