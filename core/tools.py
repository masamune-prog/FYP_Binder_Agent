"""Tool interfaces for the protein candidate agent.

This module defines the *contract* that bioinformatics tools must satisfy.
The concrete implementations live in ``smolagent_tools.py`` as ``@tool``
decorated functions for direct use with smolagents ``CodeAgent``.

The Protocol below is kept for documentation and for any non-agent code paths
(tests, batch pipelines) that want a typed tool surface.
"""

from __future__ import annotations

from typing import Any, Protocol


class BioinformaticsTools(Protocol):
    """Expected tool surface for the candidate protein agent.

    Each method corresponds to a ``@tool`` function in ``smolagent_tools.py``.
    """

    def blastp(self, query_fasta: str, database: str) -> dict[str, Any]:
        """Run local MMseqs2 search against a named database.

        Returns a dict with at least: max_identity, num_hits, top_hits.
        """

    def lookup_sabdab(self, target_name: str | None = None) -> dict[str, Any]:
        """Retrieve known antibody binders from SAbDab for a target."""

    def lookup_iedb(self, target_name: str | None = None) -> dict[str, Any]:
        """Retrieve known epitopes from IEDB for a target."""

    def save_trace(self, payload: dict[str, Any]) -> str:
        """Persist compact reasoning summaries and return their URI/path."""
