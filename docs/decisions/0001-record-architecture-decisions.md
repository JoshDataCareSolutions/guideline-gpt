# 1. Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-01-15

## Context

guideline-gpt is built as an engineering showcase: the point is not only that
it works, but that a reader can understand *why* it is built the way it is. RAG
systems involve many defensible-but-consequential choices (which retrievers, how
to fuse them, where to rerank, which vector store) and those choices are easy to
forget once the code is written. Code comments capture the *what*; they rarely
capture the alternatives that were rejected.

## Decision

We keep a log of Architecture Decision Records in `docs/decisions/`, one Markdown
file per decision, numbered sequentially (`NNNN-title.md`). Each record states
the context, the decision, and its consequences. Records are immutable once
accepted: a change of direction is a new ADR that supersedes the old one, rather
than an edit.

We use the lightweight format popularized by Michael Nygard. Source-code comments
may reference a record by number (e.g. "see ADR-002") rather than repeating the
rationale inline.

## Consequences

- The reasoning behind non-obvious choices is discoverable in one place.
- Reviewers can challenge a decision by reading its ADR instead of reverse-
  engineering it from the diff.
- There is a small upkeep cost: a genuinely architectural change must be
  accompanied by a new record.
