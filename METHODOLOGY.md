# Methodology Appendix

This document describes the automated literature mining and ranking pipeline used in Stage 1 of the LLM-Guided Robust Optimization for Epilepsy Neurostimulation project. It is intended to make the process auditable and reproducible.

---

## 1. PubMed Queries

Ten queries were issued against the PubMed Entrez API via Biopython. Each query retrieved up to 20 results (title + abstract + metadata). Queries were designed to cover the intersection of computational brain modeling, epilepsy treatment, and stimulation optimization.

```
1.  "virtual brain epilepsy stimulation"
2.  "epilepsy digital twin intervention"
3.  "virtual epileptic patient seizure"
4.  "TVB seizure optimization"
5.  "epilepsy AND whole-brain model AND treatment"
6.  "seizure onset zone AND computational AND stimulation"
7.  "epilepsy AND network model AND brain stimulation"
8.  "responsive neurostimulation AND computational"
9.  "epilepsy AND in silico AND stimulation"
10. "virtual brain twins stimulation epilepsy"
```

**Script:** `fetch_all_papers.py`

---

## 2. Deduplication

Deduplication was performed using exact PMID matching across all query results. Papers retrieved by multiple queries were counted once. The final deduplicated corpus contained **136 papers**.

No title normalization or fuzzy matching was applied — only exact PMID equality. This means near-duplicate papers with different PMIDs (e.g., erratum notices) are counted separately.

---

## 3. Gap Extraction Prompt (Claude Haiku)

Each paper's title and abstract were passed to `claude-haiku-4-5-20251001` with the following system prompt:

```
You are analyzing a neuroscience paper about epilepsy and brain stimulation.
Extract structured research gaps from the following title and abstract.

Return a JSON object with:
- open_questions: list of 1-3 specific unanswered scientific questions
- future_experiments: list of 1-3 concrete experiments suggested or implied
- limitations: list of 1-3 methodological limitations
- tvb_relevant: boolean, whether this paper is relevant to whole-brain
  computational modeling

Focus on non-trivial gaps that are: (1) not already addressed in the paper,
(2) relevant to clinical translation or computational modeling limitations.
```

**User message format:**

```
Title: <paper title>

Abstract: <paper abstract>
```

Each call used `max_tokens=1024`. The model was instructed to return valid JSON only (no markdown fences). JSON parse errors were logged and assigned empty fields.

**Script:** `extract_gaps.py`  
**Total extraction tasks:** 1,080 (across 136 papers, ~7.9 gaps per paper on average)  
**Output:** `gaps.json`

---

## 4. Clustering Method

Extracted gaps (`open_questions` + `future_experiments`) were aggregated across all papers into a flat list of 1,080 items, then grouped into thematic clusters using a two-stage LLM process:

**Stage 4a — Batch grouping (Claude Haiku)**  
Ideas were split into batches of 150. Each batch was sent to `claude-haiku-4-5` with a system prompt instructing it to group ideas into 25–40 thematic clusters and return a JSON array. The model was asked to merge near-duplicates and prefer clusters with direct computational modeling or brain stimulation angles.

**Stage 4b — Consolidation (Claude Haiku)**  
When multiple batches were processed, a second Haiku call consolidated the raw clusters across batches into a final set of ~30 non-overlapping themes.

**Note:** this is LLM-based semantic grouping, not embedding-based clustering. Results are not deterministic — re-running the pipeline may produce different cluster boundaries and slightly different final rankings.

**Script:** `rank_ideas.py` (`group_ideas()`, `consolidate_clusters()`)  
**Final cluster count:** ~30

---

## 5. Scoring Rubric

Each cluster was scored by `claude-opus-4-5` on three axes (1–10 each):

| Axis | Definition | Low (1–3) | High (8–10) |
|---|---|---|---|
| **Novelty** | How unlikely is this gap to be explicitly addressed in existing literature? | Well-studied, incremental, active research area | Genuinely unexplored territory, no direct precedent found |
| **TVB Feasibility** | Can this question be directly tested within The Virtual Brain simulation framework? | Requires fundamentally different tools or data not available in TVB | Directly implementable with current or near-term TVB capabilities |
| **Clinical Impact** | How relevant is solving this to real-world epilepsy treatment challenges? | Purely theoretical, distant from patient care | Directly translatable; could inform treatment decisions within 5 years |

**Final score = Novelty + TVB Feasibility + Clinical Impact** (max 30, equal weights)

The scoring prompt provided detailed anchor descriptions for scores 1–3, 4–6, and 7–10 on each axis, along with instructions to return a structured JSON object including scores, rationale, and a "key opportunity" sentence per cluster.

Clusters were then ranked by total score descending. The top 20 were saved to `ranked_ideas.json` and `ranked_ideas.md`.

**Top result:** Score 25/30 for the cluster *"Digital Twins & Multimodal AI for Precision Neurology"* — and the gap driving Stage 3 (robust RL transfer across patients) also scored 25/30 under this rubric.

---

## 6. Limitations

The following limitations apply to the entire pipeline and should be considered before treating any output as ground truth:

- **No human expert validation.** No domain expert reviewed the extracted gaps, cluster assignments, or final rankings. All outputs are LLM-generated.
- **Hallucination risk.** Claude Haiku may extract gaps not actually present in the abstract, or mischaracterize a paper's limitations.
- **Non-deterministic clustering.** LLM-based grouping is sensitive to batch ordering and temperature. Re-running will produce different clusters.
- **Single judge, no inter-rater reliability.** All scoring was performed by a single Claude Opus instance. No second judge or human reviewer was used to check consistency.
- **PubMed coverage gaps.** The corpus excludes conference papers, preprints (bioRxiv/medRxiv), book chapters, and non-English publications that may contain relevant work.
- **Query design bias.** The 10 queries were hand-designed to favour TVB-relevant computational work. Other research directions (e.g., purely clinical or purely genetic) are underrepresented.
- **Score interpretation.** A score of 25/30 means one LLM judge assessed the cluster as highly novel, TVB-feasible, and clinically impactful. It does not constitute peer review or independent verification.

Ranked ideas should be treated as **structured hypotheses** for further human scrutiny, not as validated research priorities.
