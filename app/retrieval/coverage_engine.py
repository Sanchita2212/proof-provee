import re
import math
from typing import List, Dict, Any, Set
from collections import Counter
from app.config import settings
from app.retrieval.bm25 import bm25_scores

class ContrastiveCoverageEngine:
    """
    CEC Stage D — Evidence coverage and bounded retrieval
    Instead of standard top-k similarity or reranking:
    1. Evaluates candidate chunks from prioritized documents.
    2. Measures coverage against compiled query facets (subquestions / requirements).
    3. Greedily selects evidence chunks that maximize marginal facet coverage and novel information.
    4. Enforces multi-page / multi-document distribution when queries are comparative or require multiple facts.
    5. Bounds retrieval strictly (stops once requirements are covered or limits reached).
    """

    RELATIVE_RELEVANCE_CUTOFF = 0.55
    BROAD_QUERY_PATTERN = re.compile(
        r"\b(list|all|across|summarize|summary|each|compare|comparison|versus|vs\.?|contrast)\b",
        flags=re.IGNORECASE,
    )

    PAGE_SPANNING_PATTERN = re.compile(
        r"\b(section|sections|page|pages|portfolio|process|framework|steps|stages|lifecycle|flow|timeline|guidance|overview|detail|comprehensive|throughout|breakdown)\b",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "into", "tell",
            "what", "when", "where", "which", "will", "would", "could", "should",
            "about", "their", "there", "then", "been", "have", "were", "they",
            "who", "whose", "whom", "how", "why", "does", "did", "are", "was",
            "all", "any", "each", "every", "some"
        }
        return {w for w in words if w not in stopwords}

    @staticmethod
    def compute_facet_overlap(chunk_text: str, facet_text: str) -> float:
        chunk_tokens = ContrastiveCoverageEngine._tokenize(chunk_text)
        facet_tokens = ContrastiveCoverageEngine._tokenize(facet_text)
        if not facet_tokens or not chunk_tokens:
            return 0.0
        intersection = chunk_tokens.intersection(facet_tokens)
        if not intersection:
            return 0.0

        # Guard against tiny noise / header fragments (< 20 tokens)
        short_penalty = min(1.0, len(chunk_tokens) / 20.0) if len(chunk_tokens) < 20 else 1.0

        # Term coverage (what fraction of query requirements are addressed by this chunk)
        coverage = len(intersection) / len(facet_tokens)

        # Balanced length normalization: avoids penalizing rich 100-300 word pages
        len_norm = 1.0 / (1.0 + 0.15 * math.log(max(1.0, len(chunk_tokens) / 30.0)))

        return (coverage * 2.0 + len(intersection) * 0.5) * short_penalty * len_norm

    @staticmethod
    def _facet_coverage(chunk_text: str, facet_text: str) -> float:
        facet_tokens = ContrastiveCoverageEngine._tokenize(facet_text)
        if not facet_tokens:
            return 0.0
        return len(ContrastiveCoverageEngine._tokenize(chunk_text) & facet_tokens) / len(facet_tokens)

    @staticmethod
    def selection_limit(requirements: Dict[str, Any], requested_limit: int) -> int:
        query = requirements.get("query", "")
        facets = requirements.get("required_facets") or [query]
        facet_count = sum(bool(ContrastiveCoverageEngine._tokenize(facet)) for facet in facets)
        broad_query = bool(
            ContrastiveCoverageEngine.BROAD_QUERY_PATTERN.search(query)
            or requirements.get("is_comparative", False)
            or requirements.get("synthesis", False)
        )
        if broad_query:
            query_limit = max(8, facet_count * 3)
        else:
            query_limit = max(3, facet_count * 2)
        return min(requested_limit, settings.MAX_EVIDENCE_CHUNKS, query_limit)

    @staticmethod
    def select_evidence(
        candidate_chunks: List[Dict[str, Any]],
        requirements: Dict[str, Any],
        max_chunks: int = 8,
    ) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []

        facets = requirements.get("required_facets") or [requirements.get("query", "")]
        req_periods = [p.lower() for p in requirements.get("periods", [])]
        primary_query = requirements.get("query", "")
        max_chunks = ContrastiveCoverageEngine.selection_limit(requirements, max_chunks)

        # Detect page-spanning intent: questions asking for processes, sections, page ranges, or broad synthesis
        is_page_spanning = bool(
            ContrastiveCoverageEngine.PAGE_SPANNING_PATTERN.search(primary_query)
            or len(facets) >= 4
        )

        selected_chunks = []
        covered_facets = set()
        seen_pages = set()
        selected_chunk_texts = []

        # The compiler stores the original query at position zero as a routing
        # signal, followed by answerable subquestions.  A long query rarely
        # appears verbatim in one chunk, so it must not prevent completion of
        # coverage accounting.  Hand-authored requirements that do not include
        # the query retain all their facets.
        normalized_query = " ".join(primary_query.lower().split())
        normalized_first_facet = " ".join(str(facets[0]).lower().split()) if facets else ""
        has_routing_facet = len(facets) > 1 and normalized_first_facet == normalized_query
        answerable_facet_indices = list(range(1, len(facets))) if has_routing_facet else list(range(len(facets)))
        if not answerable_facet_indices:
            answerable_facet_indices = [0]
        # A focused lookup needs only its best supporting excerpt once its sole
        # facet is covered.  Requiring a second chunk lowers context precision
        # by pulling in neighbouring but non-answering text.
        min_chunks = min(
            max_chunks,
            1 if len(answerable_facet_indices) == 1
            else (2 if len(answerable_facet_indices) == 2 else len(answerable_facet_indices)),
        )
        per_facet = 2 if requirements.get("synthesis", False) else 1

        clean_q = re.sub(r"[^a-zA-Z0-9\s]", " ", primary_query.lower())
        q_words = [w for w in clean_q.split() if len(w) > 2]
        bigrams = [f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words) - 1)]
        trigrams = [f"{q_words[i]} {q_words[i+1]} {q_words[i+2]}" for i in range(len(q_words) - 2)]
        sparse_scores = bm25_scores(
            " ".join([primary_query, *requirements.get("keywords", [])]),
            [chunk.get("text", "") for chunk in candidate_chunks],
        )
        max_sparse_score = max(sparse_scores, default=0.0)

        # Score every chunk against every facet, primary query, and phrase overlap
        scored_candidates = []
        for chunk_index, chunk in enumerate(candidate_chunks):
            text = chunk.get("text", "")
            doc = chunk.get("document", "")
            page = chunk.get("page", 0)
            lower_text = text.lower()

            # Base relevance across all facets
            facet_scores = [ContrastiveCoverageEngine.compute_facet_overlap(text, f) for f in facets]
            max_facet_score = max(facet_scores) if facet_scores else 0.0
            avg_facet_score = sum(facet_scores) / len(facet_scores) if facet_scores else 0.0

            # Direct primary query score
            primary_query_score = ContrastiveCoverageEngine.compute_facet_overlap(text, primary_query) if primary_query else max_facet_score

            # N-gram exact phrase alignment (rewards exact co-occurrence of adjacent query terms)
            phrase_bonus = 0.0
            for bg in bigrams:
                if bg in lower_text:
                    phrase_bonus += 0.35
            for tg in trigrams:
                if tg in lower_text:
                    phrase_bonus += 0.70

            # Boost for temporal/period alignment
            period_bonus = 0.0
            for p in req_periods:
                clean_p = p.replace("fy", "").strip()
                if clean_p in lower_text:
                    period_bonus += 0.15

            total_relevance = (
                primary_query_score * 0.45
                + max_facet_score * 0.35
                + avg_facet_score * 0.20
                + phrase_bonus
                + period_bonus
            )
            if max_sparse_score:
                total_relevance += (
                    max(total_relevance, 1.0)
                    * 0.35
                    * sparse_scores[chunk_index]
                    / max_sparse_score
                )
            scored_candidates.append({
                "chunk": chunk,
                "relevance": total_relevance,
                "facet_scores": facet_scores,
                "facet_coverages": [
                    ContrastiveCoverageEngine._facet_coverage(text, facet)
                    for facet in facets
                ],
            })

        # Sort initially by raw relevance
        scored_candidates.sort(key=lambda x: x["relevance"], reverse=True)
        top_relevance = scored_candidates[0]["relevance"]
        if top_relevance <= 0:
            return []
        relevance_cutoff = top_relevance * ContrastiveCoverageEngine.RELATIVE_RELEVANCE_CUTOFF

        eligible_candidates = [
            item for item in scored_candidates
            if item["relevance"] >= relevance_cutoff
        ]

        def redundancy_for(item: Dict[str, Any]) -> float:
            chunk_tokens = ContrastiveCoverageEngine._tokenize(item["chunk"].get("text", ""))
            redundancy = 0.0
            for prev_text in selected_chunk_texts:
                prev_tokens = ContrastiveCoverageEngine._tokenize(prev_text)
                if prev_tokens and chunk_tokens:
                    redundancy = max(redundancy, len(chunk_tokens & prev_tokens) / len(chunk_tokens | prev_tokens))
            return redundancy

        def can_select(item: Dict[str, Any]) -> bool:
            chunk = item["chunk"]
            if chunk in selected_chunks:
                return False
            return not (
                redundancy_for(item) > 0.65
                and (chunk.get("document"), chunk.get("page")) in seen_pages
            )

        def add(item: Dict[str, Any]) -> None:
            chunk = item["chunk"]
            selected_chunks.append(chunk)
            selected_chunk_texts.append(chunk.get("text", ""))
            seen_pages.add((chunk.get("document"), chunk.get("page")))
            covered_facets.update(
                idx for idx in answerable_facet_indices
                if item["facet_coverages"][idx] >= settings.MIN_COVERAGE_THRESHOLD
            )

        # Phase 1: reserve the best eligible evidence for every answerable
        # subquestion.  This prevents the full query from monopolising the
        # selection and gives synthesis prompts up to two sources per facet.
        for facet_idx in answerable_facet_indices:
            ranked = sorted(
                eligible_candidates,
                key=lambda item: (item["facet_scores"][facet_idx], item["relevance"]),
                reverse=True,
            )
            taken = 0
            for item in ranked:
                if taken >= per_facet or len(selected_chunks) >= max_chunks:
                    break
                if not can_select(item):
                    continue
                add(item)
                taken += 1

        # Phase 2: if the required facets are not genuinely covered, use the
        # remaining budget by relevance.  Lack of another coverage gain is not
        # a stopping condition; the relative cutoff still excludes weak hits.
        all_answerable_facets_covered = set(answerable_facet_indices).issubset(covered_facets)
        # If no answerable facet reaches the calibrated threshold, retain a
        # small diagnostic fallback rather than filling the prompt with weak
        # topical matches. Normal queries with at least one covered facet may
        # use their full bounded budget.
        weak_evidence_limit = min(max_chunks, 2)
        fill_limit = max_chunks if covered_facets else weak_evidence_limit
        if not (all_answerable_facets_covered and len(selected_chunks) >= min_chunks):
            for item in eligible_candidates:
                if len(selected_chunks) >= fill_limit:
                    break
                if can_select(item):
                    add(item)

        # Phase 2: Page-spanning & section expansion (bounded step inside coverage logic)
        # If the question is page-spanning / multi-point and budget remains,
        # once the top chunk lands in a section, take neighbouring pages/sections that score above the cutoff.
        if is_page_spanning and len(selected_chunks) < max_chunks:
            selected_sections = {
                c.get("section_title") for c in selected_chunks if c.get("section_title")
            }
            selected_doc_pages = {
                (c.get("document"), c.get("page")) for c in selected_chunks
            }

            while len(selected_chunks) < max_chunks:
                best_expansion = None
                best_expansion_key = None

                for item in eligible_candidates:
                    chunk = item["chunk"]
                    if chunk in selected_chunks:
                        continue

                    doc_page_key = (chunk.get("document"), chunk.get("page"))
                    sec = chunk.get("section_title")

                    # Check if neighbouring page (+/- 1) or in same section
                    is_neighbour_page = any(
                        chunk.get("document") == s_doc
                        and abs(chunk.get("page", -999) - s_page) == 1
                        for (s_doc, s_page) in selected_doc_pages
                    )
                    is_same_section = bool(sec and sec in selected_sections)

                    if not (is_neighbour_page or is_same_section):
                        continue

                    text = chunk.get("text", "")
                    chunk_tokens = ContrastiveCoverageEngine._tokenize(text)
                    redundancy = 0.0
                    for prev_text in selected_chunk_texts:
                        prev_tokens = ContrastiveCoverageEngine._tokenize(prev_text)
                        if prev_tokens and chunk_tokens:
                            jaccard = len(chunk_tokens & prev_tokens) / len(chunk_tokens | prev_tokens)
                            redundancy = max(redundancy, jaccard)
                    if redundancy > 0.65 and doc_page_key in seen_pages:
                        continue

                    exp_key = (1 if is_same_section else 0, item["relevance"], -redundancy)
                    if best_expansion_key is None or exp_key > best_expansion_key:
                        best_expansion = item
                        best_expansion_key = exp_key

                if best_expansion is None:
                    break

                exp_chunk = best_expansion["chunk"]
                selected_chunks.append(exp_chunk)
                selected_chunk_texts.append(exp_chunk.get("text", ""))
                doc_page_key = (exp_chunk.get("document"), exp_chunk.get("page"))
                seen_pages.add(doc_page_key)
                selected_doc_pages.add(doc_page_key)
                if exp_chunk.get("section_title"):
                    selected_sections.add(exp_chunk.get("section_title"))

        return selected_chunks
