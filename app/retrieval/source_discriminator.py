import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.retrieval.bm25 import bm25_scores

logger = logging.getLogger(__name__)

class SourceDiscriminator:
    """
    CEC Stage B — Contrastive source discrimination
    When multiple documents exist in a collection, discriminates among candidate documents
    using contrastive features:
    - Entity & organization matching
    - Reporting period alignment (FY25 vs FY26)
    - Document type relevance (Integrated Annual Report vs Regulatory Standard)
    - Jurisdictional scope
    - Lexical contrast against candidate document titles and extracted headers
    """

    @staticmethod
    def rank_candidate_documents(
        documents: List[Dict[str, Any]],
        requirements: Dict[str, Any],
        max_candidates: int = 5,
        document_texts: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        query_text = requirements.get("query", "").lower()
        req_entities = [e.lower() for e in requirements.get("entities", [])]
        req_periods = [p.lower() for p in requirements.get("periods", [])]
        req_keywords = [k.lower() for k in requirements.get("keywords", [])]
        document_texts = document_texts or {}
        lexical_scores = bm25_scores(
            " ".join([query_text, *req_keywords]),
            [document_texts.get(doc.get("filename", ""), "") for doc in documents],
        )
        max_lexical_score = max(lexical_scores, default=0.0)

        scored_docs = []

        for index, doc in enumerate(documents):
            filename = doc.get("filename", "").lower()
            doc_type = (doc.get("document_type") or "").lower()
            doc_scope = (doc.get("scope") or "").lower()
            doc_entities = [e.lower() for e in doc.get("entities", [])]
            doc_periods = [p.lower() for p in doc.get("periods", [])]

            score = 0.0
            reasons = []

            # 1. Entity Contrastive Matching
            # Direct match in filename or extracted entities (require word-level or meaningful token length)
            matched_entity = False
            for ent in req_entities:
                if len(ent) < 4:
                    # For short acronyms (like FED, OCC), match on word boundaries only
                    pattern = rf"\b{re.escape(ent)}\b"
                    if re.search(pattern, filename) or any(re.search(pattern, de) for de in doc_entities):
                        score += 15.0
                        matched_entity = True
                        reasons.append(f"entity_match({ent})")
                else:
                    if ent in filename or any(ent in de for de in doc_entities):
                        score += 15.0
                        matched_entity = True
                        reasons.append(f"entity_match({ent})")

            # 2. Period / Temporal Alignment
            # Critical for distinguishing multi-year reports (e.g. FY 2025 vs FY 2026)
            for per in req_periods:
                clean_per = per.replace("fy", "").strip()
                if clean_per in filename or any(clean_per in dp for dp in doc_periods):
                    score += 10.0
                    reasons.append(f"period_match({per})")

            # 3. Document Type & Structural Alignment
            # Match discovered query document-type intent with document's discovered metadata or filename
            req_doc_types = [t.lower() for t in requirements.get("document_types", [])]
            for req_dt in req_doc_types:
                if req_dt in doc_type or req_dt in filename:
                    score += 6.0
                    reasons.append(f"doc_type_match({req_dt})")

            if max_lexical_score:
                lexical_boost = 12.0 * lexical_scores[index] / max_lexical_score
                score += lexical_boost
                if lexical_boost:
                    reasons.append(f"bm25_content({lexical_boost:.2f})")

            # 4. Lexical Keyword Overlap with filename
            overlap_words = [kw for kw in req_keywords if kw in filename]
            if overlap_words:
                score += len(overlap_words) * 3.0
                reasons.append(f"filename_keywords({','.join(overlap_words)})")

            # Soft baseline: never completely zero out documents unless collection has strong positive entity matches
            if score == 0.0:
                score = 0.5
                reasons.append("unmatched_candidate_baseline")

            scored_docs.append((doc, score, "; ".join(reasons)))

        # Sort descending by contrastive score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:max_candidates]
