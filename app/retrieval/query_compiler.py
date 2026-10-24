import re
import json
import logging
from typing import Dict, Any, List, Optional
from app.generation.provider import LLMProvider

logger = logging.getLogger(__name__)

class QueryCompiler:
    """
    CEC Stage C — Query requirement compilation
    Deconstructs user queries into explicit operational requirements:
    - Primary entities
    - Reporting periods / temporal constraints
    - Target domain / document scope
    - Metrics & definitions requested
    - Comparison targets (if multi-entity or multi-period comparison)
    - Discrete factual requirements (subquestions)
    """

    BROAD_QUERY_PATTERN = re.compile(
        r"\b(list|all|across|summarize|summary|each|compare|comparison|versus|vs\.?|contrast)\b",
        flags=re.IGNORECASE,
    )

    COMPLEX_SPLIT_PATTERN = re.compile(
        r"\s*(?:;|\bthen\s+(?:contrast|compare)\b|[-–—]\s*and\s+|\band\s+(?=(?:what|which|how|who|why|where)\b))\s*",
        flags=re.IGNORECASE,
    )

    @classmethod
    def _deterministic_subquestions(cls, query: str) -> List[str]:
        """Extract only explicit question clauses when planning is unavailable.

        This is facet decomposition, not query expansion: every returned clause
        is copied from the user's wording and remains traceable to the query.
        """
        clauses = [" ".join(part.split()) for part in cls.COMPLEX_SPLIT_PATTERN.split(query)]
        return [
            clause.strip(" ,.-")
            for clause in clauses
            if len(cls._meaningful_tokens(clause)) >= 3
        ][:4]

    @staticmethod
    def _meaningful_tokens(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9]{3,}\b", text)

    @staticmethod
    def compile_deterministic(query: str) -> Dict[str, Any]:
        """Fast, robust regex-based extraction of temporal and comparative features."""
        # Detect periods / years
        years = re.findall(r"\b(?:FY\s*[-–]?\s*\d{2,4}(?:[-–/]\d{2,4})?|\b20\d{2}(?:[-–/]\d{2,4})?|\bQ[1-4]\s*20\d{2})\b", query, flags=re.IGNORECASE)
        clean_years = list(dict.fromkeys([y.strip().upper() for y in years]))

        # Detect comparison intent
        lower = query.lower()
        is_comparative = any(term in lower for term in ["compare", "comparison", "difference between", "versus", "vs.", "higher than", "lower than", "increased from", "decreased from"])

        # Detect entities via capitalization heuristics
        action_verbs = {"what", "which", "where", "when", "how", "who", "why", "pull", "explain", "describe", "table", "report", "state", "total", "compare", "analyze", "list", "summarize", "find", "check"}
        raw_caps = re.findall(r"\b[A-Z][a-zA-Z0-9&.\-]{2,}(?:\s+[A-Z][a-zA-Z0-9&.\-]{2,})*\b", query)
        filtered_caps = []
        for phrase in raw_caps:
            parts = phrase.split()
            # Strip leading action verbs if captured
            while parts and parts[0].lower() in action_verbs:
                parts = parts[1:]
            if parts:
                clean_phrase = " ".join(parts)
                if clean_phrase.lower() not in action_verbs and len(clean_phrase) > 2:
                    filtered_caps.append(clean_phrase)

        # Key topic words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        stopwords = {"what", "which", "where", "when", "how", "who", "why", "the", "and", "for", "with", "this", "that", "from", "into", "tell", "every", "all"}
        keywords = [w for w in words if w not in stopwords]

        # Generic document type mentions
        doc_type_hints = [w for w in ["report", "filing", "framework", "standard", "guidance", "handbook", "statement", "memo", "paper"] if w in lower]

        subquestions = QueryCompiler._deterministic_subquestions(query)
        # A single unsplit clause is merely the original query, not a facet.
        if len(subquestions) < 2:
            subquestions = []

        return {
            "query": query,
            "entities": filtered_caps,
            "periods": clean_years,
            "is_comparative": is_comparative,
            "document_types": doc_type_hints,
            "keywords": keywords[:15],
            "required_facets": [query] + [sq for sq in subquestions if sq != query],
            "synthesis": bool(
                QueryCompiler.BROAD_QUERY_PATTERN.search(query)
                or len(subquestions) >= 3
            ),
        }

    @staticmethod
    async def compile_query(query: str, llm_provider: Optional[LLMProvider] = None) -> Dict[str, Any]:
        base_reqs = QueryCompiler.compile_deterministic(query)

        # If LLM provider is available and configured, enrich requirements with subquestions & comparison targets
        if llm_provider and llm_provider.api_key:
            try:
                system_prompt = (
                    "Extract compact retrieval requirements. Return only valid JSON."
                )
                user_prompt = (
                    f"Query: {query}\n"
                    'Return {"entities":[],"periods":[],"is_comparative":false,'
                    '"comparison_targets":[],"subquestions":[]} with at most 3 entities, '
                    "2 periods, 2 comparison targets, and up to 4 short subquestions. "
                    "Do not repeat the query or add explanations."
                )
                raw_json = await llm_provider.generate_completion(system_prompt, user_prompt, json_mode=True, max_tokens=400)
                parsed = json.loads(raw_json)
                
                # Merge enriched fields safely
                base_reqs["entities"] = list(dict.fromkeys(base_reqs["entities"] + parsed.get("entities", [])))
                base_reqs["periods"] = list(dict.fromkeys(base_reqs["periods"] + parsed.get("periods", [])))
                base_reqs["is_comparative"] = parsed.get("is_comparative", base_reqs["is_comparative"])
                base_reqs["comparison_targets"] = parsed.get("comparison_targets", [])
                subquestions = parsed.get("subquestions", [])
                base_reqs["required_facets"] = list(dict.fromkeys(
                    [query]
                    + base_reqs["required_facets"][1:]
                    + [sq for sq in subquestions if sq != query]
                ))
                base_reqs["synthesis"] = bool(
                    QueryCompiler.BROAD_QUERY_PATTERN.search(query)
                    or len(base_reqs["required_facets"]) - 1 >= 3
                )
            except Exception as e:
                logger.warning(f"LLM query compilation fallback to deterministic: {e}")

        return base_reqs
