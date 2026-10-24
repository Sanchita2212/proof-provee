import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.schemas import ChunkCitation
from app.generation.provider import LLMProvider

logger = logging.getLogger(__name__)
NO_INFORMATION_RESPONSE = "No information was retrieved for this question."
REFUSAL_RE = re.compile(
    r"\bNo information was retrieved for this(?: specific)? question\.?",
    flags=re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[[^\]]+\s+p\.\d+", flags=re.IGNORECASE)

class GroundedResult(tuple):
    """Backwards-compatible tuple that unpacks as (answer, citations) or (answer, citations, usage)."""
    def __new__(cls, answer: str, citations: List[ChunkCitation], usage: Optional[Dict[str, Any]] = None):
        return super().__new__(cls, (answer, citations, usage))

    def __iter__(self):
        # Allow default 2-value unpacking for legacy tests: answer, citations = func()
        yield self[0]
        yield self[1]

    @property
    def answer(self) -> str:
        return self[0]

    @property
    def citations(self) -> List[ChunkCitation]:
        return self[1]

    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        return self[2]

class GroundedAnswerGenerator:
    """
    CEC Stage E — Grounded answer generation
    Synthesizes concise, strictly factual answers from retrieved evidence chunks.
    Requirements:
    - Cite source document and page number for every material claim (e.g., [doc.pdf p.12]).
    - Format response so that both the grounded text and the cited chunks are preserved.
    """

    MULTI_PART_PATTERN = re.compile(
        r"(\band\b.*\?|"
        r"\b(?:what|which|how|who|why|where)\b.*\b(?:what|which|how|who|why|where)\b|"
        r"[,;]\s*(?:and\s+)?(?:what|which|how|who|why|where)\b|"
        r"\b(?:compare|contrast|difference between|versus|vs\.?)\b|"
        r"\b(?:explain\s+how\b.*\band\b)|"
        r"\b(?:first|second|steps?|building blocks?)\b|"
        r"[-–—]\s*(?:at\s+adoption|on\s+an\s+ongoing\s+basis|in\s+the\s+timing)\b)",
        flags=re.IGNORECASE,
    )

    @classmethod
    def is_multi_part_query(cls, query: str) -> bool:
        if not query:
            return False
        return bool(cls.MULTI_PART_PATTERN.search(query))

    @staticmethod
    def format_context_prompt(chunks: List[Dict[str, Any]]) -> Tuple[str, List[ChunkCitation]]:
        formatted_chunks = []
        citations = []

        for idx, chunk in enumerate(chunks):
            doc = chunk.get("document", "Unknown_Doc")
            page = chunk.get("page", 1)
            text = chunk.get("text", "")
            section_title = chunk.get("section_title")
            periods = chunk.get("report_periods") or chunk.get("periods") or []
            metadata = []
            if section_title:
                metadata.append(f"Section: {section_title}")
            if periods:
                period_text = ", ".join(str(period) for period in periods)
                metadata.append(f"Report period: {period_text}")
            metadata_text = f" ({'; '.join(metadata)})" if metadata else ""
            chunk_header = f"--- Source {idx+1}: [{doc} p.{page}]{metadata_text} ---"
            formatted_chunks.append(f"{chunk_header}\n{text}")
            citations.append(ChunkCitation(
                document=doc,
                page=page,
                text=text,
                section=section_title or chunk.get("doc_type"),
            ))

        return "\n\n".join(formatted_chunks), citations

    @staticmethod
    async def generate_grounded_answer(
        query: str,
        chunks: List[Dict[str, Any]],
        llm_provider: LLMProvider,
        synthesis: bool = False,
    ) -> GroundedResult:
        if not chunks:
            return GroundedResult(
                "Insufficient evidence available in the indexed collection to answer the query.",
                [],
                None,
            )

        context_str, citations = GroundedAnswerGenerator.format_context_prompt(chunks)

        is_multi_part = synthesis or GroundedAnswerGenerator.is_multi_part_query(query)
        if synthesis:
            length_instruction = (
                "2. Synthesis mode: answer every part of the question as one cited line per fact, each ending with [document p.X]. "
                "Include figures, dates, names, and conditions when stated. If a requested part is unsupported, add a line "
                "'Not stated in excerpts: [the unsupported part].' Never reply only with a refusal when any excerpt is relevant."
            )
        elif is_multi_part:
            length_instruction = (
                "2. For multi-part questions, answer each requested item directly with one cited line per requested item. "
                "Keep answers focused strictly on what was requested and cite only the single best source per requested item."
            )
        else:
            length_instruction = (
                "2. Answer only what was asked, in 1-2 sentences. Cite only the single best source. Do not add corroboration or extra details. "
                "Do not add source descriptions, section titles, as-of dates, periods, or other qualifiers unless the question explicitly asks for them."
            )

        system_prompt = (
            "You are an expert financial and regulatory research assistant providing accurate, traceable answers.\n"
            "STRICT GROUNDING RULES:\n"
            "1. Base your answer ONLY on the provided source excerpts.\n"
            f"{length_instruction}\n"
            "3. Cite each material claim using the single best source document and page number in brackets, e.g., [DocumentName.pdf p.X]. Do not cite additional sources for corroboration.\n"
            "4. When the evidence supports it, use this format: 'Answer: [requested value and its stated period or as-of date]. Source: [document p.X].' Include only period or date metadata explicitly provided in the evidence and requested by the question. Do not turn citation or source metadata into a separate sentence or claim. For comparison questions, state the change and its base year only when the evidence states them.\n"
            "5. Never invent page numbers, facts, dates, periods, or section titles, and never extrapolate beyond the text.\n"
            f"6. Reply with exactly '{NO_INFORMATION_RESPONSE}' only when none of the excerpts supports any part of the question. For partially supported multi-part questions, retain the supported cited lines and identify unsupported parts as not stated.\n"
            "7. Avoid filler commentary, unnecessary introductions, or lengthy inventories of irrelevant data."
        )

        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Retrieved Source Evidence:\n{context_str}\n\n"
            "Provide a direct, concise, and cited answer based strictly on the source evidence above:"
        )

        try:
            answer, usage = await llm_provider.generate_completion_with_usage(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=2500,
            )
            answer = answer.strip()
            stripped = REFUSAL_RE.sub("", answer).strip(" \n.-:")
            if REFUSAL_RE.search(answer) and not CITATION_RE.search(stripped):
                answer = NO_INFORMATION_RESPONSE
            return GroundedResult(answer, citations, usage)
        except Exception as e:
            logger.error(f"Error in grounded answer generation: {e}")
            # Fallback deterministic grounded summary from available chunks
            fallback_answer = (
                f"Based on retrieved evidence from {len(chunks)} sources:\n\n"
                + "\n\n".join([f"- [{c.get('document')} p.{c.get('page')}]: {c.get('text')[:300]}..." for c in chunks[:3]])
            )
            return GroundedResult(fallback_answer, citations, None)
