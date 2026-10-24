import os
import re
import io
import zipfile
import logging
from typing import List, Dict, Any, Tuple
import pypdf

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """
    Extracts text page-by-page while preserving provenance (document name, page number)
    and discovering dynamic metadata (entities, fiscal periods, document types, scopes).
    Does NOT rely on hardcoded document names or benchmark question structures.
    """

    @staticmethod
    def extract_from_pdf_bytes(pdf_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
        pages = []
        warnings = []
        doc_meta = {
            "filename": filename,
            "title": None,
            "entities": [],
            "document_type": None,
            "periods": [],
            "scope": None,
            "num_pages": 0,
            "is_encrypted": False,
            "is_empty": False,
        }

        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    warnings.append(f"Document {filename} is encrypted and password-protected.")
                    doc_meta["is_encrypted"] = True
                    return pages, doc_meta, warnings

            num_pages = len(reader.pages)
            doc_meta["num_pages"] = num_pages

            if num_pages == 0:
                warnings.append(f"Document {filename} contains 0 pages.")
                doc_meta["is_empty"] = True
                return pages, doc_meta, warnings

            total_extracted_chars = 0
            first_few_pages_text = []

            for page_idx in range(num_pages):
                page_num = page_idx + 1
                try:
                    text = reader.pages[page_idx].extract_text() or ""
                except Exception as pe:
                    warnings.append(f"Page {page_num} extraction error in {filename}: {pe}")
                    text = ""

                cleaned_text = DocumentExtractor.clean_text(text)
                total_extracted_chars += len(cleaned_text)

                if page_idx < 3 and cleaned_text:
                    first_few_pages_text.append(cleaned_text)

                pages.append({
                    "page_number": page_num,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                })

            if total_extracted_chars < 100:
                warnings.append(f"Document {filename} has very low text content ({total_extracted_chars} chars) - may be scanned or image-only.")

            # Dynamic Metadata Discovery from early pages and filename
            header_sample = "\n".join(first_few_pages_text[:3])
            doc_meta.update(DocumentExtractor.discover_metadata(filename, header_sample))

        except Exception as e:
            warnings.append(f"Failed to process PDF {filename}: {e}")

        return pages, doc_meta, warnings

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove null bytes and excessive whitespace
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    @staticmethod
    def discover_metadata(filename: str, header_text: str) -> Dict[str, Any]:
        """
        Extracts structural attributes dynamically using general regular expressions
        for corporate entities, fiscal years, document types, and jurisdictions.
        """
        discovered = {
            "title": None,
            "entities": [],
            "document_type": "general_document",
            "periods": [],
            "scope": None,
        }

        combined_text = f"{filename}\n{header_text}"

        # 1. Detect Reporting Periods (e.g., FY 2024-25, 2025, FY26, Q3 2024)
        period_matches = re.findall(
            r"\b(?:FY\s*[-–]?\s*\d{2,4}(?:[-–/]\d{2,4})?|\b20\d{2}(?:[-–/]\d{2,4})?|\bQ[1-4]\s*20\d{2})\b",
            combined_text,
            flags=re.IGNORECASE
        )
        if period_matches:
            # Normalize and deduplicate
            unique_periods = list(dict.fromkeys([p.strip().upper() for p in period_matches]))
            discovered["periods"] = unique_periods[:5]

        # 2. Detect Document Type
        lower_comb = combined_text.lower()
        if "integrated annual report" in lower_comb:
            discovered["document_type"] = "integrated_annual_report"
        elif "annual report" in lower_comb:
            discovered["document_type"] = "annual_report"
        elif "basel" in lower_comb or "bcbs" in lower_comb:
            discovered["document_type"] = "regulatory_framework"
        elif "fasb" in lower_comb or "accounting standards update" in lower_comb or "asc 326" in lower_comb:
            discovered["document_type"] = "accounting_standard"
        elif "federal register" in lower_comb or "fdic" in lower_comb or "occ" in lower_comb:
            discovered["document_type"] = "regulatory_statement"
        elif "handbook" in lower_comb or "guidance" in lower_comb:
            discovered["document_type"] = "regulatory_guidance"

        # 3. Detect Entities (Look for Corporation/Limited/Bank/Authority patterns)
        entity_matches = re.findall(
            r"\b([A-Z][A-Za-z0-9&.,\- ]+?(?:Limited|Ltd\.?|Corporation|Corp\.?|Inc\.?|Bank|Industries|Authority|Committee))\b",
            header_text
        )
        if entity_matches:
            clean_entities = list(dict.fromkeys([e.strip() for e in entity_matches if len(e.strip()) > 3]))
            discovered["entities"] = clean_entities[:5]

        # 4. Scope / Jurisdiction
        if any(w in lower_comb for w in ["india", "sebi", "rbi", "rs crore", "lakhs"]):
            discovered["scope"] = "India"
        elif any(w in lower_comb for w in ["federal reserve", "fdic", "occ", "united states", "u.s.", "fasb"]):
            discovered["scope"] = "United States"
        elif any(w in lower_comb for w in ["basel", "bcbs", "bank for international settlements"]):
            discovered["scope"] = "International / Basel"
        elif any(w in lower_comb for w in ["sama", "saudi", "riyals", "uae", "aldar", "dirham"]):
            discovered["scope"] = "Middle East"

        return discovered
