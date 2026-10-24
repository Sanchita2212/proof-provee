import os
import io
import re
import zipfile
import logging
from typing import List, Dict, Any, Tuple
from app.config import settings
from app.ingestion.extractor import DocumentExtractor

logger = logging.getLogger(__name__)

class ArchiveProcessor:
    """
    Safely processes uploaded ZIP files, extracts PDF documents,
    segments them into provenance-tracked chunks, and discovers metadata.
    Includes strict path traversal prevention and size checks.
    """

    MAX_FILE_SIZE = 150 * 1024 * 1024  # 150 MB safety cap per file
    MAX_TOTAL_FILES = 200

    @staticmethod
    def process_zip(zip_bytes: bytes) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[str]]:
        warnings = []
        documents = []
        all_chunks = []
        total_pages = 0

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                namelist = z.namelist()
                if not namelist:
                    warnings.append("Uploaded ZIP file is empty.")
                    return documents, all_chunks, {"total_docs": 0, "total_pages": 0}, warnings

                pdf_names = [
                    name for name in namelist
                    if name.lower().endswith(".pdf") and not name.startswith("__MACOSX") and not os.path.basename(name).startswith(".")
                ]

                if not pdf_names:
                    warnings.append("No valid PDF documents found in the uploaded ZIP archive.")

                for name in pdf_names[:ArchiveProcessor.MAX_TOTAL_FILES]:
                    # Prevent Zip Slip / Path Traversal
                    clean_filename = os.path.basename(name)
                    if not clean_filename:
                        continue

                    try:
                        file_info = z.getinfo(name)
                        if file_info.file_size > ArchiveProcessor.MAX_FILE_SIZE:
                            warnings.append(f"Skipping {clean_filename}: File size exceeds safety limit.")
                            continue

                        pdf_data = z.read(name)
                        pages, doc_meta, doc_warnings = DocumentExtractor.extract_from_pdf_bytes(pdf_data, clean_filename)
                        warnings.extend(doc_warnings)

                        documents.append(doc_meta)
                        total_pages += len(pages)

                        # Chunk each page preserving provenance
                        doc_chunks = ArchiveProcessor.chunk_pages(pages, clean_filename, doc_meta)
                        all_chunks.extend(doc_chunks)

                    except Exception as fe:
                        warnings.append(f"Error extracting {clean_filename} from archive: {fe}")

        except zipfile.BadZipFile:
            raise ValueError("The provided file is not a valid ZIP archive or is corrupted.")
        except Exception as e:
            raise RuntimeError(f"Archive processing error: {e}")

        manifest = {
            "total_documents": len(documents),
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
            "document_filenames": [d["filename"] for d in documents],
        }

        return documents, all_chunks, manifest, warnings

    @staticmethod
    def chunk_pages(pages: List[Dict[str, Any]], filename: str, doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Creates granular text chunks from pages.
        Maintains document provenance, page numbers, and detected section headers.
        """
        chunks = []
        chunk_size = max(1, settings.CHUNK_SIZE_CHARS)
        overlap = min(max(0, settings.CHUNK_OVERLAP_CHARS), chunk_size - 1)

        def is_section_heading(text: str) -> bool:
            normalized = " ".join(text.split())
            words = normalized.split()
            if not normalized or len(normalized) > 120 or len(words) > 14:
                return False
            if normalized[-1:] in ".?!;:":
                return False
            return (
                bool(re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", normalized))
                or normalized.isupper()
                or (len(words) <= 10 and normalized.istitle())
            )

        def split_text(text: str) -> List[str]:
            pieces = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text):
                    boundary = max(
                        text.rfind("\n\n", start, end),
                        text.rfind("\n", start, end),
                        text.rfind(". ", start, end),
                        text.rfind(" ", start, end),
                    )
                    if boundary > start + chunk_size * 0.6:
                        end = boundary + (2 if text[boundary:boundary + 2] == "\n\n" else 1)
                piece = text[start:end].strip()
                if piece:
                    pieces.append(piece)
                if end >= len(text):
                    break
                start = max(start + 1, end - overlap)
                while start < len(text) and text[start].isspace():
                    start += 1
            return pieces

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            if not text:
                continue

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            section_heading = None
            section_paragraphs = []

            def append_section_chunks() -> None:
                section_text = "\n\n".join(section_paragraphs)
                for piece in split_text(section_text):
                    chunk_text = f"{section_heading}\n{piece}" if section_heading else piece
                    chunks.append({
                        "chunk_id": f"{filename}_p{page_num}_c{len(chunks)}",
                        "document": filename,
                        # Contract-facing provenance.  Chunks currently do
                        # not span pages, but a list makes the representation
                        # lossless if a future extractor emits page ranges.
                        "file_name": filename,
                        "page": page_num,
                        "page_numbers": [page_num],
                        "text": chunk_text,
                        "section_title": section_heading,
                        "report_periods": doc_meta.get("periods", []),
                        "entities": doc_meta.get("entities", []),
                        "periods": doc_meta.get("periods", []),
                        "scope": doc_meta.get("scope"),
                        "doc_type": doc_meta.get("document_type"),
                    })

            for paragraph in paragraphs:
                if is_section_heading(paragraph):
                    if section_paragraphs:
                        append_section_chunks()
                        section_paragraphs = []
                    section_heading = " ".join(paragraph.split())
                else:
                    section_paragraphs.append(paragraph)

            if section_paragraphs:
                append_section_chunks()

        return chunks
