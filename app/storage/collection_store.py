import os
import json
import shutil
import hashlib
from typing import Dict, Any, List, Optional
from app.config import settings

class StorageManager:
    """
    Manages isolated, on-disk collections.
    Each collection lives in its own subdirectory with an atomic manifest, document metadata,
    and indexed text chunks.
    """
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or settings.DATA_DIR)
        os.makedirs(self.base_dir, exist_ok=True)

    def get_collection_dir(self, collection_name: str) -> str:
        # Sanitize collection name to prevent path traversal
        clean_name = "".join(c for c in collection_name if c.isalnum() or c in ("-", "_")).strip()
        if not clean_name:
            raise ValueError(f"Invalid collection name: {collection_name}")
        return os.path.join(self.base_dir, clean_name)

    def collection_exists(self, collection_name: str) -> bool:
        cdir = self.get_collection_dir(collection_name)
        manifest_file = os.path.join(cdir, "manifest.json")
        return os.path.isdir(cdir) and os.path.isfile(manifest_file)

    def save_collection(
        self,
        collection_name: str,
        manifest: Dict[str, Any],
        documents: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ):
        cdir = self.get_collection_dir(collection_name)
        temp_dir = f"{cdir}_tmp_{os.getpid()}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save manifest
            with open(os.path.join(temp_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # Save document metadata
            with open(os.path.join(temp_dir, "documents.json"), "w", encoding="utf-8") as f:
                json.dump(documents, f, indent=2)

            # Save chunks
            with open(os.path.join(temp_dir, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)

            # Atomic swap
            backup_dir = f"{cdir}_backup_{os.getpid()}"
            if os.path.exists(cdir):
                os.rename(cdir, backup_dir)
            os.rename(temp_dir, cdir)
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)

        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise RuntimeError(f"Failed to persist collection {collection_name}: {e}")

    def load_collection(self, collection_name: str) -> Dict[str, Any]:
        cdir = self.get_collection_dir(collection_name)
        if not self.collection_exists(collection_name):
            raise FileNotFoundError(f"Collection '{collection_name}' not found.")

        with open(os.path.join(cdir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(os.path.join(cdir, "documents.json"), "r", encoding="utf-8") as f:
            documents = json.load(f)
        with open(os.path.join(cdir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        return {
            "manifest": manifest,
            "documents": documents,
            "chunks": chunks,
        }

    def list_collections(self) -> List[str]:
        if not os.path.exists(self.base_dir):
            return []
        return [
            d for d in os.listdir(self.base_dir)
            if os.path.isfile(os.path.join(self.base_dir, d, "manifest.json"))
        ]

storage_manager = StorageManager()
