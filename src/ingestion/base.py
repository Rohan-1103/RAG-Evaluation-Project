"""
src/ingestion/base.py

Abstract base class for all document loaders.

Design contract:
  - Every loader receives a Path and returns list[Document].
  - No loader ever touches chunking, embedding, or storage.
    Single responsibility: raw file → structured Document objects.
  - Loaders are stateless. The same instance can load multiple files.
  - All loaders must implement `supported_extensions` so the
    IngestionPipeline can auto-select the correct loader for a file
    without any if/elif format-checking logic in the orchestrator.

Dependency injection pattern:
    pipeline = IngestionPipeline(
        loader=PDFLoader(settings.ingestion),
        chunker=RecursiveChunker(settings.ingestion),
        embedder=EmbeddingManager(settings.embedding),
        vectorstore=ChromaVectorStore(settings.chroma),
    )
    The pipeline never imports PDFLoader directly — it depends on
    BaseLoader. Swapping loaders requires zero pipeline code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentMetadata:
    """
    Structured metadata attached to every loaded Document.

    Frozen dataclass — metadata is set at load time and never mutated.
    Using a typed dataclass instead of dict[str, Any] means:
      - Every downstream module knows exactly what fields exist.
      - Missing fields are caught at construction, not at access time.
      - The schema is self-documenting.

    Fields marked with default=None are optional — not every format
    can provide every field (e.g. plain text has no page number).
    """

    source_file: str                        # Original filename e.g. "policy.pdf"
    source_path: str                        # Absolute path as string
    file_type: str                          # Extension without dot: "pdf", "txt", "html"
    file_size_bytes: int                    # Raw file size
    total_pages: int | None = None          # PDF/DOCX page count; None for txt/html
    page_number: int | None = None          # Page this document came from; None if N/A
    title: str | None = None               # Extracted document title if available
    author: str | None = None              # Extracted author if available
    created_at: str | None = None          # ISO 8601 string if extractable
    loader_class: str = ""                  # Set automatically by BaseLoader.load()
    extra: dict[str, Any] = field(         # Escape hatch for format-specific fields
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to flat dict for ChromaDB metadata storage.

        ChromaDB requires metadata values to be str | int | float | bool.
        None values are converted to empty string to satisfy this constraint.
        Nested dicts (extra) are JSON-serialised to string.
        """
        import json

        result: dict[str, Any] = {
            "source_file":      self.source_file,
            "source_path":      self.source_path,
            "file_type":        self.file_type,
            "file_size_bytes":  self.file_size_bytes,
            "total_pages":      self.total_pages   if self.total_pages   is not None else "",
            "page_number":      self.page_number   if self.page_number   is not None else "",
            "title":            self.title         if self.title         is not None else "",
            "author":           self.author        if self.author        is not None else "",
            "created_at":       self.created_at    if self.created_at    is not None else "",
            "loader_class":     self.loader_class,
            "extra":            json.dumps(self.extra) if self.extra else "{}",
        }
        return result


@dataclass
class Document:
    """
    A single unit of loaded text content with structured metadata.

    This is the universal currency of the ingestion layer.
    Every loader produces Documents. Every chunker consumes and
    produces Documents. The vectorstore indexes Documents.

    Not frozen — chunkers must be able to set chunk-level fields
    (chunk_index, chunk_of) after splitting a parent Document.

    Design note: we define our own Document rather than using
    LangChain's because:
      1. Full control over the metadata schema (typed, not dict).
      2. No hidden LangChain version coupling in the data model.
      3. LangChain Documents can be constructed from ours trivially
         when the vectorstore layer needs them.
    """

    page_content: str
    metadata: DocumentMetadata

    # Chunk-level fields — set by RecursiveChunker, not by loaders
    chunk_index: int | None = None      # 0-based index of this chunk
    chunk_of: int | None = None         # Total chunks this document was split into
    parent_doc_id: str | None = None    # ID of the pre-split parent document

    def __post_init__(self) -> None:
        if not isinstance(self.page_content, str):
            raise TypeError(
                f"Document.page_content must be str, "
                f"got {type(self.page_content).__name__}."
            )
        if not isinstance(self.metadata, DocumentMetadata):
            raise TypeError(
                f"Document.metadata must be DocumentMetadata, "
                f"got {type(self.metadata).__name__}."
            )

    @property
    def is_chunk(self) -> bool:
        """True if this document is a chunk of a larger parent document."""
        return self.chunk_index is not None

    @property
    def content_length(self) -> int:
        """Character count of page_content."""
        return len(self.page_content)

    @property
    def is_empty(self) -> bool:
        """True if page_content is blank after stripping whitespace."""
        return not self.page_content.strip()

    def to_langchain_document(self) -> Any:
        """
        Convert to LangChain Document for compatibility with
        LangChain retrievers and vectorstores.

        Import is deferred — this method is only called by the
        vectorstore layer, not by loaders or chunkers.
        """
        from langchain_core.documents import Document as LCDocument

        meta = self.metadata.to_dict()
        if self.chunk_index is not None:
            meta["chunk_index"] = self.chunk_index
        if self.chunk_of is not None:
            meta["chunk_of"] = self.chunk_of
        if self.parent_doc_id is not None:
            meta["parent_doc_id"] = self.parent_doc_id

        return LCDocument(
            page_content=self.page_content,
            metadata=meta,
        )

    def __repr__(self) -> str:
        return (
            f"Document("
            f"source='{self.metadata.source_file}', "
            f"page={self.metadata.page_number}, "
            f"chunk={self.chunk_index}/{self.chunk_of}, "
            f"length={self.content_length}"
            f")"
        )


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.

    Subclasses implement:
      - supported_extensions — declares which file types this loader handles
      - _load_file           — core loading logic for a single file

    Subclasses must NOT implement:
      - Chunking logic         (RecursiveChunker's responsibility)
      - Embedding logic        (EmbeddingManager's responsibility)
      - Storage logic          (VectorStore's responsibility)
      - Retry logic            (IngestionPipeline's responsibility)

    The load() method is concrete — it handles common concerns:
      - Path validation
      - Extension checking
      - Empty document filtering
      - Automatic loader_class stamping on metadata
    Subclasses never override load() — only _load_file().
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """
        Return the file extensions this loader handles.

        Extensions are lowercase without the leading dot.
        Examples: frozenset({"pdf"}), frozenset({"txt", "text", "md"})

        Used by LoaderRegistry to auto-select the correct loader.
        Must return frozenset (immutable) — callers must not be able
        to mutate the loader's declared capabilities.
        """
        ...

    @abstractmethod
    def _load_file(self, path: Path) -> list[Document]:
        """
        Core loading logic. Called by load() after validation.

        Implementations must:
          - Return at least one Document for non-empty files.
          - Populate DocumentMetadata as completely as the format allows.
          - Raise LoaderError on unrecoverable parsing failures.
          - Never return Documents with empty page_content
            (load() filters these, but _load_file should not produce them).

        Implementations must NOT:
          - Catch all exceptions silently.
          - Modify the path or rename/move the file.
          - Perform any I/O other than reading the given path.
        """
        ...

    def load(self, path: Path) -> list[Document]:
        """
        Public loading interface. Do not override in subclasses.

        Handles:
          1. Path existence validation
          2. Extension compatibility check
          3. Delegation to _load_file
          4. Empty document filtering
          5. Automatic loader_class stamping on metadata

        Returns an empty list (not raises) if the file is valid but
        contains no extractable text — this is a data quality issue,
        not a programming error. The pipeline logs and skips these.
        """
        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__}: file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"{self.__class__.__name__}: path is not a file: {path}"
            )

        ext = path.suffix.lstrip(".").lower()
        if ext not in self.supported_extensions:
            raise UnsupportedFormatError(
                loader=self.__class__.__name__,
                extension=ext,
                supported=self.supported_extensions,
                path=path,
            )

        documents = self._load_file(path)

        # Filter empty documents — log how many were dropped
        non_empty = [doc for doc in documents if not doc.is_empty]
        dropped = len(documents) - len(non_empty)
        if dropped > 0:
            import sys
            print(
                f"[{self.__class__.__name__}] Dropped {dropped} empty "
                f"document(s) from '{path.name}'.",
                file=sys.stderr,
            )

        # Stamp loader_class on every document's metadata
        # Done here (not in _load_file) so subclasses never forget it
        stamped: list[Document] = []
        for doc in non_empty:
            stamped_meta = DocumentMetadata(
                **{
                    **{
                        f: getattr(doc.metadata, f)
                        for f in doc.metadata.__dataclass_fields__
                        if f != "loader_class"
                    },
                    "loader_class": self.__class__.__name__,
                }
            )
            stamped.append(
                Document(
                    page_content=doc.page_content,
                    metadata=stamped_meta,
                    chunk_index=doc.chunk_index,
                    chunk_of=doc.chunk_of,
                    parent_doc_id=doc.parent_doc_id,
                )
            )

        return stamped

    def can_load(self, path: Path) -> bool:
        """
        Return True if this loader can handle the given file.

        Used by LoaderRegistry.get_loader_for() without raising.
        """
        ext = path.suffix.lstrip(".").lower()
        return ext in self.supported_extensions

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"extensions={sorted(self.supported_extensions)})"
        )


class LoaderRegistry:
    """
    Maps file extensions to loader instances.

    Centralises loader selection so the IngestionPipeline never
    contains format-checking logic. Adding support for a new file
    format requires:
      1. Implementing a new BaseLoader subclass.
      2. Registering it here — zero pipeline changes.

    Usage:
        registry = LoaderRegistry()
        registry.register(PDFLoader(settings.ingestion))
        registry.register(TXTLoader(settings.ingestion))

        loader = registry.get_loader_for(Path("report.pdf"))
        documents = loader.load(Path("report.pdf"))
    """

    def __init__(self) -> None:
        self._registry: dict[str, BaseLoader] = {}

    def register(self, loader: BaseLoader) -> None:
        """
        Register a loader for all its supported extensions.

        Raises ValueError if an extension is already registered —
        ambiguous routing is a configuration error, not a runtime error.
        """
        for ext in loader.supported_extensions:
            if ext in self._registry:
                existing = self._registry[ext].__class__.__name__
                raise ValueError(
                    f"Extension '.{ext}' is already registered to "
                    f"'{existing}'. Cannot register "
                    f"'{loader.__class__.__name__}' for the same extension. "
                    f"Unregister the existing loader first."
                )
            self._registry[ext] = loader

    def get_loader_for(self, path: Path) -> BaseLoader:
        """
        Return the registered loader for the given file's extension.

        Raises UnsupportedFormatError if no loader is registered.
        Never returns None — callers should not need to null-check.
        """
        ext = path.suffix.lstrip(".").lower()
        if ext not in self._registry:
            raise UnsupportedFormatError(
                loader="LoaderRegistry",
                extension=ext,
                supported=frozenset(self._registry.keys()),
                path=path,
            )
        return self._registry[ext]

    def supported_extensions(self) -> frozenset[str]:
        """Return all extensions currently registered."""
        return frozenset(self._registry.keys())

    def is_supported(self, path: Path) -> bool:
        """Return True if a loader is registered for this file's extension."""
        ext = path.suffix.lstrip(".").lower()
        return ext in self._registry

    def registered_loaders(self) -> list[BaseLoader]:
        """Return unique loader instances (deduplicated by identity)."""
        seen: set[int] = set()
        result: list[BaseLoader] = []
        for loader in self._registry.values():
            if id(loader) not in seen:
                seen.add(id(loader))
                result.append(loader)
        return result

    def __repr__(self) -> str:
        mapping = {
            ext: loader.__class__.__name__
            for ext, loader in self._registry.items()
        }
        return f"LoaderRegistry({mapping})"


# ===========================================================================
# CUSTOM EXCEPTIONS
# Specific exception types force callers to handle ingestion failures
# explicitly. Catching bare Exception in the pipeline is a code smell —
# catching LoaderError or UnsupportedFormatError is intentional.
# ===========================================================================


class LoaderError(Exception):
    """
    Raised when a loader fails to parse a file it is responsible for.

    Distinct from FileNotFoundError (path issue) and
    UnsupportedFormatError (wrong loader). LoaderError means:
    "the file exists, I own this format, but I cannot parse it."

    The pipeline catches LoaderError per-file and continues
    (if continue_on_failure=True), recording the failure.
    """

    def __init__(
        self,
        loader: str,
        path: Path,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.loader = loader
        self.path = path
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[{loader}] Failed to load '{path.name}': {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )


class UnsupportedFormatError(Exception):
    """
    Raised when no registered loader supports the given file extension.

    Provides an actionable message listing what IS supported,
    so the developer immediately knows what to implement or register.
    """

    def __init__(
        self,
        loader: str,
        extension: str,
        supported: frozenset[str],
        path: Path,
    ) -> None:
        self.loader = loader
        self.extension = extension
        self.supported = supported
        self.path = path
        supported_str = ", ".join(f".{e}" for e in sorted(supported))
        super().__init__(
            f"[{loader}] Unsupported file extension '.{extension}' "
            f"for file '{path.name}'. "
            f"Supported extensions: {supported_str or 'none registered'}."
        )


__all__ = [
    "DocumentMetadata",
    "Document",
    "BaseLoader",
    "LoaderRegistry",
    "LoaderError",
    "UnsupportedFormatError",
]