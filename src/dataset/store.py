"""
src/dataset/store.py

DatasetStore — persistent storage and retrieval for EvalDatasets.

Responsibilities:
  - Persist EvalDatasets to versioned JSON files on disk.
  - Load datasets by ID, name, or listing all available datasets.
  - Maintain an index file (index.json) for fast metadata lookup
    without loading full dataset files.
  - Support dataset versioning: saving an updated dataset increments
    its version and archives the previous version.
  - Export datasets to CSV for external analysis.

Storage layout:
    data/datasets/
    ├── index.json                    ← Fast metadata index (all datasets)
    ├── ds_01J2K3M.../
    │   ├── dataset.json              ← Current version (full EvalDataset)
    │   ├── metadata.json             ← DatasetMetadata only (fast load)
    │   └── archive/
    │       ├── dataset_v1.0.0.json   ← Previous versions
    │       └── dataset_v1.0.1.json
    └── ds_01J2K3N.../
        ├── dataset.json
        └── metadata.json

Why file-based storage instead of SQLite for datasets:
  - EvalDatasets are large nested JSON objects. Storing them in
    SQLite requires either a single JSON column (loses queryability)
    or a complex normalised schema with 5+ tables.
  - File-based storage makes datasets directly inspectable with any
    text editor — useful for debugging and manual review.
  - Datasets are written once and read many times. File I/O is
    perfectly adequate for this access pattern.
  - The index.json provides O(1) metadata lookup without loading
    full dataset files — the UI list view only reads index.json.

Run results (EvalReports) are stored in SQLite via RunRepository
because they are queried across runs, filtered, and aggregated.
Datasets are stored in files because they are self-contained units.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings
from src.dataset.schema import (
    DatasetMetadata,
    DatasetStatus,
    EvalDataset,
    GenerationMethod,
)

# ===========================================================================
# INDEX ENTRY
# ===========================================================================

class DatasetIndexEntry:
    """
    Lightweight representation of a dataset in the index.

    The index stores one entry per dataset — just enough to populate
    the UI dataset list without loading full EvalDataset files.

    Serialisable to/from plain dict for JSON persistence.
    """

    __slots__ = (
        "id",
        "name",
        "description",
        "created_at",
        "updated_at",
        "generation_method",
        "source_collection",
        "source_files",
        "generator_model",
        "total_pairs",
        "evaluated_pairs",
        "status",
        "tags",
        "version",
        "dataset_dir",
    )

    def __init__(
        self,
        id: str,
        name: str,
        description: str | None,
        created_at: str,
        updated_at: str,
        generation_method: str,
        source_collection: str | None,
        source_files: list[str],
        generator_model: str | None,
        total_pairs: int,
        evaluated_pairs: int,
        status: str,
        tags: list[str],
        version: str,
        dataset_dir: str,
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        self.created_at = created_at
        self.updated_at = updated_at
        self.generation_method = generation_method
        self.source_collection = source_collection
        self.source_files = source_files
        self.generator_model = generator_model
        self.total_pairs = total_pairs
        self.evaluated_pairs = evaluated_pairs
        self.status = status
        self.tags = tags
        self.version = version
        self.dataset_dir = dataset_dir

    @classmethod
    def from_eval_dataset(
        cls,
        dataset: EvalDataset,
        dataset_dir: Path,
    ) -> DatasetIndexEntry:
        """Build an index entry from a full EvalDataset."""
        evaluated = sum(
            1 for p in dataset.pairs
            if p.status.value == "evaluated"
        )
        return cls(
            id=dataset.metadata.id,
            name=dataset.metadata.name,
            description=dataset.metadata.description,
            created_at=dataset.metadata.created_at.isoformat(),
            updated_at=dataset.metadata.updated_at.isoformat(),
            generation_method=dataset.metadata.generation_method.value,
            source_collection=dataset.metadata.source_collection,
            source_files=dataset.metadata.source_files,
            generator_model=dataset.metadata.generator_model,
            total_pairs=len(dataset.pairs),
            evaluated_pairs=evaluated,
            status=dataset.metadata.status.value,
            tags=dataset.metadata.tags,
            version=dataset.metadata.version,
            dataset_dir=str(dataset_dir),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":                self.id,
            "name":              self.name,
            "description":       self.description,
            "created_at":        self.created_at,
            "updated_at":        self.updated_at,
            "generation_method": self.generation_method,
            "source_collection": self.source_collection,
            "source_files":      self.source_files,
            "generator_model":   self.generator_model,
            "total_pairs":       self.total_pairs,
            "evaluated_pairs":   self.evaluated_pairs,
            "status":            self.status,
            "tags":              self.tags,
            "version":           self.version,
            "dataset_dir":       self.dataset_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetIndexEntry:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            generation_method=data["generation_method"],
            source_collection=data.get("source_collection"),
            source_files=data.get("source_files", []),
            generator_model=data.get("generator_model"),
            total_pairs=data["total_pairs"],
            evaluated_pairs=data.get("evaluated_pairs", 0),
            status=data["status"],
            tags=data.get("tags", []),
            version=data.get("version", "1.0.0"),
            dataset_dir=data["dataset_dir"],
        )

    @property
    def completion_rate(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return self.evaluated_pairs / self.total_pairs

    def __repr__(self) -> str:
        return (
            f"DatasetIndexEntry("
            f"id='{self.id}', "
            f"name='{self.name}', "
            f"pairs={self.total_pairs}, "
            f"status={self.status})"
        )

# ===========================================================================
# DATASET STORE
# ===========================================================================

class DatasetStore:
    """
    Manages persistence of EvalDatasets to the filesystem.

    All dataset files are stored under a configurable base directory.
    The store maintains an index.json at the root for fast listing.

    Constructor:
        store = DatasetStore(base_dir=Path("./data/datasets"))

    Factory:
        store = DatasetStore.from_settings(settings)

    Usage:
        # Save a new dataset
        store.save(dataset)

        # Load by ID
        dataset = store.load("ds_01J2K3M...")

        # List all datasets (metadata only, fast)
        entries = store.list_datasets()

        # Delete
        store.delete("ds_01J2K3M...")

        # Export to CSV
        csv_path = store.export_csv("ds_01J2K3M...", output_dir)
    """

    _INDEX_FILE = "index.json"
    _DATASET_FILE = "dataset.json"
    _METADATA_FILE = "metadata.json"
    _ARCHIVE_DIR = "archive"

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base_dir / self._INDEX_FILE

        # In-memory index cache — invalidated on every write
        self._index_cache: dict[str, DatasetIndexEntry] | None = None

        logger.info(
            f"DatasetStore initialised. "
            f"base_dir='{self._base_dir}'"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> DatasetStore:
        """Construct from application Settings."""
        return cls(base_dir=settings.storage.datasets_dir)

    # ------------------------------------------------------------------
    # WRITE OPERATIONS
    # ------------------------------------------------------------------

    def save(
        self,
        dataset: EvalDataset,
        archive_previous: bool = True,
    ) -> Path:
        """
        Persist an EvalDataset to disk.

        If the dataset ID already exists:
          - The current dataset.json is archived with the version suffix
            (if archive_previous=True).
          - The new dataset.json replaces it.
          - The version is auto-incremented.

        If the dataset ID is new:
          - A new directory is created.
          - dataset.json and metadata.json are written.
          - The index is updated.

        Args:
            dataset:          The EvalDataset to persist.
            archive_previous: If True, archive current version before
                              overwriting. Set False for rapid iteration
                              during development to save disk space.

        Returns:
            Path to the saved dataset.json file.
        """
        dataset_dir = self._dataset_dir(dataset.metadata.id)
        is_update = dataset_dir.exists()

        if is_update and archive_previous:
            self._archive_current_version(dataset_dir, dataset.metadata)

        # Create directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = dataset_dir / self._ARCHIVE_DIR
        archive_dir.mkdir(exist_ok=True)

        # Auto-increment version on update
        if is_update:
            new_version = self._increment_version(dataset.metadata.version)
            # We cannot mutate frozen metadata directly —
            # rebuild with incremented version
            updated_metadata = self._rebuild_metadata_with_version(
                dataset.metadata,
                new_version,
            )
            # Reconstruct dataset with updated metadata
            dataset = EvalDataset(
                metadata=updated_metadata,
                pairs=dataset.pairs,
            )

        # Update updated_at timestamp
        dataset = self._touch_updated_at(dataset)

        # Write dataset.json (atomic)
        dataset_path = dataset_dir / self._DATASET_FILE
        dataset.to_file(dataset_path)

        # Write metadata.json (fast load for UI list)
        self._write_metadata(
            dataset.metadata,
            dataset_dir / self._METADATA_FILE,
        )

        # Update index
        self._update_index(
            DatasetIndexEntry.from_eval_dataset(dataset, dataset_dir)
        )

        action = "Updated" if is_update else "Saved"
        logger.info(
            f"DatasetStore: {action} dataset '{dataset.metadata.id}' "
            f"v{dataset.metadata.version} "
            f"({len(dataset.pairs)} pairs) → '{dataset_path}'"
        )

        return dataset_path

    def delete(self, dataset_id: str) -> bool:
        """
        Delete a dataset and all its files.

        Args:
            dataset_id: Dataset ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        dataset_dir = self._dataset_dir(dataset_id)

        if not dataset_dir.exists():
            logger.warning(
                f"DatasetStore.delete: Dataset '{dataset_id}' not found."
            )
            return False

        try:
            shutil.rmtree(dataset_dir)
            self._remove_from_index(dataset_id)
            logger.info(
                f"DatasetStore: Deleted dataset '{dataset_id}' "
                f"and all its files."
            )
            return True
        except Exception as exc:
            raise DatasetStoreError(
                operation="delete",
                dataset_id=dataset_id,
                reason=str(exc),
                original_exception=exc,
            ) from exc

    def update_status(
        self,
        dataset_id: str,
        status: DatasetStatus,
    ) -> None:
        """
        Update a dataset's status without rewriting the full file.

        Used by EvaluationEngine to transition:
          ready → running → completed / partial
        Rewrites only metadata.json and updates the index.
        Full dataset.json is rewritten only at end of eval run.
        """
        dataset = self.load(dataset_id)

        updated_meta = DatasetMetadata(
            id=dataset.metadata.id,
            name=dataset.metadata.name,
            description=dataset.metadata.description,
            created_at=dataset.metadata.created_at,
            updated_at=datetime.now(timezone.utc),
            generation_method=dataset.metadata.generation_method,
            source_collection=dataset.metadata.source_collection,
            source_files=dataset.metadata.source_files,
            generator_model=dataset.metadata.generator_model,
            total_pairs=dataset.metadata.total_pairs,
            status=status,
            tags=dataset.metadata.tags,
            version=dataset.metadata.version,
        )

        dataset_dir = self._dataset_dir(dataset_id)
        self._write_metadata(
            updated_meta,
            dataset_dir / self._METADATA_FILE,
        )

        # Update index entry status
        index = self._load_index()
        if dataset_id in index:
            entry = index[dataset_id]
            index[dataset_id] = DatasetIndexEntry(
                id=entry.id,
                name=entry.name,
                description=entry.description,
                created_at=entry.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
                generation_method=entry.generation_method,
                source_collection=entry.source_collection,
                source_files=entry.source_files,
                generator_model=entry.generator_model,
                total_pairs=entry.total_pairs,
                evaluated_pairs=entry.evaluated_pairs,
                status=status.value,
                tags=entry.tags,
                version=entry.version,
                dataset_dir=entry.dataset_dir,
            )
            self._persist_index(index)

        logger.debug(
            f"DatasetStore: Updated status of '{dataset_id}' "
            f"→ {status.value}"
        )

    # ------------------------------------------------------------------
    # READ OPERATIONS
    # ------------------------------------------------------------------

    def load(self, dataset_id: str) -> EvalDataset:
        """
        Load a full EvalDataset by ID.

        Args:
            dataset_id: Dataset ID (e.g. 'ds_01J2K3M...').

        Returns:
            EvalDataset with all pairs.

        Raises:
            DatasetNotFoundError: if dataset ID does not exist.
            DatasetStoreError:    if the file is corrupted.
        """
        dataset_path = self._dataset_dir(dataset_id) / self._DATASET_FILE

        if not dataset_path.exists():
            raise DatasetNotFoundError(dataset_id=dataset_id)

        try:
            dataset = EvalDataset.from_file(dataset_path)
            logger.debug(
                f"DatasetStore: Loaded '{dataset_id}' "
                f"({len(dataset.pairs)} pairs) "
                f"from '{dataset_path}'"
            )
            return dataset
        except Exception as exc:
            raise DatasetStoreError(
                operation="load",
                dataset_id=dataset_id,
                reason=f"Failed to parse dataset file: {exc}",
                original_exception=exc,
            ) from exc

    def load_metadata(self, dataset_id: str) -> DatasetMetadata:
        """
        Load only the metadata for a dataset (fast, no pairs).

        Used by the UI when only header information is needed,
        e.g. in the evaluation run configuration screen.

        Raises:
            DatasetNotFoundError: if dataset ID does not exist.
        """
        meta_path = (
            self._dataset_dir(dataset_id) / self._METADATA_FILE
        )

        if not meta_path.exists():
            # Fall back to loading full dataset
            return self.load(dataset_id).metadata

        try:
            raw = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
            return DatasetMetadata.model_validate(raw)
        except Exception as exc:
            raise DatasetStoreError(
                operation="load_metadata",
                dataset_id=dataset_id,
                reason=f"Failed to parse metadata file: {exc}",
                original_exception=exc,
            ) from exc

    def exists(self, dataset_id: str) -> bool:
        """Return True if a dataset with the given ID exists."""
        return (
            self._dataset_dir(dataset_id) / self._DATASET_FILE
        ).exists()

    def list_datasets(
        self,
        status_filter: DatasetStatus | None = None,
        generation_method_filter: GenerationMethod | None = None,
        tag_filter: str | None = None,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> list[DatasetIndexEntry]:
        """
        List all datasets from the index (fast — no full file loads).

        Args:
            status_filter:            Only return datasets with this status.
            generation_method_filter: Only return datasets with this method.
            tag_filter:               Only return datasets with this tag.
            sort_by:                  Field to sort by. Options:
                                      "created_at", "updated_at", "name",
                                      "total_pairs", "completion_rate".
            descending:               Sort direction.

        Returns:
            List of DatasetIndexEntry sorted by sort_by.
        """
        index = self._load_index()
        entries = list(index.values())

        # Apply filters
        if status_filter is not None:
            entries = [
                e for e in entries
                if e.status == status_filter.value
            ]

        if generation_method_filter is not None:
            entries = [
                e for e in entries
                if e.generation_method == generation_method_filter.value
            ]

        if tag_filter is not None:
            entries = [
                e for e in entries
                if tag_filter in e.tags
            ]

        # Sort
        sort_key_map = {
            "created_at":      lambda e: e.created_at,
            "updated_at":      lambda e: e.updated_at,
            "name":            lambda e: e.name.lower(),
            "total_pairs":     lambda e: e.total_pairs,
            "completion_rate": lambda e: e.completion_rate,
        }

        key_fn = sort_key_map.get(sort_by, sort_key_map["created_at"])
        entries.sort(key=key_fn, reverse=descending)

        return entries

    def list_archive_versions(
        self,
        dataset_id: str,
    ) -> list[str]:
        """
        Return a list of archived version filenames for a dataset.

        Returns:
            List of filename strings (e.g. ["dataset_v1.0.0.json"]),
            sorted newest first.
            Empty list if no archive exists.
        """
        archive_dir = (
            self._dataset_dir(dataset_id) / self._ARCHIVE_DIR
        )

        if not archive_dir.exists():
            return []

        versions = sorted(
            [f.name for f in archive_dir.glob("dataset_v*.json")],
            reverse=True,
        )
        return versions

    def load_archived_version(
        self,
        dataset_id: str,
        version_filename: str,
    ) -> EvalDataset:
        """
        Load a specific archived version of a dataset.

        Args:
            dataset_id:        Dataset ID.
            version_filename:  Filename from list_archive_versions()
                               (e.g. "dataset_v1.0.0.json").

        Raises:
            DatasetNotFoundError: if the archive file does not exist.
        """
        archive_path = (
            self._dataset_dir(dataset_id)
            / self._ARCHIVE_DIR
            / version_filename
        )

        if not archive_path.exists():
            raise DatasetNotFoundError(
                dataset_id=dataset_id,
                detail=f"Archive version '{version_filename}' not found.",
            )

        try:
            return EvalDataset.from_file(archive_path)
        except Exception as exc:
            raise DatasetStoreError(
                operation="load_archived_version",
                dataset_id=dataset_id,
                reason=f"Failed to parse archive file: {exc}",
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # EXPORT OPERATIONS
    # ------------------------------------------------------------------

    def export_csv(
        self,
        dataset_id: str,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Export an EvalDataset's Q&A pairs to a CSV file.

        Args:
            dataset_id:  Dataset to export.
            output_dir:  Directory to write CSV into.
                         Defaults to the dataset's own directory.

        Returns:
            Path to the written CSV file.
        """
        dataset = self.load(dataset_id)
        rows = dataset.to_csv_rows()

        if not rows:
            raise DatasetStoreError(
                operation="export_csv",
                dataset_id=dataset_id,
                reason="Dataset has no pairs to export.",
            )

        df = pd.DataFrame(rows)

        target_dir = output_dir or self._dataset_dir(dataset_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Sanitise dataset name for filename
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in dataset.metadata.name
        )
        csv_filename = (
            f"{safe_name}_{dataset_id[-8:]}.csv"
        )
        csv_path = target_dir / csv_filename

        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(
            f"DatasetStore: Exported '{dataset_id}' "
            f"→ '{csv_path}' ({len(rows)} rows)"
        )

        return csv_path

    def export_json_minimal(
        self,
        dataset_id: str,
        output_dir: Path | None = None,
        include_evaluated_only: bool = False,
    ) -> Path:
        """
        Export a minimal JSON file containing only question, reference,
        and (optionally) generated answer + scores.

        Useful for sharing datasets with external tools without
        exposing full internal schema.

        Args:
            dataset_id:           Dataset to export.
            output_dir:           Output directory.
            include_evaluated_only: If True, only include evaluated pairs.

        Returns:
            Path to the written JSON file.
        """
        dataset = self.load(dataset_id)

        pairs = dataset.pairs
        if include_evaluated_only:
            pairs = dataset.evaluated_pairs

        minimal = [
            {
                "id":               pair.id,
                "question":         pair.question,
                "ground_truth":     pair.ground_truth_answer,
                "source_file":      pair.source_file,
                "source_page":      pair.source_page,
                "generated_answer": pair.generated_answer,
                "composite_score":  pair.composite_score,
                "scores": {
                    name: {
                        "score":     ms.score,
                        "reasoning": ms.reasoning,
                    }
                    for name, ms in pair.metric_scores.items()
                },
            }
            for pair in pairs
        ]

        target_dir = output_dir or self._dataset_dir(dataset_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in dataset.metadata.name
        )
        json_filename = f"{safe_name}_{dataset_id[-8:]}_minimal.json"
        json_path = target_dir / json_filename

        # Atomic write
        tmp_path = json_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(minimal, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(json_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        logger.info(
            f"DatasetStore: Exported minimal JSON for '{dataset_id}' "
            f"→ '{json_path}' ({len(minimal)} pairs)"
        )

        return json_path

    # ------------------------------------------------------------------
    # INDEX MANAGEMENT
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, DatasetIndexEntry]:
        """
        Load the index from disk into memory.

        Returns empty dict if index does not exist yet.
        Caches the result — invalidated on every write operation.
        """
        if self._index_cache is not None:
            return self._index_cache

        if not self._index_path.exists():
            self._index_cache = {}
            return self._index_cache

        try:
            raw = json.loads(
                self._index_path.read_text(encoding="utf-8")
            )
            self._index_cache = {
                dataset_id: DatasetIndexEntry.from_dict(entry_data)
                for dataset_id, entry_data in raw.items()
            }
            return self._index_cache

        except Exception as exc:
            logger.warning(
                f"DatasetStore: Failed to load index from "
                f"'{self._index_path}': {exc}. "
                f"Rebuilding index from filesystem."
            )
            self._index_cache = self._rebuild_index_from_filesystem()
            return self._index_cache

    def _persist_index(
        self,
        index: dict[str, DatasetIndexEntry],
    ) -> None:
        """Write the index to disk atomically. Invalidates cache."""
        raw = {
            dataset_id: entry.to_dict()
            for dataset_id, entry in index.items()
        }

        tmp_path = self._index_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self._index_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        finally:
            # Always invalidate cache after write attempt
            self._index_cache = None

    def _update_index(self, entry: DatasetIndexEntry) -> None:
        """Add or update a single entry in the index."""
        index = self._load_index()
        index[entry.id] = entry
        self._persist_index(index)

    def _remove_from_index(self, dataset_id: str) -> None:
        """Remove a dataset from the index."""
        index = self._load_index()
        index.pop(dataset_id, None)
        self._persist_index(index)

    def _rebuild_index_from_filesystem(
        self,
    ) -> dict[str, DatasetIndexEntry]:
        """
        Scan the filesystem and rebuild the index from metadata.json files.

        Called when index.json is missing or corrupted.
        Walks all dataset directories and reads their metadata.json.
        """
        index: dict[str, DatasetIndexEntry] = {}

        for dataset_dir in self._base_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if dataset_dir.name == "archive":
                continue

            dataset_path = dataset_dir / self._DATASET_FILE
            if not dataset_path.exists():
                continue

            try:
                dataset = EvalDataset.from_file(dataset_path)
                entry = DatasetIndexEntry.from_eval_dataset(
                    dataset, dataset_dir
                )
                index[dataset.metadata.id] = entry
                logger.debug(
                    f"DatasetStore: Rebuilt index entry for "
                    f"'{dataset.metadata.id}'"
                )
            except Exception as exc:
                logger.warning(
                    f"DatasetStore: Could not read dataset from "
                    f"'{dataset_dir}': {exc}. Skipping."
                )

        logger.info(
            f"DatasetStore: Rebuilt index with {len(index)} datasets."
        )
        self._persist_index(index)
        return index

    def rebuild_index(self) -> int:
        """
        Public method to force index rebuild from filesystem.

        Used by maintenance scripts and the admin UI panel.

        Returns:
            Number of datasets found and indexed.
        """
        self._index_cache = None
        index = self._rebuild_index_from_filesystem()
        return len(index)

    # ------------------------------------------------------------------
    # VERSIONING HELPERS
    # ------------------------------------------------------------------

    def _archive_current_version(
        self,
        dataset_dir: Path,
        metadata: DatasetMetadata,
    ) -> None:
        """
        Copy the current dataset.json to the archive directory
        with the current version as a filename suffix.
        """
        current_path = dataset_dir / self._DATASET_FILE
        if not current_path.exists():
            return

        archive_dir = dataset_dir / self._ARCHIVE_DIR
        archive_dir.mkdir(exist_ok=True)

        archive_filename = f"dataset_v{metadata.version}.json"
        archive_path = archive_dir / archive_filename

        try:
            shutil.copy2(current_path, archive_path)
            logger.debug(
                f"DatasetStore: Archived v{metadata.version} "
                f"of '{metadata.id}' → '{archive_path.name}'"
            )
        except Exception as exc:
            logger.warning(
                f"DatasetStore: Could not archive version "
                f"v{metadata.version} of '{metadata.id}': {exc}. "
                f"Proceeding with save anyway."
            )

    @staticmethod
    def _increment_version(version: str) -> str:
        """
        Increment the patch component of a semantic version string.

        "1.0.0" → "1.0.1"
        "1.0.9" → "1.0.10"
        "2.3.4" → "2.3.5"

        Falls back to appending ".1" for non-standard version strings.
        """
        parts = version.split(".")
        if len(parts) == 3:
            try:
                patch = int(parts[2]) + 1
                return f"{parts[0]}.{parts[1]}.{patch}"
            except ValueError:
                pass
        return f"{version}.1"

    @staticmethod
    def _rebuild_metadata_with_version(
        metadata: DatasetMetadata,
        new_version: str,
    ) -> DatasetMetadata:
        """
        Construct a new DatasetMetadata with an updated version.

        DatasetMetadata is frozen — cannot be mutated.
        We reconstruct it with model_copy(update=...).
        """
        return metadata.model_copy(
            update={
                "version": new_version,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    @staticmethod
    def _touch_updated_at(dataset: EvalDataset) -> EvalDataset:
        """
        Return a new EvalDataset with updated_at set to now.

        DatasetMetadata is frozen, so we reconstruct the full object.
        """
        updated_meta = dataset.metadata.model_copy(
            update={"updated_at": datetime.now(timezone.utc)}
        )
        return EvalDataset(
            metadata=updated_meta,
            pairs=dataset.pairs,
        )

    # ------------------------------------------------------------------
    # FILE HELPERS
    # ------------------------------------------------------------------

    def _dataset_dir(self, dataset_id: str) -> Path:
        """Return the directory path for a given dataset ID."""
        return self._base_dir / dataset_id

    @staticmethod
    def _write_metadata(
        metadata: DatasetMetadata,
        path: Path,
    ) -> None:
        """Write DatasetMetadata to a JSON file atomically."""
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                metadata.model_dump_json(indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def get_store_stats(self) -> dict[str, Any]:
        """
        Return aggregate statistics about the dataset store.

        Used by the admin panel in the Streamlit UI.
        """
        entries = self.list_datasets()

        total_pairs = sum(e.total_pairs for e in entries)
        evaluated_pairs = sum(e.evaluated_pairs for e in entries)

        by_status: dict[str, int] = {}
        by_method: dict[str, int] = {}

        for entry in entries:
            by_status[entry.status] = (
                by_status.get(entry.status, 0) + 1
            )
            by_method[entry.generation_method] = (
                by_method.get(entry.generation_method, 0) + 1
            )

        # Compute total disk usage
        total_bytes = 0
        for dataset_id in [e.id for e in entries]:
            dataset_dir = self._dataset_dir(dataset_id)
            if dataset_dir.exists():
                total_bytes += sum(
                    f.stat().st_size
                    for f in dataset_dir.rglob("*")
                    if f.is_file()
                )

        return {
            "total_datasets":    len(entries),
            "total_pairs":       total_pairs,
            "evaluated_pairs":   evaluated_pairs,
            "by_status":         by_status,
            "by_method":         by_method,
            "total_disk_bytes":  total_bytes,
            "total_disk_mb":     round(total_bytes / (1024 * 1024), 2),
            "base_dir":          str(self._base_dir),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def dataset_count(self) -> int:
        return len(self._load_index())

    def __repr__(self) -> str:
        return (
            f"DatasetStore("
            f"base_dir='{self._base_dir}', "
            f"datasets={self.dataset_count})"
        )

# ===========================================================================
# CUSTOM EXCEPTIONS
# ===========================================================================

class DatasetNotFoundError(Exception):
    """
    Raised when a requested dataset ID does not exist in the store.

    Distinct from DatasetStoreError (I/O or parsing failure).
    DatasetNotFoundError means the ID simply doesn't exist.
    The UI catches this and shows a "Dataset not found" message
    rather than a generic error page.
    """

    def __init__(
        self,
        dataset_id: str,
        detail: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        msg = f"Dataset '{dataset_id}' not found in the store."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class DatasetStoreError(Exception):
    """
    Raised on I/O failures, parse errors, or corrupt dataset files.

    Distinct from DatasetNotFoundError (missing) and
    ValidationError (schema mismatch).
    DatasetStoreError means the file exists but cannot be read,
    written, or parsed.
    """

    def __init__(
        self,
        operation: str,
        dataset_id: str,
        reason: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.operation = operation
        self.dataset_id = dataset_id
        self.reason = reason
        self.original_exception = original_exception
        super().__init__(
            f"[DatasetStore/{operation}] Dataset '{dataset_id}': {reason}"
            + (
                f" (caused by {type(original_exception).__name__}: "
                f"{original_exception})"
                if original_exception
                else ""
            )
        )

__all__ = [
    "DatasetIndexEntry",
    "DatasetStore",
    "DatasetNotFoundError",
    "DatasetStoreError",
]