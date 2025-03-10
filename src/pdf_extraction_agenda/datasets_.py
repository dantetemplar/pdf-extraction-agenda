import glob
import os
import tarfile
from pathlib import Path
from typing import Protocol

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from logging_ import logger
from pydantic import BaseModel, ValidationError


class OlmoOCRResponse(BaseModel):
    """OCRed Page Information"""

    primary_language: str
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool
    natural_text: str  # Extracted text from PDF


def parse_response(example: dict, warn: bool = True) -> tuple[bool, OlmoOCRResponse | None]:
    try:
        return False, OlmoOCRResponse.model_validate_json(example["response"])
    except ValidationError as e:
        if warn:
            logger.warning(f"Malformed response for {example.get('id')}\n{e}")
        return True, None


def extract_tarballs(source_dir: str | os.PathLike, destination_dir: str | os.PathLike) -> None:
    """Extracts all tarball files from the source directory into the destination directory."""
    os.makedirs(destination_dir, exist_ok=True)

    tarballs = glob.glob(os.path.join(source_dir, "*.tar*"))  # Matches .tar, .tar.gz, .tar.bz2, etc.
    for tarball in tarballs:
        try:
            with tarfile.open(tarball, "r:*") as tar:
                tar.extractall(path=destination_dir, filter="fully_trusted")
        except Exception as e:
            logger.info(f"Failed to extract {tarball}: {e}")


class IdToPathProto(Protocol):
    def __call__(self, id: str, warn: bool = False) -> Path | None:
        """Converts an ID to a file path."""
        pass


def prepare_olmocr_dataset() -> tuple[Dataset, IdToPathProto]:
    dataset = load_dataset("allenai/olmOCR-mix-0225", "00_documents", split="eval_s2pdf")
    path_to_snaphot = snapshot_download(
        repo_id="dantetemplar/pdf-extraction-agenda", repo_type="dataset", allow_patterns=["*.tar.gz"]
    )
    source_tarball_dir = os.path.join(path_to_snaphot, "data", "olmOCR-mix-0225")
    destination_dir = Path("data/olmOCR-mix-0225-extracted")

    extract_tarballs(source_tarball_dir, destination_dir)

    def id_to_path(id: str, warn: bool = False) -> Path | None:
        path = destination_dir / f"{id}.pdf"
        if path.exists():
            return path
        else:
            if warn:
                logger.warning(f"File {path} not found")
            return None

    return dataset, id_to_path


def main():
    dataset, id_to_path = prepare_olmocr_dataset()

    for s in dataset:
        path = id_to_path(s["id"], warn=True)
        malformed, response = parse_response(s, warn=True)
        if malformed:
            continue


if __name__ == "__main__":
    main()
