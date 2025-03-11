from os import PathLike
from typing import Literal, NewType, Protocol, assert_never

import pandas as pd
from tqdm import tqdm

from .datasets_ import parse_response, prepare_olmocr_dataset
from .metrics import calc_nid


class PipelineProto(Protocol):
    def __call__(self, path: str | PathLike) -> str:
        """Runs the pipeline on the given path and returns the md result."""
        pass


EvaluationResult = NewType("EvaluationResult", pd.DataFrame)


def evaluate_pipeline(run_pipeline: PipelineProto) -> EvaluationResult:
    dataset, id_to_path = prepare_olmocr_dataset()

    metrics_raw = []

    for s in tqdm(dataset):
        path = id_to_path(s["id"], warn=True)
        malformed, response = parse_response(s, warn=True)
        if malformed:
            continue

        md_result = run_pipeline(path)
        nid = calc_nid(response.natural_text, md_result)
        metrics_raw.append({"nid": nid})

    metrics_df = pd.DataFrame(metrics_raw)
    return EvaluationResult(metrics_df)


def main(pipeline: Literal["docling"]):
    if pipeline == "docling":
        from .pipeline_docling import run_docling_pipeline

        run_pipeline = run_docling_pipeline
    else:
        assert_never(pipeline)

    metrics_df = evaluate_pipeline(run_pipeline)

    print(metrics_df)


if __name__ == "__main__":
    main("docling")
