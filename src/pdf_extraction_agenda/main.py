from os import PathLike
from typing import Literal, NewType, Protocol, assert_never, overload

import pandas as pd

from .datasets_ import parse_response, prepare_olmocr_dataset
from .metrics import calc_nid


class PipelineProto(Protocol):
    def __call__(self, path: str | PathLike) -> str | None:
        """Runs the pipeline on the given path and returns the md result."""
        pass


class PipelineBatchedProto(Protocol):
    def __call__(self, paths: list[str | PathLike], tqdm_pbar) -> list[str | None]:
        """Runs the pipeline on the given paths and returns the md results."""
        pass


EvaluationResult = NewType("EvaluationResult", pd.DataFrame)


@overload
def evaluate_pipeline(run_pipeline: PipelineProto, mode: Literal["single"]) -> EvaluationResult: ...


@overload
def evaluate_pipeline(run_pipeline: PipelineBatchedProto, mode: Literal["all"]) -> EvaluationResult: ...


def evaluate_pipeline(
    run_pipeline: PipelineProto | PipelineBatchedProto, mode: Literal["singe", "all"]
) -> EvaluationResult:
    from tqdm.auto import tqdm

    dataset, id_to_path = prepare_olmocr_dataset()

    metrics_raw = []
    if mode == "all":
        paths = [id_to_path(s["id"], warn=True) for s in dataset]
        with tqdm(total=len(paths), desc="Processing files") as pbar:
            run_pipeline: PipelineBatchedProto
            md_results = run_pipeline(paths, pbar)
    elif mode == "single":
        paths = [id_to_path(s["id"], warn=True) for s in dataset]
        md_results = [run_pipeline(path) for path in paths]
    else:
        assert_never(mode)

    for s, md_result in zip(dataset, md_results):
        malformed, response = parse_response(s, warn=True)
        if malformed:
            continue
        nid = calc_nid(response.natural_text, md_result)
        metrics_raw.append({"id": s["id"], "nid": nid})

    metrics_df = pd.DataFrame(metrics_raw)
    return EvaluationResult(metrics_df)


def main(pipeline: Literal["docling"]):
    if pipeline == "docling":
        from .pipeline_docling import run_docling_pipeline

        run_pipeline = run_docling_pipeline
    else:
        assert_never(pipeline)

    metrics_df = evaluate_pipeline(run_pipeline, mode="single")

    print(metrics_df)


if __name__ == "__main__":
    main("docling")
