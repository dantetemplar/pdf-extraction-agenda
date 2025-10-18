import re
import warnings
from pathlib import Path

import pandas as pd
from py_markdown_table.markdown_table import markdown_table

NOT_PIPELINES = ["↓ Specialized VLMs", "↓ Proprietary pipelines", "↓ General VLMs"]
RESULTS_REPORTED_BY_AUTHORS = {
    "OmniDocBench Overall ↓": ["MinerU", "MonkeyOCR", "dots.ocr"],
    "OmniDocBench New ↑": ["MinerU"],
    "olmOCR Overall ↑": ["olmOCR", "MonkeyOCR", "dots.ocr"],
    "dp-bench NID ↑": ["upstage"],
}
CONVERT_COLUMNS = {
    "OmniDocBench Overall ↓": " [OmniDocBench](#omnidocbench) Overall ↓ ",
    "OmniDocBench New ↑": " [OmniDocBench](#omnidocbench) New ↑ ",
    "olmOCR Overall ↑": " [olmOCR](#olmoocr-eval) Overall ↑ ",
    "dp-bench NID ↑": " [dp-bench](#dp-bench) NID ↑ ",
}

def main(path_to_benches: Path) -> None:
    df = pd.read_excel(path_to_benches)
    print(df)
    print("-" * 100, "\n\n")

    # replace nans with empty strings
    df = df.fillna("")

    # for each column get top 3 pipelines
    for column in df.columns:
        maximum_or_minimum = "maximum" if column.endswith("↑") else "minimum" if column.endswith("↓") else None
        if maximum_or_minimum is None:
            warnings.warn(f"Column {column} does not end with ↑ or ↓")
            continue

        # convert to floats
        copy = df.copy()
        copy = copy[copy[column] != ""]
        copy[column] = copy[column].astype(float)
        top_3 = copy.sort_values(by=column, ascending=maximum_or_minimum == "minimum").head(3)
        print(f"Top 3 pipelines for '{column}':")
        print(top_3[["Pipeline", column]])
        print("----")

        # add <sup>[1]</sup>, <sup>[2]</sup>, <sup>[3]</sup> to top3
        for i in range(3):
            df.loc[df["Pipeline"] == top_3["Pipeline"].iloc[i], column] = f"{top_3[column].iloc[i]} <sup>[{i+1}]</sup>"

        # mark first one bold in original dataframe
        df.loc[df["Pipeline"] == top_3["Pipeline"].iloc[0], column] = (
            f"**{df.loc[df['Pipeline'] == top_3['Pipeline'].iloc[0], column].iloc[0]}**"
        )

        # add "⚠️" to pipeline if it was reported by the authors
        for pipeline in top_3["Pipeline"]:
            if pipeline in RESULTS_REPORTED_BY_AUTHORS[column]:
                df.loc[df["Pipeline"] == pipeline, column] = f"{df.loc[df['Pipeline'] == pipeline, column].iloc[0]} ⚠️"

    # wrap NOT PIPELINES with "_"
    for pipeline in NOT_PIPELINES:
        df.loc[df["Pipeline"] == pipeline, "Pipeline"] = f"_{pipeline}_"

    # open README.md and get all sections from it in such format [MinerU](https://github.com/opendatalab/MinerU)
    with open("README.md", "r") as f:
        readme = f.read()
    sections = re.finditer(r"## \[(?P<pipeline_name>.*)\]\(.*\)", readme)
    sections = [section.group("pipeline_name") for section in sections]
    print(sections)

    # for each pipeline, try to find it in sections
    for pipeline in df["Pipeline"]:
        if pipeline.startswith("_"):
            continue
        if pipeline in sections: # if found, replace Pipeline with [Pipeline](#escape(Pipeline))
            df.loc[df["Pipeline"] == pipeline, "Pipeline"] = f"[{pipeline}](#{(pipeline.replace(" ", "-").replace(".", ""))})"
        else:
            warnings.warn(f"Pipeline {pipeline} not found in README.md")
    
    # rename columns
    df.rename(columns=CONVERT_COLUMNS, inplace=True)

    as_dicts = df.to_dict(orient="records")
    md = (
        markdown_table(as_dicts)
        .set_params(
            row_sep="markdown",
            quote=False,
            padding_width=4,
            padding_weight="centerright",
        )
        .get_markdown()
    )
    print(md)

    # find <benches>...</benches> and replace inner content with md, handling multiline
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()
    readme_new = re.sub(
        r"<benches>.*?</benches>",
        f"<benches>\n\n{md}\n</benches>",
        readme,
        flags=re.DOTALL
    )
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_new)


if __name__ == "__main__":
    main(Path("benches.xlsx"))
