import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.create_figure_qa_benchmark import (
    Paper,
    extract_candidates,
    normalize_id,
    render_image,
    select_diverse,
    write_splits,
)


class GeneratorTests(unittest.TestCase):
    def test_normalize_id(self):
        value = "https://arxiv.org/pdf/1234.56789v2.pdf"
        self.assertEqual(normalize_id(value), "1234.56789")

    def test_extracts_figure_with_discussion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "plot.png").write_bytes(b"image")
            (source / "main.tex").write_text(
                r"""\begin{figure}
\includegraphics{plot}
\caption{Measured trend.}
\label{fig:trend}
\end{figure}

The response rises because defects become mobile, as shown in \ref{fig:trend}.
"""
            )
            paper = Paper("1234.56789", "Title", "Abstract", ["cond-mat.mtrl-sci"], "2026", "url")
            records = extract_candidates(paper, source)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["caption"], "Measured trend.")
            self.assertIn("defects become mobile", records[0]["discussion"])

    def test_rejects_extreme_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wide.png"
            Image.new("RGB", (500, 100)).save(source)
            self.assertIsNone(render_image(source, root / "rendered"))

    def test_downsamples_longest_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "large.png"
            Image.new("RGB", (3000, 1500)).save(source)
            output = render_image(source, root / "rendered")
            with Image.open(output) as image:
                self.assertEqual(max(image.size), 2048)

    def test_diverse_selection_round_robins_papers(self):
        rows = [
            {"candidate_id": "a1", "paper_id": "a", "published": "2"},
            {"candidate_id": "a2", "paper_id": "a", "published": "1"},
            {"candidate_id": "b1", "paper_id": "b", "published": "2"},
        ]
        selected = select_diverse(rows, 2)
        self.assertEqual({row["paper_id"] for row in selected}, {"a", "b"})

    def test_splits_are_paper_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"candidate_id": "a1", "paper_id": "a"},
                {"candidate_id": "a2", "paper_id": "a"},
                {"candidate_id": "b1", "paper_id": "b"},
            ]
            write_splits(rows, Path(directory), 0.5)
            validation = [json.loads(line) for line in (Path(directory) / "validation.jsonl").read_text().splitlines()]
            test = [json.loads(line) for line in (Path(directory) / "test.jsonl").read_text().splitlines()]
            self.assertTrue({row["paper_id"] for row in validation}.isdisjoint(
                {row["paper_id"] for row in test}
            ))


if __name__ == "__main__":
    unittest.main()
