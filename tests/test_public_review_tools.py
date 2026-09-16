import csv
import json

from scripts.make_public_review_sheet import main as make_main
from scripts.score_public_video_review import main as score_main


def test_review_sheet_and_scoring(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    revealed = tmp_path / "revealed.json"
    sheet = tmp_path / "reviews.csv"
    scored = tmp_path / "score.json"
    manifest.write_text(json.dumps({"items": [{"review_id": "review_0000", "task": "push", "episode_index": 0, "timestamp": 1.0}]}))
    revealed.write_text(json.dumps({"items": [{"review_id": "review_0000", "proxy_label": 1}]}))
    monkeypatch.setattr("sys.argv", ["make", "--manifest", str(manifest), "--output", str(sheet)])
    make_main()
    rows = list(csv.DictReader(sheet.open()))
    for row in rows:
        row["human_label"] = "1"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr("sys.argv", ["score", "--reviews", str(sheet), "--revealed", str(revealed), "--output", str(scored)])
    score_main()
    result = json.loads(scored.read_text())
    assert result["proxy_vs_human_accuracy"] == 1.0
    assert result["proxy_vs_human_balanced_accuracy"] is None
    assert result["proxy_vs_human_auroc"] is None
    assert result["n_consensus_items"] == 1
    assert result["n_disagreement_items"] == 0
    assert result["proxy_metrics"]["confusion_matrix_human_rows_proxy_cols"] == [[0, 0], [0, 1]]
    assert result["per_rater"]["rater_1"]["confusion_matrix_human_rows_proxy_cols"] == [[0, 0], [0, 1]]
