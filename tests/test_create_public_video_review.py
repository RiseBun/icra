import json

import pandas as pd

from scripts.create_public_video_review import main


def test_review_manifest_separates_proxy_labels(tmp_path, monkeypatch):
    audit = tmp_path / "audit.json"
    episodes = tmp_path / "episodes.parquet"
    out = tmp_path / "review"
    eid = str(audit) + "#episode_0"
    audit.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "episode_id": eid,
                        "window_start": 0,
                        "skill": "push",
                        "label": 0,
                        "progress": 0.1,
                        "label_source": "progress_proxy",
                    },
                    {
                        "episode_id": eid,
                        "window_start": 8,
                        "skill": "push",
                        "label": 1,
                        "progress": 0.9,
                        "label_source": "progress_proxy",
                    },
                ]
            }
        )
    )
    pd.DataFrame(
        {
            "episode_index": [0],
            "videos/observation.images.image_0/from_timestamp": [0.0],
            "videos/observation.images.image_0/to_timestamp": [2.0],
        }
    ).to_parquet(episodes)
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_public_video_review.py",
            "--audit",
            str(audit),
            "--episodes",
            str(episodes),
            "--output-dir",
            str(out),
            "--per-task",
            "1",
        ],
    )
    main()
    blind = json.loads((out / "blind_manifest.json").read_text())
    revealed = json.loads((out / "revealed_proxy_labels.json").read_text())
    assert "proxy_label" not in blind["items"][0]
    assert revealed["items"][0]["proxy_label"] == 1
    assert revealed["items"][0]["progress"] == 0.9
