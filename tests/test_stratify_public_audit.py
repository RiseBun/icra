import json

from scripts.stratify_public_audit import main


def test_stratification_uses_episode_bootstrap(tmp_path, monkeypatch):
    audit = tmp_path / "audit.json"
    output = tmp_path / "stratified.json"
    rows = []
    for task in ("a", "b"):
        for episode in range(4):
            for window in range(2):
                rows.append(
                    {
                        "episode_id": f"{task}-{episode}",
                        "skill": task,
                        "label": int((episode + window) % 2),
                        "mean_amp": float(episode + window),
                    }
                )
    audit.write_text(json.dumps({"rows": rows}))
    monkeypatch.setattr(
        "sys.argv",
        ["stratify_public_audit.py", "--audit", str(audit), "--output", str(output), "--bootstrap", "20"],
    )
    main()
    data = json.loads(output.read_text())
    assert data["n_groups"] == 2
    assert data["n_groups_with_label_variation"] == 2
    assert all("amplitude_auroc_ci95" in v for v in data["groups"].values())
