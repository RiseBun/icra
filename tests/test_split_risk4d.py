import numpy as np

from scripts.split_risk4d_episodes import stratified_partitions, variation_flags


def _episode(root, index, success, collision):
    episode = root / f"episode_{index:04d}"
    episode.mkdir()
    np.savez_compressed(
        episode / "decision_0000.npz",
        success=np.asarray(success, np.float32),
        collision=np.asarray(collision, np.float32),
    )
    return episode


def test_stratified_split_keeps_collision_variation_in_train_and_test(tmp_path):
    episodes = [_episode(tmp_path, i, [0, 1] if i in (0, 1, 2, 3) else [1],
                         [0, 1] if i in (0, 1, 2) else [0])
                for i in range(10)]
    partitions = stratified_partitions(episodes, np.random.default_rng(7))
    assert len(partitions["train"]) == 7
    assert len(partitions["test"]) == 1
    assert any(variation_flags(ep)[1] for ep in partitions["train"])
    assert any(variation_flags(ep)[1] for ep in partitions["test"])
    all_names = [ep.name for split in partitions.values() for ep in split]
    assert len(all_names) == len(set(all_names)) == 10
