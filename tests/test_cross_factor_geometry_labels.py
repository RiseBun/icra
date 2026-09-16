from scripts.collect_cross_factor import classify_ring_geometry


def base(**updates):
    value = {
        "ring_z": 0.884,
        "pillar_z": 0.832,
        "nearest_xy_distance": 0.005,
        "nearest_pillar_index": 1,
        "target_pillar_index": 1,
        "ring_grasped": False,
    }
    value.update(updates)
    return value


def test_geometry_labels_do_not_depend_on_success_centre():
    assert classify_ring_geometry(base()) == "success"
    assert classify_ring_geometry(base(nearest_pillar_index=2)) == "wrong_target"
    assert classify_ring_geometry(base(ring_z=0.777)) == "success"
    assert classify_ring_geometry(base(ring_z=0.74)) == "success"
    assert classify_ring_geometry(base(ring_z=0.777, ring_grasped=True)) == "drop"


def test_far_ring_is_drop_even_when_high():
    assert classify_ring_geometry(base(nearest_xy_distance=0.08)) == "drop"
