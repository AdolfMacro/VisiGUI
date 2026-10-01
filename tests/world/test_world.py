import numpy as np
import pytest

from visigui.interaction.world_controller import WorldMotion
from visigui.vision.tracking import TwoHandState
from visigui.world.camera import CameraController
from visigui.world.procedural import SectorGenerator, SectorStreamer
from visigui.world.simulation import GenerativeWorld


def test_sector_generation_is_deterministic_for_same_world_seed_and_coordinates():
    first = SectorGenerator(world_seed=31).generate((2, -1, 5))
    repeated = SectorGenerator(world_seed=31).generate((2, -1, 5))

    assert np.array_equal(first.line_vertices, repeated.line_vertices)
    assert np.array_equal(first.point_vertices, repeated.point_vertices)
    assert first.line_vertices.shape[1] == 6
    assert 500 <= len(first.line_vertices) <= 850
    assert first.point_vertices.shape[1] == 7
    assert 12 <= len(first.point_vertices) <= 24
    assert len(np.unique(first.line_vertices[:, 3:], axis=0)) > 20
    assert not first.line_vertices.flags.writeable
    assert not first.point_vertices.flags.writeable
    lower = np.asarray(first.bounds_center) - first.bounds_half_extents
    upper = np.asarray(first.bounds_center) + first.bounds_half_extents
    assert np.all(first.line_vertices[:, :3] >= lower - 1e-5)
    assert np.all(first.line_vertices[:, :3] <= upper + 1e-5)
    assert np.all(first.point_vertices[:, :3] >= lower - 1e-5)
    assert np.all(first.point_vertices[:, :3] <= upper + 1e-5)


def test_sector_generation_changes_with_sector_or_world_seed():
    generator = SectorGenerator(world_seed=31)
    origin = generator.generate((0, 0, 0))
    neighbor = generator.generate((1, 0, 0))
    other_seed = SectorGenerator(world_seed=32).generate((0, 0, 0))

    assert not np.array_equal(origin.line_vertices, neighbor.line_vertices)
    assert not np.array_equal(origin.point_vertices, other_seed.point_vertices)


@pytest.mark.parametrize("side_count", (3, 6, 8, 9))
def test_polygonal_spiral_preserves_its_endpoints_and_straight_facets(side_count):
    first = np.asarray((-4.0, 0.0, 0.0), dtype=np.float32)
    second = np.asarray((4.0, 0.0, 0.0), dtype=np.float32)
    spiral = SectorGenerator._polygonal_spiral(
        first,
        second,
        side_count,
        np.random.default_rng(side_count),
    )

    assert spiral.shape == (2 * side_count + 3, 3)
    assert spiral[0] == pytest.approx(first)
    assert spiral[-1] == pytest.approx(second)
    assert np.all(np.linalg.norm(np.diff(spiral, axis=0), axis=1) > 0.0)


def test_sector_lines_form_one_connected_network_with_shared_depth_boundaries():
    generator = SectorGenerator(world_seed=103)
    sector = generator.generate((0, 0, 3))
    forward_neighbor = generator.generate((0, 0, 4))
    edges = sector.line_vertices[:, :3].reshape(-1, 2, 3)
    adjacency: dict[tuple[float, float, float], set[tuple[float, float, float]]] = {}
    for start, end in edges:
        start_key = tuple(map(float, start))
        end_key = tuple(map(float, end))
        adjacency.setdefault(start_key, set()).add(end_key)
        adjacency.setdefault(end_key, set()).add(start_key)
    connected = {next(iter(adjacency))}
    pending = list(connected)
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in connected:
                connected.add(neighbor)
                pending.append(neighbor)

    expected_boundary = np.asarray((0.0, 0.0, 3.5 * sector.sector_size), dtype=np.float32)
    sector_boundary = sector.line_vertices[
        np.isclose(sector.line_vertices[:, 2], expected_boundary[2])
    ][:, :3]
    neighbor_boundary = forward_neighbor.line_vertices[
        np.isclose(forward_neighbor.line_vertices[:, 2], expected_boundary[2])
    ][:, :3]

    assert sector.line_vertices.shape[0] % 2 == 0
    assert 250 <= len(sector.line_vertices) // 2 <= 430
    assert len(connected) == len(adjacency)
    assert len(sector_boundary) > 0
    assert len(neighbor_boundary) > 0
    assert np.array_equal(sector_boundary[0], neighbor_boundary[0])
    assert sector.line_vertices[1, :3] == pytest.approx(sector.line_vertices[2, :3])


def test_default_world_streams_far_ahead_and_keeps_sector_window_cached():
    streamer = SectorStreamer()
    camera = CameraController()
    frame = camera.update(WorldMotion(), 0.0)

    sectors = ()
    for _ in range(60):
        sectors = streamer.visible_sectors(frame)

    assert len(sectors) == 175
    assert len(streamer.cached_coordinates) == 175
    assert streamer.generation_budget == 3
    assert any(sector.coordinate[2] == -3 for sector in sectors)


def test_streamer_limits_geometry_generation_work_per_update():
    streamer = SectorStreamer(
        horizontal_radius=1,
        vertical_radius=1,
        depth_radius=1,
        generation_budget=4,
    )
    camera = CameraController()
    snapshot = camera.update(WorldMotion(), 0.0)

    first = streamer.visible_sectors(snapshot)
    second = streamer.visible_sectors(snapshot)

    assert len(first) == 4
    assert len(second) == 8
    assert len(streamer.cached_coordinates) == 8


def test_streamer_reuses_cached_geometry_and_recycles_distant_sectors():
    streamer = SectorStreamer(
        generator=SectorGenerator(world_seed=44),
        cache_size=64,
        horizontal_radius=0,
        vertical_radius=0,
        depth_radius=1,
    )
    camera = CameraController()
    origin = camera.update(WorldMotion(), 0.0)
    first = streamer.visible_sectors(origin)
    repeated = streamer.visible_sectors(origin)

    assert len(first) == 3
    assert first[0] is repeated[0]
    camera.position[2] -= camera.sector_size * 3
    moved = camera.update(WorldMotion(), 0.0)
    next_sectors = streamer.visible_sectors(moved)

    assert next_sectors[0].coordinate != first[0].coordinate
    assert len(streamer.cached_coordinates) <= 64


def test_camera_motion_is_smooth_has_inertia_and_is_frame_rate_independent():
    at_30 = CameraController()
    at_60 = CameraController()
    for _ in range(30):
        snapshot_30 = at_30.update(WorldMotion(strafe=5.0, depth=3.0), 1 / 30)
    for _ in range(60):
        snapshot_60 = at_60.update(WorldMotion(strafe=5.0, depth=3.0), 1 / 60)

    assert snapshot_30.position == pytest.approx(snapshot_60.position, abs=0.08)
    assert snapshot_30.velocity[0] > 0
    assert snapshot_30.position[2] < 26.0

    moving_position = snapshot_60.position
    for _ in range(10):
        stopped = at_60.update(WorldMotion(), 1 / 60)
    assert stopped.position[0] > moving_position[0]
    assert abs(stopped.velocity[0]) < snapshot_60.velocity[0]


def test_default_camera_speed_cap_is_four_times_the_current_tuning():
    previous = CameraController(max_speed=88.0, max_depth_speed=112.0)
    current = CameraController()
    old_snapshot = new_snapshot = None
    for _ in range(180):
        old_snapshot = previous.update(WorldMotion(strafe=88.0, depth=112.0), 1 / 60)
        new_snapshot = current.update(WorldMotion(strafe=352.0, depth=448.0), 1 / 60)

    assert old_snapshot is not None
    assert new_snapshot is not None
    assert new_snapshot.velocity[0] == pytest.approx(old_snapshot.velocity[0] * 4)
    assert new_snapshot.velocity[2] == pytest.approx(old_snapshot.velocity[2] * 4)


def test_world_and_procedural_sectors_persist_through_tracking_loss():
    world = GenerativeWorld(
        seed=91,
        sectors=SectorStreamer(
            generator=SectorGenerator(world_seed=91),
            horizontal_radius=0,
            vertical_radius=0,
            depth_radius=0,
        ),
    )
    tracked = world.update(TwoHandState(None, None, 1.0), 1 / 60)
    after_loss = world.update(TwoHandState(None, None, 1.5), 0.5)

    assert world.seed == 91
    assert after_loss.camera.sector == tracked.camera.sector
    assert after_loss.sectors[0] is tracked.sectors[0]
    assert world.camera.position == list(tracked.camera.position)
