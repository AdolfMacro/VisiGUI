import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from visigui.renderer.opengl_world import (
    GenerativeWorldView,
    _FRUSTUM_FADE_DISTANCE,
    _VIEW_FAR,
    _visible_sectors,
)
from visigui.vision.tracking import TwoHandState
from visigui.world.procedural import SectorStreamer
from visigui.world.simulation import GenerativeWorld


def test_view_frustum_culls_offscreen_sectors_but_keeps_ahead_geometry():
    world = GenerativeWorld()
    hands = TwoHandState(None, None, 0.0)
    frame = world.update(hands, 1 / 30)
    for _ in range(40):
        frame = world.update(hands, 1 / 30)

    visible = _visible_sectors(frame, 16 / 9)

    assert 0 < len(visible) < len(frame.sectors)
    assert all(
        any(sector is candidate for candidate in frame.sectors)
        for sector in visible
    )
    assert any(sector.coordinate[2] < 0 for sector in visible)
    assert _VIEW_FAR == 210.0
    assert any(sector.coordinate[2] <= -2 for sector in visible)


def test_frustum_culling_keeps_a_fade_margin_on_all_four_sides():
    world = GenerativeWorld(
        sectors=SectorStreamer(horizontal_radius=0, vertical_radius=0, depth_radius=0),
    )
    frame = world.update(TwoHandState(None, None, 0.0), 0.0)
    source = frame.sectors[0]
    tangent_y = np.tan(np.deg2rad(68.0) * 0.5)
    tangent_x = tangent_y * 16 / 9
    depth = 60.0
    directions = (
        (1.0, 0.0, tangent_x),
        (-1.0, 0.0, tangent_x),
        (0.0, 1.0, tangent_y),
        (0.0, -1.0, tangent_y),
    )

    for horizontal_sign, vertical_sign, tangent in directions:
        for offset, expected_visible in ((8.0, True), (40.0, False)):
            edge_direction = (
                frame.camera.right if horizontal_sign else frame.camera.up
            )
            side = horizontal_sign or vertical_sign
            edge_position = depth * tangent + offset
            center_array = (
                np.asarray(frame.camera.position)
                + np.asarray(frame.camera.forward) * depth
                + np.asarray(edge_direction) * side * edge_position
            )
            center = tuple(map(float, center_array))
            outside_fade = replace(
                source,
                bounds_center=center,
                bounds_half_extents=(0.5, 0.5, 0.5),
            )
            candidate_frame = replace(frame, sectors=(outside_fade,))

            assert bool(_visible_sectors(candidate_frame, 16 / 9)) is expected_visible

    assert _FRUSTUM_FADE_DISTANCE == 24.0
    assert "edge_visibility = smoothstep" in GenerativeWorldView._FRAGMENT_SHADER


def test_streamed_sector_fade_tracks_camera_position_without_rebuilding_geometry():
    world = GenerativeWorld()
    frame = world.update(TwoHandState(None, None, 0.0), 0.0)
    assert frame.sector_radii == (2, 2, 3)
    shader = GenerativeWorldView._VERTEX_SHADER
    assert "u_camera_sector_position" in shader
    assert "axis_visibility(sector_distance.x, u_sector_radii.x)" in shader
    assert "smoothstep" not in shader


def test_renderer_packs_connected_line_colors_and_point_sizes_into_gpu_layout():
    class MemoryBuffer:
        def bind(self):
            return True

        def allocate(self, data, size):
            self.data = data
            self.size = size

        def release(self):
            pass

    app = QApplication.instance() or QApplication([])
    world = GenerativeWorld(
        sectors=SectorStreamer(
            horizontal_radius=0,
            vertical_radius=0,
            depth_radius=0,
        ),
    )
    frame = world.update(TwoHandState(None, None, 0.0), 0.0)
    renderer = GenerativeWorldView()
    renderer.resize(640, 480)
    buffer = MemoryBuffer()
    renderer._buffer = buffer

    renderer._upload_sectors(frame)

    sector = frame.sectors[0]
    packed = np.frombuffer(buffer.data, dtype=np.float32).reshape(-1, 10)
    assert renderer._line_count == len(sector.line_vertices)
    assert renderer._point_count == len(sector.point_vertices)
    assert np.array_equal(packed[0, :6], sector.line_vertices[0])
    assert packed[0, 6] == 1.0
    assert np.array_equal(packed[0, 7:10], sector.coordinate)
    point_offset = renderer._line_count
    assert np.array_equal(packed[point_offset, :6], sector.point_vertices[0, :6])
    assert packed[point_offset, 6] == sector.point_vertices[0, 6]
    assert np.array_equal(packed[point_offset, 7:10], sector.coordinate)
    app.processEvents()


def test_entering_a_new_sector_generates_only_a_small_incremental_line_batch():
    world = GenerativeWorld(
        sectors=SectorStreamer(
            horizontal_radius=1,
            vertical_radius=1,
            depth_radius=1,
            generation_budget=27,
        ),
    )
    hands = TwoHandState(None, None, 0.0)
    world.update(hands, 0.0)
    initial_cache_size = len(world.sectors.cached_coordinates)
    world.camera.position[2] -= world.camera.sector_size
    frame = world.update(hands, 0.0)

    assert initial_cache_size == 27
    assert len(world.sectors.cached_coordinates) == 36
    assert len(frame.sectors) == 27
    assert max(len(sector.line_vertices) for sector in frame.sectors) <= 850
