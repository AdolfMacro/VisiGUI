from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import math

import numpy as np

from .camera import CameraSnapshot


_PALETTES = np.asarray((
    (
        (0.018, 0.045, 0.16), (0.08, 0.18, 0.48),
        (0.16, 0.72, 0.92), (0.75, 0.95, 0.96),
        (0.48, 0.2, 0.94), (0.95, 0.16, 0.62),
        (1.0, 0.53, 0.18),
    ),
    (
        (0.025, 0.12, 0.075), (0.04, 0.5, 0.23),
        (0.08, 0.85, 0.57), (0.52, 0.96, 0.72),
        (0.04, 0.74, 0.86), (0.22, 0.32, 0.92),
        (0.92, 0.22, 0.52),
    ),
    (
        (0.16, 0.025, 0.12), (0.58, 0.045, 0.25),
        (0.96, 0.18, 0.12), (1.0, 0.52, 0.12),
        (0.98, 0.83, 0.35), (0.12, 0.79, 0.8),
        (0.22, 0.3, 0.94),
    ),
), dtype=np.float32)


@dataclass(frozen=True)
class SectorGeometry:
    coordinate: tuple[int, int, int]
    line_vertices: np.ndarray
    point_vertices: np.ndarray
    sector_size: float
    bounds_center: tuple[float, float, float]
    bounds_half_extents: tuple[float, float, float]


class SectorGenerator:
    """Generate a connected, smooth 3D line network for each world sector."""

    def __init__(self, world_seed: int = 731_921, sector_size: float = 48.0):
        if sector_size <= 0:
            raise ValueError("Sector size must be positive")
        self.world_seed = world_seed
        self.sector_size = sector_size

    def generate(self, coordinate: tuple[int, int, int]) -> SectorGeometry:
        rng = np.random.default_rng(self._sector_seed(coordinate))
        center = np.asarray(coordinate, dtype=np.float32) * self.sector_size
        half = self.sector_size * 0.5
        palette = _PALETTES[int(rng.integers(0, len(_PALETTES)))]
        start = self._depth_boundary(coordinate, positive=True)
        end = self._depth_boundary(coordinate, positive=False)
        anchor_count = int(rng.integers(12, 17))
        anchors = np.empty((anchor_count, 3), dtype=np.float32)
        anchors[0] = start
        anchors[-1] = end
        anchors[1:-1, :2] = rng.uniform(-half * 0.72, half * 0.72, (anchor_count - 2, 2))
        anchors[1:-1, 2] = rng.uniform(-half * 0.78, half * 0.78, anchor_count - 2)
        backbone = self._catmull_rom(anchors, samples_per_edge=12)
        strands = [self._line_vertices(backbone, center, palette, rng)]

        side_counts = np.resize(np.asarray((3, 6, 8, 9)), 13)
        rng.shuffle(side_counts)
        for branch_index, side_count in enumerate(side_counts):
            first_index, second_index = sorted(
                rng.choice(np.arange(2, len(backbone) - 2), size=2, replace=False)
            )
            if second_index - first_index < 5:
                second_index = min(len(backbone) - 2, first_index + 5)
            first = backbone[first_index]
            second = backbone[second_index]
            if branch_index % 5 == 0:
                branch = np.stack((first, second))
            else:
                branch = self._polygonal_spiral(
                    first,
                    second,
                    int(side_count),
                    rng,
                )
            strands.append(self._line_vertices(branch, center, palette, rng))

        line_array = np.concatenate(strands, axis=0)
        point_count = int(rng.integers(12, 25))
        directions = rng.normal(size=(point_count, 3)).astype(np.float32)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
        radii = rng.uniform(17.0, half * 1.12, (point_count, 1)).astype(np.float32)
        positions = center + directions * radii
        point_colors = palette[rng.integers(0, len(palette), point_count)]
        point_colors *= rng.uniform(0.12, 0.35, (point_count, 1))
        sizes = rng.uniform(0.55, 1.1, (point_count, 1)).astype(np.float32)
        point_array = np.concatenate(
            (positions, point_colors, sizes),
            axis=1,
        ).astype(np.float32, copy=False)
        minimum = np.minimum(line_array[:, :3].min(axis=0), point_array[:, :3].min(axis=0))
        maximum = np.maximum(line_array[:, :3].max(axis=0), point_array[:, :3].max(axis=0))
        bounds_center = (minimum + maximum) * 0.5
        bounds_half_extents = (maximum - minimum) * 0.5
        line_array.setflags(write=False)
        point_array.setflags(write=False)
        return SectorGeometry(
            coordinate,
            line_array,
            point_array,
            self.sector_size,
            tuple(map(float, bounds_center)),
            tuple(map(float, bounds_half_extents)),
        )

    def _depth_boundary(
        self,
        coordinate: tuple[int, int, int],
        positive: bool,
    ) -> np.ndarray:
        x, y, z = coordinate
        boundary_index = 2 * z + (1 if positive else -1)
        key = f"{self.world_seed}:depth-boundary:{x}:{y}:{boundary_index}".encode()
        boundary_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        boundary_rng = np.random.default_rng(boundary_seed)
        side = self.sector_size * 0.5 * (1 if positive else -1)
        return np.asarray((
            boundary_rng.uniform(-self.sector_size * 0.28, self.sector_size * 0.28),
            boundary_rng.uniform(-self.sector_size * 0.28, self.sector_size * 0.28),
            side,
        ), dtype=np.float32)

    @staticmethod
    def _catmull_rom(anchors: np.ndarray, samples_per_edge: int) -> np.ndarray:
        paths = []
        for index in range(len(anchors) - 1):
            previous = anchors[max(0, index - 1)]
            first = anchors[index]
            second = anchors[index + 1]
            following = anchors[min(len(anchors) - 1, index + 2)]
            parameter = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False, dtype=np.float32)
            t = parameter[:, None]
            points = 0.5 * (
                2.0 * first
                + (-previous + second) * t
                + (2.0 * previous - 5.0 * first + 4.0 * second - following) * t**2
                + (-previous + 3.0 * first - 3.0 * second + following) * t**3
            )
            paths.append(points)
        return np.concatenate((*paths, anchors[-1:]), axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _polygonal_spiral(
        first: np.ndarray,
        second: np.ndarray,
        side_count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        direction = second - first
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        normal = np.cross(direction, rng.normal(size=3).astype(np.float32))
        normal /= max(float(np.linalg.norm(normal)), 1e-6)
        side = np.cross(normal, direction)
        side /= max(float(np.linalg.norm(side)), 1e-6)
        center = (first + second) * 0.5
        center += rng.uniform(-3.0, 3.0) * normal
        radius = min(
            rng.uniform(6.0, 11.0),
            max(float(np.linalg.norm(second - first)) * 0.42, 3.0),
        )
        turns = 2
        corners = side_count * turns
        phase = rng.uniform(-math.pi, math.pi)
        parameters = np.arange(corners + 1, dtype=np.float32) / corners
        angles = phase + parameters * turns * 2.0 * math.pi
        radii = radius * (1.0 - 0.58 * parameters)
        spiral = (
            center
            + (radii * np.cos(angles))[:, None] * side
            + (radii * np.sin(angles))[:, None] * normal
        )
        return np.concatenate((first[None, :], spiral, second[None, :]), axis=0)

    @staticmethod
    def _line_vertices(
        points: np.ndarray,
        center: np.ndarray,
        palette: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        color_phase = int(rng.integers(0, len(palette)))
        step = int(rng.integers(1, len(palette)))
        indices = (color_phase + np.arange(len(points)) * step // 11) % len(palette)
        brightness = rng.uniform(0.62, 1.05, len(points)).astype(np.float32)
        colors = palette[indices] * brightness[:, None]
        positions = points + center
        vertices = np.empty((2 * (len(points) - 1), 6), dtype=np.float32)
        vertices[0::2, :3] = positions[:-1]
        vertices[1::2, :3] = positions[1:]
        vertices[0::2, 3:] = colors[:-1]
        vertices[1::2, 3:] = colors[1:]
        return vertices

    def _sector_seed(self, coordinate: tuple[int, int, int]) -> int:
        value = f"{self.world_seed}:{coordinate[0]}:{coordinate[1]}:{coordinate[2]}".encode()
        return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little")


class SectorStreamer:
    """Cache and stream nearby deterministic sectors around the camera."""

    def __init__(
        self,
        generator: SectorGenerator | None = None,
        cache_size: int = 192,
        horizontal_radius: int = 2,
        vertical_radius: int = 2,
        depth_radius: int = 3,
        generation_budget: int = 3,
    ) -> None:
        if (
            cache_size <= 0
            or min(horizontal_radius, vertical_radius, depth_radius) < 0
            or generation_budget <= 0
        ):
            raise ValueError("Sector cache and neighborhood sizes are invalid")
        self.generator = generator or SectorGenerator()
        self.cache_size = cache_size
        self.horizontal_radius = horizontal_radius
        self.vertical_radius = vertical_radius
        self.depth_radius = depth_radius
        self.generation_budget = generation_budget
        self._cache: OrderedDict[tuple[int, int, int], SectorGeometry] = OrderedDict()
        self._visible_coordinates: tuple[tuple[int, int, int], ...] = ()
        self._visible_sectors: tuple[SectorGeometry, ...] = ()

    def visible_sectors(self, snapshot: CameraSnapshot) -> tuple[SectorGeometry, ...]:
        center = snapshot.sector
        coordinates = tuple(sorted(
            itertools.product(
                range(center[0] - self.horizontal_radius, center[0] + self.horizontal_radius + 1),
                range(center[1] - self.vertical_radius, center[1] + self.vertical_radius + 1),
                range(center[2] - self.depth_radius, center[2] + self.depth_radius + 1),
            ),
            key=lambda point: sum((point[index] - center[index]) ** 2 for index in range(3)),
        ))
        if (
            coordinates == self._visible_coordinates
            and len(self._visible_sectors) == len(coordinates)
        ):
            return self._visible_sectors
        sectors = []
        generated = 0
        for coordinate in coordinates:
            geometry = self._cache.get(coordinate)
            if geometry is None:
                if generated >= self.generation_budget:
                    continue
                geometry = self.generator.generate(coordinate)
                self._cache[coordinate] = geometry
                generated += 1
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(coordinate)
            sectors.append(geometry)
        self._visible_coordinates = coordinates
        self._visible_sectors = tuple(sectors)
        return self._visible_sectors

    @property
    def cached_coordinates(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(self._cache)
