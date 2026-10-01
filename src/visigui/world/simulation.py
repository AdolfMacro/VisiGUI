from __future__ import annotations

from dataclasses import dataclass

from ..interaction.world_controller import TwoHandWorldController, WorldMotion
from ..vision.tracking import TwoHandState
from .camera import CameraController, CameraSnapshot
from .procedural import SectorGeometry, SectorStreamer


@dataclass(frozen=True)
class WorldFrame:
    camera: CameraSnapshot
    hands: TwoHandState
    hands_detected: int
    motion: WorldMotion
    sectors: tuple[SectorGeometry, ...]
    sector_radii: tuple[int, int, int]


class GenerativeWorld:
    """Persistent simulation state shared by tracking and rendering."""

    def __init__(
        self,
        seed: int = 731_921,
        camera: CameraController | None = None,
        interactions: TwoHandWorldController | None = None,
        sectors: SectorStreamer | None = None,
    ) -> None:
        self.camera = camera or CameraController()
        self.interactions = interactions or TwoHandWorldController()
        self.sectors = sectors or SectorStreamer()
        self.seed = seed
        self._snapshot = self.camera.update(WorldMotion(), 0.0)

    def update(
        self,
        hands: TwoHandState,
        delta_time: float,
        additional_motion: WorldMotion = WorldMotion(),
    ) -> WorldFrame:
        motion = self.interactions.update(hands, delta_time)
        motion = WorldMotion(
            motion.strafe + additional_motion.strafe,
            motion.vertical + additional_motion.vertical,
            motion.depth + additional_motion.depth,
        )
        self._snapshot = self.camera.update(motion, delta_time)
        nearby = self.sectors.visible_sectors(self._snapshot)
        return WorldFrame(
            self._snapshot,
            hands,
            hands.hands_detected,
            motion,
            nearby,
            (
                self.sectors.horizontal_radius,
                self.sectors.vertical_radius,
                self.sectors.depth_radius,
            ),
        )

    @property
    def snapshot(self) -> CameraSnapshot:
        return self._snapshot
