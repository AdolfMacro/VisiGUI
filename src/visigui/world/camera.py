from __future__ import annotations

from dataclasses import dataclass
import math

from ..interaction.world_controller import WorldMotion


@dataclass(frozen=True)
class CameraSnapshot:
    position: tuple[float, float, float]
    yaw: float
    pitch: float
    velocity: tuple[float, float, float]
    sector: tuple[int, int, int]

    @property
    def forward(self) -> tuple[float, float, float]:
        cos_pitch = math.cos(self.pitch)
        return (
            -math.sin(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cos_pitch,
        )

    @property
    def right(self) -> tuple[float, float, float]:
        return (math.cos(self.yaw), 0.0, -math.sin(self.yaw))

    @property
    def up(self) -> tuple[float, float, float]:
        forward = self.forward
        right = self.right
        return (
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        )


class CameraController:
    """Frame-rate-independent free-flight camera with accelerated hand controls."""

    def __init__(
        self,
        sector_size: float = 48.0,
        acceleration: float = 22.0,
        damping: float = 6.8,
        max_speed: float = 352.0,
        max_depth_speed: float = 448.0,
    ) -> None:
        if min(sector_size, acceleration, damping, max_speed, max_depth_speed) <= 0:
            raise ValueError("Camera physics values must be positive")
        self.sector_size = sector_size
        self.acceleration = acceleration
        self.damping = damping
        self.max_speed = max_speed
        self.max_depth_speed = max_depth_speed
        self.position = [0.0, 0.0, 26.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, motion: WorldMotion, delta_time: float) -> CameraSnapshot:
        remaining = min(max(delta_time, 0.0), 0.25)
        while remaining > 0:
            dt = min(remaining, 1 / 120)
            remaining -= dt
            forward = self._forward()
            right = (math.cos(self.yaw), 0.0, -math.sin(self.yaw))
            up = self._up(forward, right)
            target = tuple(
                right[index] * motion.strafe
                + up[index] * motion.vertical
                + forward[index] * motion.depth
                for index in range(3)
            )
            horizontal_scale = min(
                1.0,
                self.max_speed / max(math.hypot(target[0], target[1]), 1e-9),
            )
            target = (
                target[0] * horizontal_scale,
                target[1] * horizontal_scale,
                max(-self.max_depth_speed, min(self.max_depth_speed, target[2])),
            )
            rate = self.acceleration if any(abs(value) > 1e-6 for value in target) else self.damping
            blend = 1.0 - math.exp(-rate * dt)
            for axis in range(3):
                self.velocity[axis] += (target[axis] - self.velocity[axis]) * blend
                self.position[axis] += self.velocity[axis] * dt
        sector = tuple(math.floor(component / self.sector_size) for component in self.position)
        return CameraSnapshot(
            tuple(self.position),
            self.yaw,
            self.pitch,
            tuple(self.velocity),
            sector,
        )

    def rotate(self, yaw_delta: float, pitch_delta: float) -> None:
        self.yaw = (self.yaw + yaw_delta + math.pi) % (2 * math.pi) - math.pi
        self.pitch = max(-1.35, min(1.35, self.pitch + pitch_delta))

    def _forward(self) -> tuple[float, float, float]:
        cos_pitch = math.cos(self.pitch)
        return (
            -math.sin(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cos_pitch,
        )

    @staticmethod
    def _up(
        forward: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        )
