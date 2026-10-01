from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMatrix4x4, QVector3D
from PyQt6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from ..world.procedural import SectorGeometry
from ..world.simulation import WorldFrame


_VIEW_NEAR = 0.08
_VIEW_FAR = 210.0
_FOG_START = 130.0
_FOG_END = 205.0
_FRUSTUM_FADE_DISTANCE = 24.0


def _visible_sectors(
    frame: WorldFrame,
    aspect_ratio: float,
) -> tuple[SectorGeometry, ...]:
    camera = frame.camera
    half_fov_tangent = math.tan(math.radians(68.0) * 0.5)
    tangent_x = half_fov_tangent * aspect_ratio
    forward = camera.forward
    right = camera.right
    up = camera.up
    visible = []
    for sector in frame.sectors:
        offset = tuple(
            sector.bounds_center[index] - camera.position[index]
            for index in range(3)
        )
        depth = sum(offset[index] * forward[index] for index in range(3))
        horizontal = sum(offset[index] * right[index] for index in range(3))
        vertical = sum(offset[index] * up[index] for index in range(3))
        depth_radius = sum(
            sector.bounds_half_extents[index] * abs(forward[index])
            for index in range(3)
        )
        horizontal_radius = sum(
            sector.bounds_half_extents[index] * abs(right[index])
            for index in range(3)
        )
        vertical_radius = sum(
            sector.bounds_half_extents[index] * abs(up[index])
            for index in range(3)
        )
        if (
            depth + depth_radius < _VIEW_NEAR - _FRUSTUM_FADE_DISTANCE
            or depth - depth_radius > _VIEW_FAR + _FRUSTUM_FADE_DISTANCE
        ):
            continue
        if (
            abs(horizontal)
            > depth * tangent_x
            + horizontal_radius
            + depth_radius * tangent_x
            + _FRUSTUM_FADE_DISTANCE * math.sqrt(1.0 + tangent_x**2)
        ):
            continue
        if (
            abs(vertical)
            > depth * half_fov_tangent
            + vertical_radius
            + depth_radius * half_fov_tangent
            + _FRUSTUM_FADE_DISTANCE * math.sqrt(1.0 + half_fov_tangent**2)
        ):
            continue
        visible.append(sector)
    return tuple(visible)


class GenerativeWorldView(QOpenGLWidget):
    """Batched OpenGL renderer for streamed, deterministic 3D sectors."""

    renderer_error = pyqtSignal(str)

    _VERTEX_SHADER = """
        #version 330 core
        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec3 a_color;
        layout(location = 2) in float a_size;
        layout(location = 3) in vec3 a_sector_coordinate;
        uniform mat4 u_mvp;
        uniform mat4 u_view;
        uniform float u_point_scale;
        uniform vec3 u_camera_sector_position;
        uniform vec3 u_sector_radii;
        out vec3 v_color;
        out float v_view_distance;
        out vec3 v_view_position;
        out float v_sector_alpha;
        float axis_visibility(float distance_from_center, float radius) {
            if (radius <= 1.0) return 1.0;
            float fade = clamp(distance_from_center - (radius - 1.0), 0.0, 1.0);
            fade = fade * fade * (3.0 - 2.0 * fade);
            return 1.0 - fade;
        }
        void main() {
            vec4 clip = u_mvp * vec4(a_position, 1.0);
            gl_Position = clip;
            gl_PointSize = clamp(a_size * u_point_scale / max(1.0, clip.w), 1.0, 8.0);
            v_color = a_color;
            v_view_distance = clip.w;
            v_view_position = (u_view * vec4(a_position, 1.0)).xyz;
            vec3 sector_distance = abs(a_sector_coordinate - u_camera_sector_position);
            v_sector_alpha = axis_visibility(sector_distance.x, u_sector_radii.x)
                * axis_visibility(sector_distance.y, u_sector_radii.y)
                * axis_visibility(sector_distance.z, u_sector_radii.z);
        }
    """
    _FRAGMENT_SHADER = """
        #version 330 core
        in vec3 v_color;
        in float v_view_distance;
        in vec3 v_view_position;
        in float v_sector_alpha;
        uniform float u_alpha;
        uniform float u_fog_start;
        uniform float u_fog_end;
        uniform float u_tangent_x;
        uniform float u_tangent_y;
        uniform float u_near;
        uniform float u_far;
        uniform float u_frustum_fade_distance;
        uniform int u_round_points;
        out vec4 fragment_color;
        void main() {
            if (u_round_points != 0) {
                vec2 centered = gl_PointCoord - vec2(0.5);
                if (dot(centered, centered) > 0.25) discard;
            }
            float visibility = 1.0 - smoothstep(u_fog_start, u_fog_end, v_view_distance);
            float depth = -v_view_position.z;
            float horizontal_edge = (depth * u_tangent_x - abs(v_view_position.x))
                / sqrt(1.0 + u_tangent_x * u_tangent_x);
            float vertical_edge = (depth * u_tangent_y - abs(v_view_position.y))
                / sqrt(1.0 + u_tangent_y * u_tangent_y);
            float edge_distance = min(
                min(horizontal_edge, vertical_edge),
                min(depth - u_near, u_far - depth)
            );
            float edge_visibility = smoothstep(
                0.0,
                u_frustum_fade_distance,
                edge_distance
            );
            fragment_color = vec4(
                v_color,
                u_alpha * visibility * edge_visibility * v_sector_alpha
            );
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.PartialUpdate)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._program: Optional[QOpenGLShaderProgram] = None
        self._buffer: Optional[QOpenGLBuffer] = None
        self._vertex_array: Optional[QOpenGLVertexArrayObject] = None
        self._line_count = 0
        self._point_count = 0
        self._frame: Optional[WorldFrame] = None
        self._uploaded_sector_key: tuple[tuple[int, int, int], ...] = ()
        self._ready = False
        self._reported_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._ready

    def set_world_frame(self, frame: WorldFrame) -> None:
        self._frame = frame
        self.update()

    def initializeGL(self) -> None:
        try:
            from OpenGL import GL

            self._program = QOpenGLShaderProgram(self)
            if not self._program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex,
                self._VERTEX_SHADER,
            ):
                raise RuntimeError(self._program.log())
            if not self._program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment,
                self._FRAGMENT_SHADER,
            ):
                raise RuntimeError(self._program.log())
            if not self._program.link():
                raise RuntimeError(self._program.log())
            self._buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            if not self._buffer.create():
                raise RuntimeError("Could not create the OpenGL world vertex buffer")
            self._vertex_array = QOpenGLVertexArrayObject(self)
            if not self._vertex_array.create():
                raise RuntimeError("Could not create the OpenGL world vertex array")
            if not self._program.bind():
                raise RuntimeError("Could not bind the OpenGL world shader during setup")
            self._vertex_array.bind()
            if int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING)) != self._vertex_array.objectId():
                self._program.release()
                raise RuntimeError("Could not bind the OpenGL vertex array during setup")
            if not self._buffer.bind():
                self._vertex_array.release()
                self._program.release()
                raise RuntimeError("Could not bind the OpenGL vertex buffer during setup")
            try:
                self._bind_vertex_layout()
            finally:
                self._buffer.release()
                self._vertex_array.release()
                self._program.release()
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
            self._ready = True
        except Exception as error:
            self._report_error(f"{type(error).__name__}: {error}")

    def resizeGL(self, width: int, height: int) -> None:
        if self._ready:
            from OpenGL import GL

            GL.glViewport(0, 0, max(1, width), max(1, height))

    def paintGL(self) -> None:
        try:
            from OpenGL import GL
        except ImportError as error:
            self._report_error(f"OpenGL dependency unavailable: {error}")
            return

        GL.glClearColor(0.003, 0.004, 0.009, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self._ready or self._program is None or self._buffer is None:
            return
        if self._frame is None:
            return
        try:
            self._upload_sectors(self._frame)
            if self._line_count + self._point_count == 0:
                return
            if self._vertex_array is None:
                raise RuntimeError("OpenGL world vertex array is not initialized")
            if not self._program.bind():
                raise RuntimeError("Could not bind the OpenGL world shader program")
            vertex_array_bound = False
            try:
                self._vertex_array.bind()
                vertex_array_bound = True
                projection = QMatrix4x4()
                projection.perspective(
                    68.0,
                    max(1, self.width()) / max(1, self.height()),
                    _VIEW_NEAR,
                    _VIEW_FAR,
                )
                camera = self._frame.camera
                position = QVector3D(*camera.position)
                forward = QVector3D(*camera.forward)
                up = QVector3D(*camera.up)
                view = QMatrix4x4()
                view.lookAt(position, position + forward, up)
                self._program.setUniformValue("u_mvp", projection * view)
                self._program.setUniformValue("u_view", view)
                self._program.setUniformValue("u_point_scale", float(self.height()) * 0.72)
                sector_size = self._frame.sectors[0].sector_size
                camera_sector_position = tuple(
                    camera.position[index] / sector_size
                    for index in range(3)
                )
                self._program.setUniformValue(
                    "u_camera_sector_position",
                    QVector3D(*camera_sector_position),
                )
                self._program.setUniformValue(
                    "u_sector_radii",
                    QVector3D(*map(float, self._frame.sector_radii)),
                )
                self._program.setUniformValue("u_fog_start", _FOG_START)
                self._program.setUniformValue("u_fog_end", _FOG_END)
                tangent_y = math.tan(math.radians(68.0) * 0.5)
                tangent_x = tangent_y * max(1, self.width()) / max(1, self.height())
                self._program.setUniformValue("u_tangent_x", tangent_x)
                self._program.setUniformValue("u_tangent_y", tangent_y)
                self._program.setUniformValue("u_near", _VIEW_NEAR)
                self._program.setUniformValue("u_far", _VIEW_FAR)
                self._program.setUniformValue(
                    "u_frustum_fade_distance",
                    _FRUSTUM_FADE_DISTANCE,
                )
                self._program.setUniformValue("u_round_points", 0)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                self._program.setUniformValue("u_alpha", 0.82)
                GL.glDrawArrays(GL.GL_LINES, 0, self._line_count)
                self._program.setUniformValue("u_alpha", 0.46)
                self._program.setUniformValue("u_round_points", 1)
                GL.glDrawArrays(
                    GL.GL_POINTS,
                    self._line_count,
                    self._point_count,
                )
            finally:
                if vertex_array_bound:
                    self._vertex_array.release()
                self._program.release()
        except Exception as error:
            self._report_error(f"{type(error).__name__}: {error}")

    def _upload_sectors(self, frame: WorldFrame) -> None:
        if self._buffer is None:
            return
        visible_sectors = _visible_sectors(
            frame,
            max(1, self.width()) / max(1, self.height()),
        )
        sector_key = tuple(sector.coordinate for sector in visible_sectors)
        if sector_key == self._uploaded_sector_key:
            return
        self._line_count = sum(len(sector.line_vertices) for sector in visible_sectors)
        self._point_count = sum(len(sector.point_vertices) for sector in visible_sectors)
        vertices = np.empty(
            (self._line_count + self._point_count, 10),
            dtype=np.float32,
        )
        line_offset = 0
        point_offset = self._line_count
        for sector in visible_sectors:
            line_count = len(sector.line_vertices)
            line_batch = vertices[line_offset:line_offset + line_count]
            line_batch[:, :6] = sector.line_vertices
            line_batch[:, 6] = 1.0
            line_batch[:, 7:10] = sector.coordinate
            line_offset += line_count

            point_count = len(sector.point_vertices)
            point_batch = vertices[point_offset:point_offset + point_count]
            point_batch[:, :6] = sector.point_vertices[:, :6]
            point_batch[:, 6] = sector.point_vertices[:, 6]
            point_batch[:, 7:10] = sector.coordinate
            point_offset += point_count
        if len(vertices) == 0:
            self._uploaded_sector_key = sector_key
            return
        if not self._buffer.bind():
            raise RuntimeError("Could not bind the OpenGL world vertex buffer for upload")
        try:
            self._buffer.allocate(vertices.tobytes(), vertices.nbytes)
        finally:
            self._buffer.release()
        self._uploaded_sector_key = sector_key

    def _bind_vertex_layout(self) -> None:
        if self._program is None:
            return
        stride = 10 * np.dtype(np.float32).itemsize
        for index, size, offset in (
            (0, 3, 0),
            (1, 3, 3 * 4),
            (2, 1, 6 * 4),
            (3, 3, 7 * 4),
        ):
            self._program.enableAttributeArray(index)
            self._program.setAttributeBuffer(index, 0x1406, offset, size, stride)

    def _report_error(self, message: str) -> None:
        if message != self._reported_error:
            self._reported_error = message
            self.renderer_error.emit(message)

    def cleanup(self) -> None:
        resources_exist = any((
            self._vertex_array is not None,
            self._buffer is not None,
            self._program is not None,
        ))
        if resources_exist and self.context() is not None:
            self.makeCurrent()
            if self._vertex_array is not None:
                self._vertex_array.destroy()
            if self._buffer is not None:
                self._buffer.destroy()
            if self._program is not None:
                self._program.removeAllShaders()
            self.doneCurrent()
        self._vertex_array = None
        self._buffer = None
        self._program = None
        self._ready = False
