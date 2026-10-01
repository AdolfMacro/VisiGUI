import numpy as np

from visigui.core.contracts import CameraFrame, HandObservation
from visigui.vision.camera_preview import CameraPreview


def _hand():
    return tuple(
        (0.25 + (index % 4) * 0.05, 0.2 + (index // 4) * 0.1, 0.0)
        for index in range(21)
    )


class FakeCV2:
    WINDOW_NORMAL = 0
    FONT_HERSHEY_SIMPLEX = 1
    LINE_AA = 2

    def __init__(self, key=-1):
        self.key = key
        self.calls = []

    def flip(self, image, flip_code):
        self.calls.append(("flip", flip_code))
        return image.copy()

    def namedWindow(self, title, mode):
        self.calls.append(("namedWindow", title, mode))

    def resizeWindow(self, title, width, height):
        self.calls.append(("resizeWindow", title, width, height))

    def line(self, image, start, end, color, thickness, line_type):
        self.calls.append(("line", start, end, color, thickness, line_type))

    def circle(self, image, center, radius, color, fill, line_type):
        self.calls.append(("circle", center, radius, color, fill, line_type))

    def rectangle(self, image, start, end, color, thickness, line_type):
        self.calls.append(("rectangle", start, end, color, thickness, line_type))

    def putText(self, image, text, origin, font, scale, color, thickness, line_type):
        self.calls.append(("putText", text, origin, color))

    def imshow(self, title, image):
        self.calls.append(("imshow", title, image))

    def waitKey(self, delay):
        return self.key

    def destroyWindow(self, title):
        self.calls.append(("destroyWindow", title))


def test_camera_preview_draws_red_square_and_labeled_landmark_tracking():
    cv2 = FakeCV2()
    preview = CameraPreview(cv2)
    frame = CameraFrame(640, 480, 1.0, image=np.zeros((480, 640, 3), dtype=np.uint8))
    observation = HandObservation(landmark_positions=_hand(), handedness="Right")

    assert preview.show(frame, (observation,))

    rectangles = [call for call in cv2.calls if call[0] == "rectangle"]
    assert len(rectangles) == 1
    _, top_left, bottom_right, color, thickness, _ = rectangles[0]
    assert color == (0, 0, 255)
    assert bottom_right[0] - top_left[0] == bottom_right[1] - top_left[1]
    assert thickness == 3
    assert any(call[:2] == ("putText", "Right HAND - ZOOM") for call in cv2.calls)
    assert len([call for call in cv2.calls if call[0] == "line"]) == 21
    assert len([call for call in cv2.calls if call[0] == "circle"]) == 21

    preview.close()
    assert ("destroyWindow", CameraPreview.WINDOW_TITLE) in cv2.calls


def test_camera_preview_labels_left_hand_as_move_and_right_hand_as_zoom():
    cv2 = FakeCV2()
    observations = (
        HandObservation(landmark_positions=_hand(), handedness="Left"),
        HandObservation(landmark_positions=_hand(), handedness="Right"),
    )
    annotated = CameraPreview.annotate_frame(
        np.zeros((480, 640, 3), dtype=np.uint8),
        observations,
        cv2,
    )

    labels = [call[1] for call in cv2.calls if call[0] == "putText"]
    assert "Left HAND - MOVE" in labels
    assert "Right HAND - ZOOM" in labels
    assert len([call for call in cv2.calls if call[0] == "rectangle"]) == 2


def test_camera_preview_shows_no_hand_state_without_drawing_a_track_box():
    cv2 = FakeCV2()
    preview = CameraPreview(cv2)
    frame = CameraFrame(640, 480, 1.0, image=np.zeros((480, 640, 3), dtype=np.uint8))

    assert preview.show(frame, ())

    assert not [call for call in cv2.calls if call[0] == "rectangle"]
    assert any(call[:2] == ("putText", "NO HAND TRACKED") for call in cv2.calls)

    preview.close()


def test_camera_preview_escape_closes_popup_without_reopening():
    cv2 = FakeCV2(key=27)
    preview = CameraPreview(cv2)
    frame = CameraFrame(640, 480, 1.0, image=np.zeros((480, 640, 3), dtype=np.uint8))

    assert not preview.show(frame, ())
    assert not preview.show(frame, ())

    assert len([call for call in cv2.calls if call[0] == "namedWindow"]) == 1
    assert len([call for call in cv2.calls if call[0] == "destroyWindow"]) == 1
