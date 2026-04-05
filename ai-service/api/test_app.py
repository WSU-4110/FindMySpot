"""
test_app.py — Unit Tests for FindMySpot app.py
Framework : pytest
Run with  : pytest test_app.py -v

Six functions tested (one class per function):
  1. load_camera_config
  2. send_parking_checkin
  3. find_plate_candidates
  4. clean_ocr_text
  5. is_valid_plate
  6. is_cooldown_active

Every test follows the six characteristics of a good unit test from lecture:
  Automatic · Atomic · Single-Responsibility · Independent · Repeatable · Self-Validating
"""

import sys
import json
import os
from unittest import mock
import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP — stub cv2 and easyocr before importing app.py so the
# module-level camera-open code does not crash in a headless test environment.
# ─────────────────────────────────────────────────────────────────────────────
_cv2_mock = mock.MagicMock()
_easyocr_mock = mock.MagicMock()
sys.modules["cv2"] = _cv2_mock
sys.modules["easyocr"] = _easyocr_mock
_cv2_mock.VideoCapture.return_value.isOpened.return_value = True
# cap.read() must return a 2-tuple so the while loop can unpack (ret, frame)
# Returning ret=False causes the loop to break immediately and not block.
_cv2_mock.VideoCapture.return_value.read.return_value = (False, None)

_BOOT_CONFIG = {
    "cameras": [{"camera_id": 0, "floor": 1, "lot": 1, "name": "Boot Cam"}]
}

with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(_BOOT_CONFIG))), \
     mock.patch("json.load", return_value=_BOOT_CONFIG), \
     mock.patch.dict(os.environ, {"CAMERA_ID": "0"}):
    import app


# =============================================================================
# 1. load_camera_config
# =============================================================================
class TestLoadCameraConfig:

    def test_returns_parsed_dict_for_valid_file(self, tmp_path):
        """
        [Single-Responsibility] Only tests the happy path: a well-formed JSON
        config is loaded and returned unchanged.
        """
        config_data = {
            "cameras": [{"camera_id": 1, "floor": 2, "lot": 3, "name": "Entrance"}]
        }
        (tmp_path / "camera_config.json").write_text(json.dumps(config_data))

        with mock.patch("os.path.dirname", return_value=str(tmp_path)):
            result = app.load_camera_config()

        assert result == config_data

    def test_returns_none_for_missing_file(self, tmp_path):
        """
        [Independent] Uses its own empty tmp_path; does not depend on any
        other test having created or removed a file.
        """
        empty_dir = tmp_path / "no_config"
        empty_dir.mkdir()

        with mock.patch("os.path.dirname", return_value=str(empty_dir)):
            result = app.load_camera_config()

        assert result is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        """
        [Atomic] A single assertion confirms that broken JSON produces None
        rather than raising an exception.
        """
        (tmp_path / "camera_config.json").write_text("{ not valid json !!!")

        with mock.patch("os.path.dirname", return_value=str(tmp_path)):
            result = app.load_camera_config()

        assert result is None


# =============================================================================
# 2. send_parking_checkin
# =============================================================================
class TestSendParkingCheckin:

    def test_posts_correct_payload_on_201(self):
        """
        [Automatic] Mock intercepts the network call — no real server needed.
        Verifies that the plate, floor, and lot appear in the outgoing payload.
        """
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 201

        with mock.patch("requests.post", return_value=mock_resp) as mock_post:
            app.send_parking_checkin("ABC1234", 2, 3)

        mock_post.assert_called_once()
        payload = mock_post.call_args[1].get("json") or mock_post.call_args[0][1]
        assert payload["licensePlate"] == "ABC1234"
        assert payload["floor"] == 2
        assert payload["lot"] == 3

    def test_does_not_raise_on_500_status(self):
        """
        [Repeatable] The 500 is deterministic — result is always the same.
        Backend errors must be handled internally and not propagated.
        """
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 500

        with mock.patch("requests.post", return_value=mock_resp):
            app.send_parking_checkin("XYZ9999", 1, 1)   # must not raise

    def test_does_not_raise_on_network_failure(self):
        """
        [Self-Validating] An uncaught exception causes pytest to mark the test
        FAILED automatically — no manual check required.
        """
        import requests as _req
        with mock.patch("requests.post",
                        side_effect=_req.exceptions.ConnectionError("timeout")):
            app.send_parking_checkin("TIMEOUT1", 3, 2)  # must not raise


# =============================================================================
# 3. find_plate_candidates
# =============================================================================
class TestFindPlateCandidates:

    def _blank(self, h=480, w=640):
        return np.zeros((h, w), dtype=np.uint8)

    def _setup_cv2(self, contours):
        _cv2_mock.bilateralFilter.return_value = self._blank()
        _cv2_mock.Canny.return_value = self._blank()
        _cv2_mock.findContours.return_value = (contours, None)

    def test_returns_list_type(self):
        """
        [Atomic] Verifies exactly one thing: the return type is always list.
        """
        self._setup_cv2([])
        result = app.find_plate_candidates(self._blank())
        assert isinstance(result, list)

    def test_rejects_contour_with_square_aspect_ratio(self):
        """
        [Independent] Its own mock contour — does not rely on state from any
        other test. A 100×100 square (ratio 1.0) is below the 1.6 minimum.
        """
        fake_contour = mock.MagicMock()
        self._setup_cv2([fake_contour])
        _cv2_mock.boundingRect.return_value = (10, 10, 100, 100)  # ratio = 1.0

        candidates = app.find_plate_candidates(self._blank())

        assert candidates == []

    def test_accepts_contour_with_valid_plate_dimensions(self):
        """
        [Repeatable] Mock contour is always identical → deterministic result.
        w=300, h=60 gives ratio≈5.0 and area=18000, both within spec.
        """
        fake_contour = mock.MagicMock()
        self._setup_cv2([fake_contour])
        _cv2_mock.boundingRect.return_value = (50, 50, 300, 60)

        candidates = app.find_plate_candidates(self._blank())

        assert len(candidates) == 1
        assert candidates[0] == (50, 50, 300, 60)


# =============================================================================
# 4. clean_ocr_text
# =============================================================================
class TestCleanOcrText:

    def test_strips_special_characters_and_spaces(self):
        """
        [Single-Responsibility] Only tests that non-alphanumeric characters
        (dashes, spaces, dots) are removed from raw OCR output.
        """
        result = app.clean_ocr_text("A-BC 12.3")
        assert result == "ABC123"

    def test_converts_lowercase_to_uppercase(self):
        """
        [Atomic] A single assertion confirms uppercasing works independently
        of any stripping logic.
        """
        result = app.clean_ocr_text("abc123")
        assert result == "ABC123"

    def test_empty_string_returns_empty_string(self):
        """
        [Repeatable] Edge case: empty input always produces empty output.
        """
        result = app.clean_ocr_text("")
        assert result == ""


# =============================================================================
# 5. is_valid_plate
# =============================================================================
class TestIsValidPlate:

    def test_accepts_standard_plate(self):
        """
        [Automatic] Pure function — no I/O, no mocks. A typical US plate
        like 'ABC1234' (7 chars, all alphanumeric) should return True.
        """
        assert app.is_valid_plate("ABC1234") is True

    def test_rejects_plate_that_is_a_state_name(self):
        """
        [Single-Responsibility] Only tests the blocked_words filter path;
        the regex would pass 'MICHIGAN' (8 chars) but the block-list must
        catch it first.
        """
        assert app.is_valid_plate("MICHIGAN") is False

    def test_rejects_plate_shorter_than_four_chars(self):
        """
        [Independent] Isolated check of the minimum-length regex boundary.
        'AB1' has 3 characters and must fail regardless of blocked_words.
        """
        assert app.is_valid_plate("AB1") is False

    def test_rejects_plate_longer_than_eight_chars(self):
        """
        [Atomic] Verifies the upper-length boundary in one assertion.
        'ABCD12345' has 9 characters and must fail the regex.
        """
        assert app.is_valid_plate("ABCD12345") is False


# =============================================================================
# 6. is_cooldown_active
# =============================================================================
class TestIsCooldownActive:

    def test_returns_true_when_same_plate_within_cooldown(self):
        """
        [Repeatable] Deterministic timestamps — result never changes across runs.
        Same plate, 5 s elapsed, 10 s cooldown → cooldown is still active.
        """
        result = app.is_cooldown_active(
            plate="ABC123",
            last_plate="ABC123",
            finalized_at=1000.0,
            current_time=1005.0,   # only 5 s have passed
            cooldown_seconds=10.0,
        )
        assert result is True

    def test_returns_false_when_cooldown_has_expired(self):
        """
        [Atomic] One assertion: after the cooldown window has passed,
        the same plate should be allowed to trigger a new event.
        """
        result = app.is_cooldown_active(
            plate="ABC123",
            last_plate="ABC123",
            finalized_at=1000.0,
            current_time=1015.0,   # 15 s have passed > 10 s cooldown
            cooldown_seconds=10.0,
        )
        assert result is False

    def test_returns_false_for_different_plate(self):
        """
        [Single-Responsibility] Only tests the plate-identity check.
        A different plate must never be blocked by another plate's cooldown.
        """
        result = app.is_cooldown_active(
            plate="XYZ999",
            last_plate="ABC123",
            finalized_at=1000.0,
            current_time=1001.0,   # still within cooldown window, but different plate
            cooldown_seconds=10.0,
        )
        assert result is False