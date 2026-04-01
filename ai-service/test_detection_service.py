"""
Unit Tests for detection_service.py — FindMySpot
Tests the PlateDetectionService reporting layer
Framework: pytest
"""

import requests
import pytest
from unittest.mock import Mock, patch

from detection_service import PlateDetectionService, simulate_camera_detections


# ──────────────────────────────────────────────
# TEST 1 — PlateDetectionService.__init__
# ──────────────────────────────────────────────
class TestPlateDetectionServiceInit:
    """Tests for PlateDetectionService.__init__()"""

    def test_backend_url_is_set_correctly(self):
        """__init__ stores the backend URL."""
        service = PlateDetectionService("http://localhost:5000")
        assert service.backend_url == "http://localhost:5000"

    def test_detection_endpoint_is_built_correctly(self):
        """__init__ builds the detection endpoint from the backend URL."""
        service = PlateDetectionService("http://localhost:5000")
        assert service.detection_endpoint == "http://localhost:5000/api/detection/record"

    def test_default_backend_url_is_used(self):
        """__init__ uses the default backend URL when none is provided."""
        service = PlateDetectionService()
        assert service.backend_url == "http://localhost:3000"

    def test_default_detection_endpoint_is_built(self):
        """__init__ builds the default detection endpoint correctly."""
        service = PlateDetectionService()
        assert service.detection_endpoint == "http://localhost:3000/api/detection/record"


# ──────────────────────────────────────────────
# TEST 2 — PlateDetectionService.report_detection success
# ──────────────────────────────────────────────
class TestReportDetectionSuccess:
    """Tests for successful report_detection() behavior"""

    @patch("detection_service.requests.post")
    def test_report_detection_returns_backend_response(self, mock_post):
        """report_detection returns parsed JSON when the request succeeds."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "matched": False,
            "message": "Detection recorded"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService("http://localhost:3000")
        result = service.report_detection(
            license_plate="ABC123",
            floor=2,
            lot=5,
            location_description="Entry Gate - Floor 2",
            confidence=0.98,
            camera_id="CAM_01"
        )

        assert result == {
            "success": True,
            "matched": False,
            "message": "Detection recorded"
        }

    @patch("detection_service.requests.post")
    def test_report_detection_calls_requests_post_once(self, mock_post):
        """report_detection sends exactly one POST request."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        service.report_detection("ABC123", floor=1, lot=1)

        mock_post.assert_called_once()

    @patch("detection_service.requests.post")
    def test_report_detection_calls_correct_endpoint(self, mock_post):
        """report_detection sends the request to the detection endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService("http://localhost:4000")
        service.report_detection("ABC123", floor=2, lot=5)

        args, _ = mock_post.call_args
        assert args[0] == "http://localhost:4000/api/detection/record"

    @patch("detection_service.requests.post")
    def test_report_detection_returns_matched_true_response(self, mock_post):
        """report_detection returns matched=True correctly when backend reports a match."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "matched": True,
            "message": "Match found"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        result = service.report_detection("XYZ789", floor=3, lot=4)

        assert result["matched"] is True


# ──────────────────────────────────────────────
# TEST 3 — PlateDetectionService.report_detection payload
# ──────────────────────────────────────────────
class TestReportDetectionPayload:
    """Tests for request payload formatting in report_detection()"""

    @patch("detection_service.requests.post")
    def test_license_plate_is_uppercased_in_payload(self, mock_post):
        """report_detection uppercases the license plate before sending."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        service.report_detection("abc123", floor=2, lot=5)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        assert payload["licensePlate"] == "ABC123"

    @patch("detection_service.requests.post")
    def test_payload_contains_all_expected_fields(self, mock_post):
        """report_detection includes floor, lot, location, confidence, camera, and coordinates."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        service.report_detection(
            license_plate="XYZ789",
            floor=3,
            lot=12,
            location_description="Main Entrance - Floor 3",
            confidence=0.95,
            camera_id="CAM_02",
            latitude=42.3314,
            longitude=-83.0458
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        assert payload["licensePlate"] == "XYZ789"
        assert payload["floor"] == 3
        assert payload["lot"] == 12
        assert payload["location"] == "Main Entrance - Floor 3"
        assert payload["confidence"] == 0.95
        assert payload["cameraId"] == "CAM_02"
        assert payload["latitude"] == 42.3314
        assert payload["longitude"] == -83.0458

    @patch("detection_service.requests.post")
    def test_default_location_is_used_when_description_missing(self, mock_post):
        """report_detection builds a default location string when none is provided."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        service.report_detection("DEF456", floor=4, lot=9)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        assert payload["location"] == "Floor 4, Lot 9"

    @patch("detection_service.requests.post")
    def test_payload_allows_none_optional_fields(self, mock_post):
        """report_detection allows optional fields like camera and coordinates to remain None."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        service.report_detection("QWE123", floor=1, lot=2)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        assert payload["cameraId"] is None
        assert payload["latitude"] is None
        assert payload["longitude"] is None


# ──────────────────────────────────────────────
# TEST 4 — PlateDetectionService.report_detection errors
# ──────────────────────────────────────────────
class TestReportDetectionErrors:
    """Tests for exception handling in report_detection()"""

    @patch("detection_service.requests.post")
    def test_report_detection_returns_none_on_request_exception(self, mock_post):
        """report_detection returns None if the HTTP request fails."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

        service = PlateDetectionService()
        result = service.report_detection("ABC123", floor=2, lot=5)

        assert result is None

    @patch("detection_service.requests.post")
    def test_report_detection_uses_timeout_of_five_seconds(self, mock_post):
        """report_detection sends the request with timeout=5."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        service.report_detection("ABC123", floor=2, lot=5)

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 5

    @patch("detection_service.requests.post")
    def test_report_detection_returns_none_on_http_error(self, mock_post):
        """report_detection returns None when raise_for_status triggers an HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        result = service.report_detection("ERR500", floor=1, lot=1)

        assert result is None

    @patch("detection_service.requests.post")
    def test_report_detection_returns_none_on_timeout_exception(self, mock_post):
        """report_detection returns None when the request times out."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        service = PlateDetectionService()
        result = service.report_detection("TIME123", floor=3, lot=6)

        assert result is None


# ──────────────────────────────────────────────
# TEST 5 — simulate_camera_detections behavior
# ──────────────────────────────────────────────
class TestSimulateCameraDetections:
    """Tests for simulate_camera_detections()"""

    @patch("detection_service.PlateDetectionService.report_detection")
    def test_simulate_camera_detections_calls_report_detection_three_times(self, mock_report):
        """simulate_camera_detections sends three detections."""
        mock_report.return_value = {"success": True}

        simulate_camera_detections()

        assert mock_report.call_count == 3

    @patch("detection_service.PlateDetectionService.report_detection")
    @patch("builtins.print")
    def test_first_detection_uses_expected_values(self, mock_print, mock_report):
        """The first simulated detection uses the expected camera test data."""
        mock_report.return_value = {"success": True}

        simulate_camera_detections()

        first_call = mock_report.call_args_list[0]
        kwargs = first_call.kwargs

        assert kwargs["license_plate"] == "ABC123"
        assert kwargs["floor"] == 2
        assert kwargs["lot"] == 5
        assert kwargs["location_description"] == "Entry Gate - Floor 2"
        assert kwargs["confidence"] == 0.98
        assert kwargs["camera_id"] == "CAM_01"

    @patch("detection_service.PlateDetectionService.report_detection")
    @patch("builtins.print")
    def test_last_detection_uses_expected_values(self, mock_print, mock_report):
        """The last simulated detection uses the expected camera test data."""
        mock_report.return_value = {"success": True}

        simulate_camera_detections()

        last_call = mock_report.call_args_list[-1]
        kwargs = last_call.kwargs

        assert kwargs["license_plate"] == "DEF456"
        assert kwargs["floor"] == 1
        assert kwargs["lot"] == 8
        assert kwargs["location_description"] == "Ground Floor North"
        assert kwargs["confidence"] == 0.92
        assert kwargs["camera_id"] == "CAM_03"

    @patch("detection_service.PlateDetectionService.report_detection")
    @patch("builtins.print")
    def test_middle_detection_uses_expected_values(self, mock_print, mock_report):
        """The middle simulated detection uses the expected camera test data."""
        mock_report.return_value = {"success": True}

        simulate_camera_detections()

        middle_call = mock_report.call_args_list[1]
        kwargs = middle_call.kwargs

        assert kwargs["license_plate"] == "XYZ789"
        assert kwargs["floor"] == 3
        assert kwargs["lot"] == 12
        assert kwargs["location_description"] == "Main Entrance - Floor 3"
        assert kwargs["confidence"] == 0.95
        assert kwargs["camera_id"] == "CAM_02"


# ──────────────────────────────────────────────
# TEST 6 — Integration-style reporting flow
# ──────────────────────────────────────────────
class TestDetectionServiceIntegrationStyle:
    """Integration-style tests for the reporting flow"""

    @patch("detection_service.requests.post")
    def test_full_reporting_flow_success(self, mock_post):
        """A full reporting call succeeds and returns backend data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "matched": True,
            "message": "Detection recorded and user matched"
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService("http://localhost:3000")
        result = service.report_detection(
            license_plate="ghi321",
            floor=5,
            lot=7,
            location_description="South Ramp - Floor 5",
            confidence=0.97,
            camera_id="CAM_09"
        )

        assert result["success"] is True
        assert result["matched"] is True
        assert result["message"] == "Detection recorded and user matched"

    @patch("detection_service.PlateDetectionService.report_detection")
    @patch("builtins.print")
    def test_simulate_camera_detections_prints_each_result(self, mock_print, mock_report):
        """simulate_camera_detections prints each detection result."""
        mock_report.return_value = {"success": True, "matched": False}

        simulate_camera_detections()

        assert mock_print.call_count == 3

    @patch("detection_service.requests.post")
    def test_integration_flow_preserves_custom_location(self, mock_post):
        """Full reporting flow keeps a custom location description in the payload."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True, "matched": False}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        service = PlateDetectionService()
        service.report_detection(
            license_plate="AAA111",
            floor=2,
            lot=4,
            location_description="West Entrance",
            confidence=0.91,
            camera_id="CAM_10"
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        assert payload["location"] == "West Entrance"

    @patch("detection_service.PlateDetectionService.report_detection")
    @patch("builtins.print")
    def test_simulate_camera_detections_prints_json_output(self, mock_print, mock_report):
        """simulate_camera_detections prints formatted detection results."""
        mock_report.return_value = {"success": True, "matched": False}

        simulate_camera_detections()

        first_print_arg = mock_print.call_args_list[0][0][0]
        assert "Detection Result:" in first_print_arg