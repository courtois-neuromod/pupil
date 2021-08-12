"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2021 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T

import cv2
import numpy as np
from gl_utils import adjust_gl_view, clear_gl_screen, basic_gl_setup
import OpenGL.GL as gl

import glfw
from gl_utils import GLFWErrorReporting

GLFWErrorReporting.set_default()

from circle_detector import CircleTracker
from platform import system

import audio

from pyglui import ui
from pyglui.cygl.utils import draw_polyline, RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

from .mixin import MonitorSelectionMixin
from .controller import (
    GUIMonitor,
    MarkerWindowController,
    MarkerWindowStateClosed,
    MarkerWindowStateOpened,
    MarkerWindowStateIdle,
    MarkerWindowStateShowingMarker,
    MarkerWindowStateAnimatingInMarker,
    MarkerWindowStateAnimatingOutMarker,
    UnhandledMarkerWindowStateError,
)
from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyMode,
    ChoreographyAction,
    ChoreographyNotification,
)


logger = logging.getLogger(__name__)


class FixedScreenMarkerChoreographyPlugin(ScreenMarkerChoreographyPlugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between
    """

    label = "Fixed Screen Marker Calibration"

    @classmethod
    def selection_label(cls) -> str:
        return "Fixed Screen Marker"

    @classmethod
    def selection_order(cls) -> float:
        return 1.0

    @staticmethod
    def get_list_of_markers_to_show(mode: ChoreographyMode) -> list:
        if ChoreographyMode.CALIBRATION == mode:
            return [(0.5, 0.5), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        if ChoreographyMode.VALIDATION == mode:
            return [(0.5, 1.0), (1.0, 0.5), (0.5, 0.0), (0.0, 0.5)]
        raise ValueError(f"Unknown mode {mode}")

    def recent_events(self, events):
        super().recent_events(events)

        frame = events.get("frame")
        state = self.__marker_window.window_state
        should_animate = True

        self.__marker_window.update_state()

        if isinstance(state, MarkerWindowStateClosed):
            return

        elif isinstance(state, MarkerWindowStateOpened):
            assert self.is_active  # Sanity check
            pass  # Continue with processing the frame

        else:
            raise UnhandledMarkerWindowStateError(state)

        # Always save pupil positions
        self.pupil_list.extend(events["pupil"])

        # Detect reference circle marker
        #detected_marker = self.__detect_reference_circle_marker(frame.gray)

        # Signal marker window controller that a marker was detected (for feedback)
        #self.__marker_window.is_marker_detected = detected_marker is not None

        if isinstance(state, MarkerWindowStateIdle):
            assert self.__currently_shown_marker_position is None  # Sanity check
            if self.__current_list_of_markers_to_show:
                self.__currently_shown_marker_position = (
                    self.__current_list_of_markers_to_show.pop(0)
                )
                logger.debug(
                    f"Moving screen marker to site at {self.__currently_shown_marker_position}"
                )
                self.__marker_window.show_marker(
                    marker_position=self.__currently_shown_marker_position,
                    should_animate=should_animate,
                )
                return
            else:
                # No more markers to show; stop calibration choreography.
                self._signal_should_stop(mode=self.current_mode)
                return

        if isinstance(state, MarkerWindowStateAnimatingInMarker):
            assert self.__currently_shown_marker_position is not None  # Sanity check
            pass  # No-op

        elif isinstance(state, MarkerWindowStateShowingMarker):
            assert self.__currently_shown_marker_position is not None  # Sanity check

            if detected_marker is not None:
                ref = {}
                ref["norm_pos"] = self.__currently_shown_marker_position
                ref["screen_pos"] = self.__currently_shown_marker_position
                ref["timestamp"] = self.g_pool.get_time_monotonic()
                self.ref_list.append(ref)

            should_move_to_next_marker = len(self.ref_list) == self.sample_duration * (
                self.__ref_count_for_current_marker_position + 1
            )

            if should_move_to_next_marker:
                # Finished collecting samples for current active site
                self.__currently_shown_marker_position = None
                self.__ref_count_for_current_marker_position += 1
                self.__marker_window.hide_marker(should_animate=should_animate)

        elif isinstance(state, MarkerWindowStateAnimatingOutMarker):
            assert self.__currently_shown_marker_position is None  # Sanity check
            pass  # No-op

        else:
            raise UnhandledMarkerWindowStateError(state)

        # Update UI
        self.__marker_window.draw_window()
        self.status_text = self.__currently_shown_marker_position

    ### Internal

    def _perform_start(self):

        self.__current_list_of_markers_to_show = self.get_list_of_markers_to_show(
            mode=self.current_mode,
        )
        self.__currently_shown_marker_position = None
        self.__ref_count_for_current_marker_position = 0

        super()._perform_start()

        self.__marker_window.open_window(
            title=self.current_mode.label,
            monitor_name=self.selected_monitor_name,
            is_fullscreen=self.is_fullscreen,
        )

    ### Private

    def _on_window_did_close(self):
        self._signal_should_stop(mode=self.current_mode)
