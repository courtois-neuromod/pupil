"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import time
import logging
import aravis
from version_utils import VersionFormat
from .base_backend import InitialisationError, Base_Source, Base_Manager
from camera_models import load_intrinsics
from .utils import Check_Frame_Stripes, Exposure_Time

# check versions for our own depedencies as they are fast-changing
assert VersionFormat(uvc.__version__) >= VersionFormat("0.13")

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Aravis_Source(Base_Source):
    """
    Camera Capture is a class that encapsualtes uvc.Capture:
    """

    def __init__(
        self,
        g_pool,
        frame_size,
        frame_rate,
        name=None,
        uid=None,
        exposure_mode="manual",
    ):

        super().__init__(g_pool)
        self.aravis_capture = None
        self._restart_in = 3

        self.devices = aravis.get_device_ids()

        # if uid is supplied we init with that
        if uid:
            try:
                self.aravis_capture = aravis.Camera(uid)
            except aravis.AravisException:
                logger.warning(
                    "No camera found that matched {}".format(preferred_names)
                )

    def configure_capture(self, frame_size, frame_rate, controls):


    def recent_events(self, events):
        try:
            frame = self.uvc_capture.get_frame(0.05)

            if self.preferred_exposure_time:
                target = self.preferred_exposure_time.calculate_based_on_frame(frame)
                if target is not None:
                    self.exposure_time = target

            if self.checkframestripes and self.checkframestripes.require_restart(frame):
                # set the self.frame_rate in order to restart
                self.frame_rate = self.frame_rate
                logger.info("Stripes detected")

        except uvc.StreamError:
            self._recent_frame = None
            self._restart_logic()
        except (AttributeError, uvc.InitError):
            self._recent_frame = None
            time.sleep(0.02)
            self._restart_logic()
        else:
            if (
                self.ts_offset
            ):  # c930 timestamps need to be set here. The camera does not provide valid pts from device
                frame.timestamp = uvc.get_time_monotonic() + self.ts_offset
            frame.timestamp -= self.g_pool.timebase.value
            self._recent_frame = frame
            events["frame"] = frame
            self._restart_in = 3


    def get_init_dict(self):
        d = super().get_init_dict()
        d["frame_size"] = self.frame_size
        d["frame_rate"] = self.frame_rate
        d["check_stripes"] = self.check_stripes
        d["exposure_mode"] = self.exposure_mode
        if self.uvc_capture:
            d["name"] = self.name
            d["uvc_controls"] = self._get_uvc_controls()
        else:
            d["preferred_names"] = self.name_backup
        return d

    @property
    def name(self):
        if self.aravis_capture:
            return "%s %s"%(self.aravis_capture.name, self.aravis_capture.get_device_id())

    @property
    def frame_size(self):
        if self.uvc_capture:
            return self.aravis_capture.frame_size
        else:
            return self.frame_size_backup

    @frame_size.setter
    def frame_size(self, new_size):
        # closest match for size
        sizes = [
            abs(r[0] - new_size[0]) + abs(r[1] - new_size[1])
            for r in self.uvc_capture.frame_sizes
        ]
        best_size_idx = sizes.index(min(sizes))
        size = self.uvc_capture.frame_sizes[best_size_idx]
        if tuple(size) != tuple(new_size):
            logger.warning(
                "{} resolution capture mode not available. Selected {}.".format(
                    new_size, size
                )
            )
        self.uvc_capture.frame_size = size
        self.frame_size_backup = size

        self._intrinsics = load_intrinsics(
            self.g_pool.user_dir, self.name, self.frame_size
        )


    @property
    def frame_rate(self):
        if self.aravis_capture:
            return self.uvc_capture.frame_rate
        else:
            return self.frame_rate_backup

    @frame_rate.setter
    def frame_rate(self, new_rate):
        # closest match for rate
        rates = [abs(r - new_rate) for r in self.uvc_capture.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.uvc_capture.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning(
                "{}fps capture mode not available at ({}) on '{}'. Selected {}fps. ".format(
                    new_rate, self.uvc_capture.frame_size, self.uvc_capture.name, rate
                )
            )
        self.uvc_capture.frame_rate = rate
        self.frame_rate_backup = rate

    @property
    def exposure_time(self):
        if self.uvc_capture:
            try:
                controls_dict = dict(
                    [(c.display_name, c) for c in self.uvc_capture.controls]
                )
                return controls_dict["Absolute Exposure Time"].value
            except KeyError:
                return None
        else:
            return self.exposure_time_backup

    @exposure_time.setter
    def exposure_time(self, new_et):
        try:
            controls_dict = dict(
                [(c.display_name, c) for c in self.uvc_capture.controls]
            )
            if abs(new_et - controls_dict["Absolute Exposure Time"].value) >= 1:
                controls_dict["Absolute Exposure Time"].value = new_et
        except KeyError:
            pass

    @property
    def jpeg_support(self):
        return True

    @property
    def online(self):
        return bool(self.uvc_capture)

    def deinit_ui(self):
        self.remove_menu()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Local USB Source: {}".format(self.name)
        self.update_menu()

    def update_menu(self):
        del self.menu[:]
        from pyglui import ui


        ui_elements = []

        # lets define some  helper functions:
        def gui_load_defaults():
            for c in self.uvc_capture.controls:
                try:
                    c.value = c.def_val
                except:
                    pass

        def gui_update_from_device():
            for c in self.uvc_capture.controls:
                c.refresh()

        def set_frame_size(new_size):
            self.frame_size = new_size

        def set_frame_rate(new_rate):
            self.frame_rate = new_rate
            self.update_menu()

        if self.uvc_capture is None:
            ui_elements.append(ui.Info_Text("Capture initialization failed."))
            self.menu.extend(ui_elements)
            return

        ui_elements.append(ui.Info_Text("{} Controls".format(self.name)))
        sensor_control = ui.Growing_Menu(label="Sensor Settings")
        sensor_control.append(
            ui.Info_Text("Do not change these during calibration or recording!")
        )
        sensor_control.collapsed = False
        image_processing = ui.Growing_Menu(label="Image Post Processing")
        image_processing.collapsed = True

        sensor_control.append(
            ui.Selector(
                "frame_size",
                self,
                setter=set_frame_size,
                selection=self.uvc_capture.frame_sizes,
                label="Resolution",
            )
        )

        def frame_rate_getter():
            return (
                self.uvc_capture.frame_rates,
                [str(fr) for fr in self.uvc_capture.frame_rates],
            )

        sensor_control.append(
            ui.Selector(
                "frame_rate",
                self,
                selection_getter=frame_rate_getter,
                setter=set_frame_rate,
                label="Frame rate",
            )
        )

        self.menu.extend(ui_elements)

    def cleanup(self):
        self.devices.cleanup()
        self.devices = None
        if self.uvc_capture:
            self.uvc_capture.close()
            self.uvc_capture = None
        super().cleanup()


class Aravis_Manager(Base_Manager):
    """Manages Aravis (Gig-E-Vision) sources

    """

    gui_name = "Aravis (Gig-E-Vision)"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.devices = aravis.get_device_ids()

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()

        from pyglui import ui

        self.add_auto_select_button()
        ui_elements = []
        ui_elements.append(ui.Info_Text("Local UVC sources"))

        def dev_selection_list():
            default = ("Select to activate")
            self.devices = aravis.get_device_ids()
            dev_pairs = [default] + self.devices

        ui_elements.append(
            ui.Selector(
                "selected_source",
                selection_getter=dev_selection_list,
                getter=lambda: None,
                setter=self.activate,
                label="Activate source",
            )
        )
        self.menu.extend(ui_elements)

    def activate(self, source_uid):
        if not source_uid:
            return

        try:
            if source_uid not in self.devices:
                logger.error("The selected camera is not available.")
                return
        except ValueError as ve:
            logger.error(str(ve))
            return

        settings = {
            "frame_size": self.g_pool.capture.frame_size,
            "frame_rate": self.g_pool.capture.frame_rate,
            "uid": source_uid,
        }
        if self.g_pool.process == "world":
            self.notify_all(
                {"subject": "start_plugin", "name": "Aravis_Source", "args": settings}
            )
        else:
            self.notify_all(
                {
                    "subject": "start_eye_capture",
                    "target": self.g_pool.process,
                    "name": "Aravis_Source",
                    "args": settings,
                }
            )


    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.devices = None

    def recent_events(self, events):
        pass
