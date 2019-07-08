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

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp, frame, index):
        self._frame = frame
        self.timestamp = timestamp
        self.index = index
        self._img = None
        self._gray = None
        if self._frame.ndim < 3:
            self._gray = self._frame
        if self._frame.ndim == 3:
            self.img = self._frame
        self.jpeg_buffer = None
        self.yuv_buffer = None
        self.height, self.width = frame.shape[:2]

    def copy(self):
        return Frame(self.timestamp, self._frame, self.index)

    @property
    def img(self):
        return self._img

    @property
    def bgr(self):
        return self.img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = self._frame.mean(-1).astype(self._frame.dtype)
        return self._gray

class Aravis_Source(Base_Source):
    """
    Camera Capture is a class that encapsulates oython-aravis
    """

    def __init__(
        self,
        g_pool,
        frame_size,
        frame_rate,
        name=None,
        uid=None,
        exposure_mode="manual",
        nbuffers=250
    ):

        super().__init__(g_pool)
        self.aravis_capture = None
        self._restart_in = 3

        logger.warning(
            "Activating camera: %s" % uid
        )
        # if uid is supplied we init with that
        if uid:
            try:
                self.aravis_capture = aravis.Camera(uid)
            except aravis.AravisException as e:
                logger.warning(
                    "No Aravis camera found or error in initialization"
                )
                logger.warning(str(e))
        self.current_frame_idx = 0
        self.aravis_capture.start_acquisition(nbuffers)

    def configure_capture(self, frame_size, frame_rate, controls):
        pass

    def recent_events(self, events):
        try:
            ts, data = self.aravis_capture.try_pop_frame(timestamp=True)
            if data is None:
                return
            print(ts, data.shape)
            index = self.current_frame_idx
            self.current_frame_idx += 1
            frame = Frame(ts, data, index)
        except Exception:
            #TODO
            pass
        else:
            self._recent_frame = frame
            events["frame"] = frame

    def get_init_dict(self):
        d = super().get_init_dict()
        d["frame_size"] = self.frame_size
        d["frame_rate"] = self.frame_rate
        if self.aravis_capture:
            d["name"] = self.name
#            d["uvc_controls"] = self._get_uvc_controls()
        return d

    @property
    def name(self):
        if self.aravis_capture:
            return "%s %s"%(self.aravis_capture.name, self.aravis_capture.get_device_id())

    @property
    def height(self):
        if self.aravis_capture:
            return self.aravis_capture.get_feature('Height')

    @property
    def width(self):
        if self.aravis_capture:
            return self.aravis_capture.get_feature('Width')

    @property
    def frame_size(self):
        return (self.height, self.width)

    @frame_size.setter
    def frame_size(self, new_size):
        # closest match for size

        height = self.set_feature('Height', new_size[0])
        width = self.set_feature('Width', new_size[1])
        size = (height, width)

        if tuple(size) != tuple(new_size):
            logger.warning(
                "{} resolution capture mode not available. Selected {}.".format(
                    new_size, size
                )
            )
        self.frame_size_backup = size

        self._intrinsics = load_intrinsics(
            self.g_pool.user_dir, self.name, self.frame_size
        )

    @height.setter
    def height(self, new_height):
        self.frame_size = (new_height, self.frame_size[1])

    @width.setter
    def height(self, new_width):
        self.frame_size = (self.frame_size[0], new_width)

    def set_feature(self, name, value):
        # the set_feature function in python-aravis doesn't work
        # here is a workaround inspired by arv-tool sourcecode
        feat = self.aravis_capture.dev.get_feature(name)
        feat.set_value(value)
        return feat.get_value()

    @property
    def frame_rate(self):
        if self.aravis_capture:
            return self.aravis_capture.get_feature('FPS')
        else:
            return self.frame_rate_backup

    @frame_rate.setter
    def frame_rate(self, new_rate):
        rate = self.set_feature('FPS', new_rate)
        self.frame_rate_backup = rate

    @property
    def exposure_time(self):
        pass

    @exposure_time.setter
    def exposure_time(self, new_et):
        pass

    @property
    def jpeg_support(self):
        return True

    @property
    def online(self):
        return bool(self.aravis_capture)

    def deinit_ui(self):
        self.remove_menu()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Aravis Source: {}".format(self.name)
        self.update_menu()

    def update_menu(self):
        del self.menu[:]
        from pyglui import ui


        ui_elements = []

        if self.aravis_capture is None:
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
            ui.Slider(
                "width",
                self,
                min=73,
                max=640,
                step = 1,
                label="Width",
            )
        )

        sensor_control.append(
            ui.Slider(
                "height",
                self,
                min=73,
                max=640,
                step = 1,
                label="Height",
            )
        )

        sensor_control.append(
            ui.Slider(
                "frame_rate",
                self,
                min=10,
                max=1076,
                step=1,
                label="Frame rate",
            )
        )
        ui_elements.append(sensor_control)

        self.menu.extend(ui_elements)

    def cleanup(self):
        if self.aravis_capture:
            self.aravis_capture.stop_acquisition()
            self.aravis_capture.shutdown()
            self.aravis_capture = None
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
        ui_elements.append(ui.Info_Text("Aravis sources"))

        def dev_selection_list():
            default = (None, "Select to activate")
            self.devices = aravis.get_device_ids()
            dev_pairs = [default] + [(dev,dev) for dev in self.devices]
            return zip(*dev_pairs)

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
