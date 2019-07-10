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
import numpy as np
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
        exposure_time,
        global_gain,
        name=None,
        uid=None,
        exposure_mode="manual",
        nbuffers=1000
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

        self.frame_size_backup = frame_size
        self.frame_rate_backup = frame_rate
        self.exposure_time_backup = exposure_time
        self.global_gain_backup = global_gain

        self.uid = uid
        if self.aravis_capture:
            self.aravis_capture.stream.set_property('packet_timeout',100000)
            self.set_feature('GevSCPSPacketSize', 1500)
            self.timestamp_freq = float(self.aravis_capture.get_feature('GevTimestampTickFrequency'))
            self.current_frame_idx = 0

            # set exposure to the minimum, should work in semi-dark environment
            self.exposure_time = 0
            self._set_dark_image = True
            self.aravis_capture.start_acquisition(nbuffers)
            frame = None
            while frame is None:
                frame = self.get_frame()
            logger.info("Min=Max frame value %d-%d"%(frame.gray.min(),frame.gray.max()))
            logger.info("Setting exposure back to %d"%self.exposure_time_backup)
            self.exposure_time = self.exposure_time_backup
        else:
            self._intrinsics = load_intrinsics(
                self.g_pool.user_dir, self.name, self.frame_size
            )

    def configure_capture(self, frame_size, frame_rate, controls):
        pass

    def get_frame(self):
            ts, data = self.aravis_capture.try_pop_frame(timestamp=True)
            if data is None:
                return
            index = self.current_frame_idx
            self.current_frame_idx += 1

            if self._set_dark_image:
                self.dark_image = data.copy()
                self._set_dark_image = False
                self.exposure_time = self.exposure_time_backup
            if not self.dark_image is None:
                np.subtract(data, self.dark_image, data)
                #data = (data.astype(np.int16)-self.dark_image).astype(np.uint8)
                #data -= self.dark_image
                #data *= 1.5

            return Frame(time.time(), data, index)
            #return Frame(ts/self.timestamp_freq, data, index)

    def recent_events(self, events):
        if self.aravis_capture is None:
            return
        frame = self.get_frame()

        self._recent_frame = frame
        events["frame"] = frame

    def get_init_dict(self):
        d = super().get_init_dict()
        d["frame_size"] = self.frame_size
        d["frame_rate"] = self.frame_rate
        d["exposure_time"] = self.exposure_time
        d["global_gain"] = self.global_gain
        if self.aravis_capture:
            d["name"] = self.name
#            d["uvc_controls"] = self._get_uvc_controls()
        return d

    @property
    def name(self):
        if self.aravis_capture:
            return "%s %s"%(self.aravis_capture.name, self.aravis_capture.get_device_id())
        else:
            return 'Ghost capture'

    @property
    def height(self):
        if not self.frame_size_backup and self.aravis_capture:
            return self.aravis_capture.get_feature('Height')
        else:
            return self.frame_size_backup[1]

    @property
    def width(self):
        if not self.frame_size_backup and self.aravis_capture:
            return self.aravis_capture.get_feature('Width')
        else:
            return self.frame_size_backup[0]

    @property
    def frame_size(self):
        return (self.width, self.height)

    @frame_size.setter
    def frame_size(self, new_size):
        self.aravis_capture.stop_acquisition()

        height = self.set_feature('Height', new_size[1])
        width = self.set_feature('Width', new_size[0])
        size = (width, height)

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

        self.dark_image = None
        self.aravis_capture.start_acquisition()


    @height.setter
    def height(self, new_height):
        self.frame_size = (self.frame_size[0], new_height)

    @width.setter
    def width(self, new_width):
        self.frame_size = (new_width, self.frame_size[1])

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
        self.aravis_capture.stop_acquisition()
        rate = self.set_feature('FPS', new_rate)
        self.frame_rate_backup = rate
        self.aravis_capture.start_acquisition()

    @property
    def global_gain(self):
        if self.aravis_capture:
            return int(self.aravis_capture.get_feature('GlobalGain'))
        else:
            return 1

    @global_gain.setter
    def global_gain(self, new_gain):
        gain = self.set_feature('GlobalGain', int(new_gain))

    @property
    def exposure_time(self):
        if self.aravis_capture:
            return self.aravis_capture.get_feature('ExposureTime')
        else:
            return self.exposure_time_backup

    @exposure_time.setter
    def exposure_time(self, new_et):
        if self.aravis_capture:
            self.aravis_capture.set_feature('ExposureTime', new_et)

    def set_dark_image(self):
        self.exposure_time_backup = self.exposure_time
        self.exposure_time = 0
        self._set_dark_image = True

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

        sensor_control.append(
            ui.Slider(
                "width",
                self,
                min=20,
                max=self.aravis_capture.get_feature('WidthMax'),
                step = 1,
                label="Width",
            )
        )

        sensor_control.append(
            ui.Slider(
                "height",
                self,
                min=20,
                max=self.aravis_capture.get_feature('HeightMax'),
                step = 1,
                label="Height",
            )
        )

        sensor_control.append(
            ui.Slider(
                "frame_rate",
                self,
                min=10,
                max=1077,
                step=8,
                label="Frame rate",
            )
        )

        sensor_control.append(
            ui.Slider(
                "exposure_time",
                self,
                min=8,
                max=33980,
                step=8,
                label="Exposure Time",
            )
        )

        sensor_control.append(
            ui.Slider(
                "global_gain",
                self,
                min=0,
                max=16,
                step=1,
                label="Global gain",
            )
        )

        ui_elements.append(sensor_control)

        image_processing = ui.Growing_Menu(label="Image Post Processing")
        image_processing.collapsed = True

        image_processing.append(ui.Button("set dark image", self.set_dark_image))

        ui_elements.append(image_processing)
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
            "exposure_time": 4000,
            "global_gain": 1,
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
