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
import numpy as np
import ctypes
from version_utils import VersionFormat
from .base_backend import InitialisationError, Base_Source, Base_Manager
from camera_models import load_intrinsics
from .utils import Check_Frame_Stripes, Exposure_Time

from gi.repository import Aravis

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AravisException(Exception):
    pass

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
    Aravis_Source is a class that encapsulates python-aravis
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
        nbuffers=100
    ):

        super().__init__(g_pool)
        self.cam = None
        self._restart_in = 3
        self._status = False
        self._set_dark_image = True

        logger.warning(
            "Activating camera: %s" % uid
        )
        # if uid is supplied we init with that
        if uid:
            try:
                self.cam = Aravis.Camera.new(uid)
                #self.aravis_capture = aravis.Camera(uid)
            #except aravis.AravisException as e:
            except TypeError as e:
                logger.warning(
                    "No Aravis camera found or error in initialization"
                )
                logger.warning(str(e))

        self.uid = uid
        self.frame_size_backup = frame_size
        self.frame_rate_backup = frame_rate
        self.exposure_time_backup = exposure_time
        self.global_gain_backup = global_gain
        self.nbuffers = nbuffers

        if self.cam:

            self.dev = self.cam.get_device()
            self.stream = self.cam.create_stream(None, None)
            if self.stream is None:
                raise RuntimeError("Error creating stream")
            self.payload = 0

            self.stream.set_property('packet_timeout',100000)
            self.set_feature('GevSCPSPacketSize', 1500)
            self.timestamp_freq = self.get_feature('GevTimestampTickFrequency')
            self.current_frame_idx = 0

            self.exposure_time = exposure_time
            self.global_gain = global_gain
            self.frame_size = frame_size
            self.frame_rate = frame_rate

        else:
            self._intrinsics = load_intrinsics(
                self.g_pool.user_dir, self.name, self.frame_size
            )


    def create_buffers(self):

        payload = self.cam.get_payload()
        if payload == self.payload and sum(self.stream.get_n_buffers())==self.nbuffers:
            return

        # flush all buffers
        buf = True
        while buf:
            buf = self.stream.try_pop_buffer()

        self.payload = payload
        logger.info("Creating %d memory buffers of size %d"%(self.nbuffers, payload))
        for _ in range(0, self.nbuffers):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    def _start_capture(self):
        # set exposure to the minimum, should work in semi-dark environment
        self.exposure_time_backup = self.exposure_time
        self.exposure_time = 0

        self._set_dark_image = True

        self.create_buffers()
        self.cam.start_acquisition()
        frame = None
        while frame is None:
            frame = self.get_frame()

        self.exposure_time = self.exposure_time_backup
        self._status = True
        logger.info('started capture successfully')

    def _stop_capture(self):
        logger.info('stopping capture')
        self.cam.stop_acquisition()
        if self.stream:
            self.stream.set_emit_signals(False)
        self._status = False

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        if value:
            self._start_capture()
        else:
            self._stop_capture()

    def get_frame(self):
        buf = self.stream.try_pop_buffer()
        #print(self.stream.get_n_buffers())
        if buf and buf.get_status() == Aravis.BufferStatus.SUCCESS:
            data = self._array_from_buffer_address(buf)
            self.stream.push_buffer(buf)
            ts = buf.get_timestamp()
        else:
            if buf:
                logger.warning('WRONG buffer STATUS %d'%buf.get_status())
                self.stream.push_buffer(buf)
            return None

        index = self.current_frame_idx
        self.current_frame_idx += 1

        if self._set_dark_image:
            self.dark_image = data.copy()
            self._set_dark_image = False

        if not self.dark_image is None:
            np.subtract(data, self.dark_image, data)

        return Frame(time.time(), data, index)
        #return Frame(ts/self.timestamp_freq, data, index)

    def _array_from_buffer_address(self, buf):
        if not buf:
            return None
        pixel_format = buf.get_image_pixel_format()
        bits_per_pixel = pixel_format >> 16 & 0xff
        if bits_per_pixel == 8:
            INTP = ctypes.POINTER(ctypes.c_uint8)
        else:
            INTP = ctypes.POINTER(ctypes.c_uint16)
        addr = buf.get_data()
        ptr = ctypes.cast(addr, INTP)
        im = np.ctypeslib.as_array(ptr, (buf.get_image_height(), buf.get_image_width()))
        im = im.copy()
        return im

    def recent_events(self, events):
        if (self.cam is None) or (not self._status):
            return
        frame = self.get_frame()
        if frame is None:
            logger.debug('no frame')
            return
        logger.debug('frame')
        self._recent_frame = frame
        events["frame"] = frame

    def get_init_dict(self):
        d = super().get_init_dict()
        d["frame_size"] = self.frame_size
        d["frame_rate"] = self.frame_rate
        d["exposure_time"] = self.exposure_time
        d["global_gain"] = self.global_gain
        d["uid"] = self.uid
        return d

    @property
    def name(self):
        if self.cam:
            return self.uid
        else:
            return 'Ghost capture'

    @property
    def height(self):
        return self.frame_size_backup[1]

    @property
    def width(self):
        return self.frame_size_backup[0]

    @property
    def frame_size(self):
        return (self.width, self.height)

    @frame_size.setter
    def frame_size(self, new_size):
        if new_size == self.frame_size:
            return
        height = self.set_feature('Height', new_size[1])
        width = self.set_feature('Width', new_size[0])
        # we set the real size that the system accepted
        size = (width, height)
        """
        r_x,r_y,r_w,r_h,r_s= self.g_pool.u_r.get()
        if r_x+r_w > width:
            r_w = width-r_x
        if r_y+r_h > height:
            r_h = height-r_y
        self.g_pool.u_r = UIRoi(r_x,r_y,r_w,r_h,new_size)
        """
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

    @height.setter
    def height(self, new_height):
        self.frame_size = (self.frame_size[0], new_height)

    @width.setter
    def width(self, new_width):
        self.frame_size = (new_width, self.frame_size[1])


    def get_feature_type(self, name):
        genicam = self.dev.get_genicam()
        node = genicam.get_node(name)
        if not node:
            logger.error("Feature {} does not seem to exist in camera".format(name))
            return
        return node.get_node_name()

    def set_feature(self, name, val):

        ntype = self.get_feature_type(name)
        if ntype in ("String", "Enumeration", "StringReg"):
            return self.dev.set_string_feature_value(name, val)
        elif ntype == "Integer":
            return self.dev.set_integer_feature_value(name, int(val))
        elif ntype == "MaskedIntReg" or ntype == "IntReg":
            return Aravis.GcStructEntryNode.set_value(
                self.dev.get_genicam().get_node(name), int(val))
        elif ntype == "Float":
            return self.dev.set_float_feature_value(name, float(val))
        elif ntype == "Boolean":
            return self.dev.set_integer_feature_value(name, int(val))
        else:
            logger.warning("Feature type not implemented: %s", ntype)

        """
        # the set_feature function in python-aravis doesn't work
        # here is a workaround inspired by arv-tool sourcecode
        feat = self.dev.get_feature(name)
        if feat is None:
            return
        feat.set_value(value)
        return feat.get_value()
        """

    def get_feature(self, name):

        ntype = self.get_feature_type(name)
        if ntype in ("Enumeration", "String", "StringReg"):
            return self.dev.get_string_feature_value(name)
        elif ntype == "Integer":
            return self.dev.get_integer_feature_value(name)
        elif ntype == "MaskedIntReg" or ntype == "IntReg":
            return Aravis.GcStructEntryNode.get_value(
                self.dev.get_genicam().get_node(name))
        elif ntype == "Float":
            return self.dev.get_float_feature_value(name)
        elif ntype == "Boolean":
            return self.dev.get_integer_feature_value(name)
        else:
            logger.warning("Feature type not implemented: %s", ntype)

        """
        feat = self.dev.get_feature(name)
        if feat is None:
            return
        return feat.get_value()
        """

    @property
    def frame_rate(self):
        if self.cam:
            return self.get_feature('FPS')
        else:
            return self.frame_rate_backup

    @frame_rate.setter
    def frame_rate(self, new_rate):
        rate = self.set_feature('FPS', new_rate)
        self.frame_rate_backup = rate

    @property
    def global_gain(self):
        if self.cam:
            return int(self.get_feature('GlobalGain'))
        else:
            return self.global_gain_backup

    @global_gain.setter
    def global_gain(self, new_gain):
        gain = self.set_feature('GlobalGain', int(new_gain))

    @property
    def exposure_time(self):
        if self.cam:
            return self.get_feature('ExposureTime')
        else:
            return self.exposure_time_backup

    @exposure_time.setter
    def exposure_time(self, new_et):
        if self.cam:
            self.set_feature('ExposureTime', new_et)

    def set_dark_image(self):
        self.exposure_time_backup = self.exposure_time
        self.exposure_time = 0
        self._set_dark_image = True

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return bool(self.cam)

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

        if self.cam is None:
            ui_elements.append(ui.Info_Text("Capture initialization failed."))
            self.menu.extend(ui_elements)
            return

        ui_elements.append(ui.Info_Text("{} Controls".format(self.name)))
        sensor_control = ui.Growing_Menu(label="Sensor Settings")
        sensor_control.append(
            ui.Info_Text("Do not change these during calibration or recording!")
        )
        sensor_control.collapsed = False
        """
        sensor_control.append(
            ui.Selector(
                "frame_size",
                self,
                selection=[(640,480),(320,240),(256,256)],
                label="Frame size",
            )
        )
        """

        for slider_name in ['Width','Height']:
            if self.get_feature_type(slider_name):
                sensor_control.append(
                ui.Slider(
                    slider_name.lower(),
                    self,
                    min=20,
                    max=self.get_feature(slider_name+'Max'),
                    step = 1,
                    label=slider_name,
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

        self.startstop = ui.Thumb(
            "status", self, label="S", hotkey="g"
        )
        self.g_pool.quickbar = ui.Stretching_Menu("Quick Bar", (0, 100), (100, -100))
        self.g_pool.quickbar.insert(0, self.startstop)
        self.g_pool.gui.append(self.g_pool.quickbar)

    def cleanup(self):
        if self.cam:

            self.cam.stop_acquisition()
            del self.stream
            del self.cam
            del self.dev
        super().cleanup()


class Aravis_Manager(Base_Manager):
    """Manages Aravis (Gig-E-Vision) sources

    """

    gui_name = "Aravis (Gig-E-Vision)"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.devices = []

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()

        from pyglui import ui

        self.add_auto_select_button()
        ui_elements = []
        ui_elements.append(ui.Info_Text("Aravis sources"))

        def dev_selection_list():
            Aravis.update_device_list()
            n = Aravis.get_n_devices()
            self.devices = [Aravis.get_device_id(i) for i in range(0, n)]

            default = (None, "Select to activate")
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

        logger.info('activating %s' % source_uid)
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
                    "subject": "start_eye_plugin",
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
