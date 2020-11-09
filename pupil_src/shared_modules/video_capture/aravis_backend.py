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
from .base_backend import InitialisationError, Base_Source, Base_Manager, SourceInfo
from camera_models import Camera_Model
from .utils import Check_Frame_Stripes, Exposure_Time

from ._npufunc import subtract_nowrap

import gi
gi.require_version('Aravis', '0.8')
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
        if self._frame.ndim == 2:
            self._gray = self._frame
        elif self._frame.ndim == 3:
            self._img = self._frame
        self.yuv_buffer = None
        self.height, self.width = frame.shape[:2]

    def copy(self):
        return Frame(self.timestamp, self._frame, self.index)

    @property
    def img(self):
        return self.bgr

    @property
    def bgr(self):
        if self._img is None and self._gray is not None:
            self._img = np.repeat(self._gray[..., np.newaxis], 3, 2)
        return self._img

    @property
    def gray(self):
        if self._gray is None and self._img is not None:
            self._gray = (self._img.sum(2)/3).astype(np.uint8)
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
        nbuffers=1000,
        *args,
        **kwargs,
    ):

        super().__init__(g_pool, *args, **kwargs)
        self.cam = None
        self._status = False
        self._set_dark_image = False

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

            self.stream.set_property('packet_timeout', 100000)
            self.set_feature('GevSCPSPacketSize', 1500)
            #self.set_feature('PixelMappingFormat', 'LowBits')
            self.current_frame_idx = 0

            self.exposure_time = exposure_time
            self.global_gain = global_gain
            self.frame_size = frame_size
            self.frame_rate = frame_rate


            # The camera is gigevision1.2, which doesn't support PTP apparently
            # maybe this is overkill for fMRI sampling rate
            # we perform a number of timestampLatch to get an approximate time sync
            try:

                self.timestamp_freq = self.get_feature('GevTimestampTickFrequency')
                logger.info(f"timestamp_freq={self.timestamp_freq}")
                self.timestamp_offset = None
                camera_os_time_diffs = []
                for i in range(100):
                    os_time = time.time()
                    latch_res = self.dev.execute_command('GevTimestampControlLatch')
                    camera_ts = self.get_feature('GevTimestampValue')/self.timestamp_freq
                    camera_os_time_diffs.append(os_time-camera_ts)
                    logger.debug(f"{latch_res} {os_time} - {camera_ts} = {os_time-camera_ts}")
                self.timestamp_offset = np.mean(camera_os_time_diffs)
                logger.info(
                    "OS-Camera time diff: mean=%f std=%f"%(
                    self.timestamp_offset,
                    np.std(camera_os_time_diffs))
                    )

            except Exception as err:
                camera_os_time_diffs = []
                latch_res = None
                camera_ts = None

            self.create_buffers()
            #self._start_capture()
        else:
            self._intrinsics = Camera_Model.from_file(
                self.g_pool.user_dir, self.name, self.frame_size
            )


    def _flush_buffers(self, keep=True):
        buf = True
        while buf:
            buf = self.stream.try_pop_buffer()
            if buf and keep:
                self.stream.push_buffer(buf)

    def create_buffers(self):

        payload = self.cam.get_payload()
        if payload == self.payload and sum(self.stream.get_n_buffers())==self.nbuffers:
            return

        self._flush_buffers(keep=False)

        self.payload = payload
        logger.info("Creating %d memory buffers of size %d"%(self.nbuffers, payload))
        for _ in range(0, self.nbuffers):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    def _start_capture(self):

        # set exposure to the minimum, should work in semi-dark environment
        self.exposure_time_backup = self.exposure_time
        self.exposure_time = 0
        #self._set_dark_image = True

        self.cam.start_acquisition()
        first_buf_os_time = time.time() # get approximate time of the first buffer
        time.sleep(.0001)
        buf = None
        while buf is None:
            buf = self.stream.try_pop_buffer()

        if self.timestamp_offset is None:
            # get an approximate time difference if camera does not support timestamplatch
            self.timestamp_offset = first_buf_os_time - buf.get_timestamp()*1e-9
        self.stream.push_buffer(buf)

        """
        logger.info(
            'first frame at %f %f %f %f'%(
                buf.get_timestamp()*1e-9,
                buf.get_system_timestamp()*1e-9,
                self.timestamp_offset,
                self.g_pool.get_timestamp()))
        """

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
            self._flush_buffers()

    def get_frame(self):
        buf = self.stream.try_pop_buffer()
        #print(self.stream.get_n_buffers())
        if buf:
            payload_type = buf.get_payload_type()
            if payload_type != Aravis.BufferPayloadType.IMAGE:
                logger.warning("Buffer with payload of type %s"%payload_type.value_nick)
            buffer_status = buf.get_status()
            if buffer_status == Aravis.BufferStatus.SUCCESS:
                data = self._array_from_buffer_address(buf)
                ts = buf.get_timestamp()
            else:
                logger.warning('Buffer STATUS: %s'%buffer_status.value_nick)
                return None
            self.stream.push_buffer(buf)
        else:
            return None

        index = self.current_frame_idx
        self.current_frame_idx += 1

        if self._set_dark_image:
            self.dark_image = data.copy()
            logger.info('dark_image max = %d'%self.dark_image.max())
            self._set_dark_image = False
            self.exposure_time = self.exposure_time_backup

        if not self.dark_image is None:
            subtract_nowrap(data, self.dark_image, data)

        return Frame(ts*1e-9 + self.timestamp_offset, data, index)

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
        return im.copy()

    def recent_events(self, events):
        if (self.cam is None) or (not self._status):
            return
        frame = None
        while frame is None:
            frame = self.get_frame()

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
        status_back = self._status
        if status_back:
            self._stop_capture()
        self.set_feature('Height', new_size[1])
        self.set_feature('Width', new_size[0])
        # we set the real size that the system accepted
        size = (self.get_feature('Width'),
            self.get_feature('Height'))

        if tuple(size) != tuple(new_size):
            logger.warning(
                "{} resolution capture mode not available. Selected {}.".format(
                    new_size, size
                )
            )
        self.frame_size_backup = size

        self._intrinsics = Camera_Model.from_file(
            self.g_pool.user_dir, self.name, self.frame_size
        )
        self.dark_image = None
        self.create_buffers()
        if status_back:
            self._start_capture()

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
        if val is None:
            logger.error('cannot set None value')
            return
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
            logger.warning("Feature type not implemented: %s"%ntype)


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
            gg = self.get_feature('GlobalGain')
            if gg:
                return int(gg)
        else:
            return self.global_gain_backup

    @global_gain.setter
    def global_gain(self, new_gain):
        if new_gain and self.cam:
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
        self._status = False
        self._flush_buffers()
        self._set_dark_image = True
        self._status = True

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
                    max=self.get_feature(slider_name+'Max') or 640,
                    step = 1,
                    label=slider_name,
                    )
                )

        if self.frame_rate:
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

        if self.exposure_time:
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

        if self.global_gain:
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

    def get_cameras(self):
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        self.devices = [Aravis.get_device_id(i) for i in range(0, n)]

        print(self.devices)
        return [
            SourceInfo(
                label=f"{device} @ Aravis",
                manager=self,
                key=f"cam.{device}",
            )
            for device in self.devices
        ]

    def activate(self, key):
        if not key:
            return

        try:
            source_uid = key[4:]
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
            "global_gain": 0,
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

    def cleanup(self):
        self.devices = None
