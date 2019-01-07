'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os, stat
import time

from .base_backend import Base_Source, Playback_Source, Base_Manager, EndofVideoError
from camera_models import load_intrinsics

import numpy as np
from multiprocessing import cpu_count
import os.path
from fractions import Fraction

import v4l2capture
import select

# logging
import logging
logger = logging.getLogger(__name__)

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


class V4L2_Source(Base_Source):
    """Simple V4L2-based capture for non-UVC sources.

    Attributes:
        source_path (str): Path to source file
        timestamps (str): Path to timestamps file
    """

    def __init__(self, g_pool, device_path=None, flip_vertical=False, flip_horizontal=False, capture_grey=True, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)

        # minimal attribute set
        self._initialised = True
        self.device_path = device_path
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.capture_grey = capture_grey
        self.current_frame_idx = 0
        if not self.device_path or not stat.S_ISCHR(os.stat(self.device_path).st_mode):
            logger.error('Init failed. Device could not be found at `%s`'%source_path)
            self._initialised = False
            return

        try:
            self._v4l2_cap = v4l2capture.Video_device(self.device_path)
            fourcc = 'BGR'
            if self.capture_grey:
                fourcc='GREY'
            self._width, self._height = self._v4l2_cap.set_format(640, 480, fourcc=fourcc)
            self._v4l2_cap.create_buffers(30)
            self._v4l2_cap.queue_all_buffers()
            self._v4l2_cap.start()
            self._initialised = True
        except Exception as e:
            logger.error("%s:%s"%(e.__class__,str(e)))
            logger.error("TODO: find potential exceptions source: %s"%self.device_path)
            self._initialised = False
            return

    def ensure_initialisation(fallback_func=None):
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def run_func(self, *args, **kwargs):
                if self._initialised:
                    return func(self, *args, **kwargs)
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    logger.debug('Initialisation required.')
            return run_func
        return decorator

    @property
    def initialised(self):
        return self._initialised

    @property
    def frame_size(self):
        return self._v4l2_cap.get_format()[:2]

    def get_frame_index(self):
        return self.current_frame_idx

    @property
    def frame_rate(self):
        return self._v4l2_cap.set_fps(30)/1000.

    def get_init_dict(self):
        if self.g_pool.app == 'capture':
            settings = super().get_init_dict()
            settings['device_path'] = self.device_path
            return settings
        else:
            raise NotImplementedError()

    @property
    def name(self):
        if self.device_path:
            return self.device_path
        else:
            return 'File source in ghost mode'

    @ensure_initialisation()
    def get_frame(self):
        try:
            select.select((self._v4l2_cap,), (), ())
            raw_frame = self._v4l2_cap.read_and_queue()
            if self.capture_grey:
                image = np.frombuffer(raw_frame,dtype=np.uint8).reshape(self._height, self._width).copy()
            else:
                image = np.frombuffer(raw_frame,dtype=np.uint8).reshape(self._height, self._width, 3).copy()
            if self.flip_vertical:
                image[:] = np.flip(image,0)
            if self.flip_horizontal:
                image[:] = np.flip(image,1)
            #timestamp = self.gpool.get_timestamp()
            timestamp = time.clock_gettime(time.CLOCK_MONOTONIC)
            index = self.current_frame_idx
            self.current_frame_idx += 1
            #logger.info("shape: %s, dtype: %s"%(str(image.shape),str(image.dtype)))
            return Frame(timestamp, image, index=index)
            print(frame.img.shape)
        except Exception as e:
            logger.error("get_frame: %s:%s"%(e.__class__,str(e)))

    def recent_events(self, events):
        try:
            frame = self.get_frame()
            events['frame'] = frame
        except Exception as e:
            logger.error("recent_event: %s:%s"%(e.__class__,str(e)))

    @property
    def jpeg_support(self):
        return False

    def cleanup(self):
        if self._v4l2_cap:
            self._v4l2_cap.stop()
            self._v4l2_cap.close()
            self._v4l2_cap = None
        super().cleanup()


class V4L2_Manager(Base_Manager):
    """Summary

    Attributes:
        file_exts (list): File extensions to filter displayed files
        root_folder (str): Folder path, which includes file sources
    """
    gui_name = 'V4L2 source'
    video_device_pattern = '/dev/video*'

    def __init__(self, g_pool,):
        super().__init__(g_pool)
        self.flip_vertical, self.flip_horizontal = False, False
        self.capture_grey = True

    def init_ui(self):
        self.add_menu()
        from pyglui import ui
        self.menu.append(ui.Info_Text('Select the video source'))

        def split_enumeration():
            import glob
            video_devices = [(s,s) for s in sorted(glob.glob(self.video_device_pattern))]
            video_devices.insert(0, (None, 'Select to activate'))
            return zip(*video_devices)

        self.menu.append(ui.Info_Text("Flip capture (both to rotate 180): "))
        self.menu.append(ui.Switch('flip_horizontal',self,label='horizontally'))
        self.menu.append(ui.Switch('flip_vertical',self,label='vertically'))

        self.menu.append(ui.Switch('capture_grey',self,label='capture grayscale'))

        self.menu.append(ui.Selector(
            'selected_source',
            selection_getter=split_enumeration,
            getter=lambda: None,
            setter=self.activate,
            label='Video Source'
        ))

    def deinit_ui(self):
        self.remove_menu()

    def activate(self, full_path):
        if not full_path:
            return
        settings = {
            'device_path': full_path,
            'flip_horizontal': self.flip_horizontal,
            'flip_vertical': self.flip_vertical,
            'capture_grey':self.capture_grey
        }
        self.activate_source(settings)

    def on_drop(self, paths):
        for p in paths:
            if os.path.splitext(p)[-1] in self.file_exts:
                self.activate(p)
                return

    def activate_source(self, settings={}):
        if self.g_pool.process == 'world':
            self.notify_all({'subject':'start_plugin',"name":"V4L2_Source",'args':settings})
        else:
            self.notify_all({'subject':'start_eye_capture','target':self.g_pool.process, "name":"V4L2_Source",'args':settings})

    def recent_events(self,events):
        pass
