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
from time import sleep

from .base_backend import Base_Source, Playback_Source, Base_Manager, EndofVideoError
from camera_models import load_intrinsics

import numpy as np
from multiprocessing import cpu_count
import os.path
from fractions import Fraction

import cv2

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
        self.jpeg_buffer = None
        self.yuv_buffer = None
        self.height, self.width, _ = frame.shape

    def copy(self):
        return Frame(self.timestamp, self._frame, self.index)

    @property
    def img(self):
        return self._frame

    @property
    def bgr(self):
        return self.img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = self._frame.mean(-1).astype(self._frame.dtype)
        return self._gray


class OpenCV_Source(Base_Source):
    """Simple OpenCV-based capture for non-UVC sources.

    Attributes:
        source_path (str): Path to source file
        timestamps (str): Path to timestamps file
    """

    def __init__(self, g_pool, device_path=None, flip_vertical=False, flip_horizontal=False, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)

        # minimal attribute set
        self._initialised = True
        self.device_path = device_path
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.current_frame_idx = 0
        if not self.device_path or not stat.S_ISCHR(os.stat(self.device_path).st_mode):
            logger.error('Init failed. Device could not be found at `%s`'%source_path)
            self._initialised = False
            return

        try:
            self._opencv_cap = cv2.VideoCapture(self.device_path)
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
                if self._initialised and self.video_stream:
                    # test self.play only if requires_playback is True
                    if not requires_playback or self.play:
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
        return int(self._opencv_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._opencv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_index(self):
        return self.current_frame_idx

    @property
    def frame_rate(self):
        return self._opencv_cap.get(cv2.CAP_PROP_FPS)

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

    def get_frame(self):
        try:
            ret, image = self._opencv_cap.read()
            if self.flip_vertical:
                image[:] = np.flip(image,0)
            if self.flip_horizontal:
                image[:] = np.flip(image,1)
            timestamp = self._opencv_cap.get(cv2.CAP_PROP_POS_MSEC)
            index = self.current_frame_idx
            self.current_frame_idx += 1
            #logger.info("shape: %s, dtype: %s"%(str(image.shape),str(image.dtype)))
            return Frame(timestamp, image, index=index)
        except Exception as e:
            logger.error("get_frame: %s:%s"%(e.__class__,str(e)))

    def recent_events(self, events):
        try:
            frame = self.get_frame()
        except Exception as e:
            logger.error("recent_event: %s:%s"%(e.__class__,str(e)))
        events['frame'] = frame

    @property
    def jpeg_support(self):
        return False

    def cleanup(self):
        if self._opencv_cap:
            self._opencv_cap.release()
            self._opencv_cap = None
        super().cleanup()


class OpenCV_Manager(Base_Manager):
    """Summary

    Attributes:
        file_exts (list): File extensions to filter displayed files
        root_folder (str): Folder path, which includes file sources
    """
    gui_name = 'OpenCV source'
    video_device_pattern = '/dev/video*'

    def __init__(self, g_pool,):
        super().__init__(g_pool)
        self.flip_vertical, self.flip_horizontal = False, False

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
            'flip_vertical': self.flip_vertical
        }
        self.activate_source(settings)

    def on_drop(self, paths):
        for p in paths:
            if os.path.splitext(p)[-1] in self.file_exts:
                self.activate(p)
                return

    def activate_source(self, settings={}):
        if self.g_pool.process == 'world':
            self.notify_all({'subject':'start_plugin',"name":"OpenCV_Source",'args':settings})
        else:
            self.notify_all({'subject':'start_eye_capture','target':self.g_pool.process, "name":"OpenCV_Source",'args':settings})

    def recent_events(self,events):
        pass
