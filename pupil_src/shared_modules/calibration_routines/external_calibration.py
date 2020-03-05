import logging

from .calibration_plugin_base import Calibration_Plugin
from .finish_calibration import finish_calibration

logger = logging.getLogger(__name__)

class External_Calibration(Calibration_Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between

    """

    def __init__(self, g_pool, frame_size=None):
        super().__init__(g_pool)

        class MockCapture():
            def __init__(self, frame_size):
                self.frame_size = frame_size
        self.g_pool.capture = MockCapture(frame_size)

    def init_ui(self):
        pass

    def on_notify(self, notification):
        if notification["subject"] == 'calibrate.from_external_data':
            print(self.g_pool.active_gaze_mapping_plugin)
            logger.info("calibrate from external data")
            pupil_list = notification['pupil_list']
            ref_list = notification['ref_list']
            finish_calibration(self.g_pool, pupil_list, ref_list)
            print(self.g_pool.active_gaze_mapping_plugin)

    def get_init_dict(self):
        settings = super().get_init_dict()
        settings['frame_size'] = self.g_pool.capture.frame_size
        return settings
