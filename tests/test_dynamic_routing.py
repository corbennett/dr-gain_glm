import unittest

import numpy as np
import polars as pl

from gain_glm.dynamic_routing import (
    DEFAULT_MODEL,
    SessionData,
    model_data,
    prepare,
)


class DynamicRoutingAdapterTests(unittest.TestCase):
    def make_session(self):
        trial_rows = {
            "start_time": [10.0, 11.0],
            "stop_time": [11.0, 12.0],
            "stim_start_time": [10.2, 11.2],
            "is_vis_rewarded": [False, True],
            "is_aud_target": [True, False],
            "is_aud_nontarget": [False, False],
            "is_vis_target": [False, True],
            "is_vis_nontarget": [False, False],
            "is_hit": [True, True],
            "is_miss": [False, False],
            "is_correct_reject": [False, False],
            "is_false_alarm": [False, False],
        }
        sample_times = np.linspace(10, 12, 41)
        pose_columns = {}
        for feature in ("ear_base_l", "jaw", "nose_tip", "whisker_pad_l_side"):
            pose_columns[feature + "_x"] = np.linspace(0, 1, sample_times.size)
            pose_columns[feature + "_y"] = np.linspace(1, 0, sample_times.size)
            pose_columns[feature + "_likelihood"] = np.ones(sample_times.size)
            pose_columns[feature + "_temporal_norm"] = np.zeros(sample_times.size)
        return SessionData(
            nwb_path="fake.nwb",
            dt=0.1,
            task_start_time=10.0,
            task_end_time=12.0,
            n_time=20,
            trials=pl.DataFrame(trial_rows),
            trial_start_times=np.array([10.0, 11.0]),
            trial_end_times=np.array([11.0, 12.0]),
            trial_context=np.array([-1.0, 1.0]),
            lick_times=np.array([10.4, 11.4]),
            reward_times=np.array([10.5, 11.5]),
            running_speed=np.column_stack((sample_times, np.sin(sample_times))),
            pupil=np.column_stack((sample_times, np.cos(sample_times))),
            pose=pl.DataFrame(pose_columns),
            side_frame_times=sample_times,
        )

    def test_named_inputs_are_task_relative(self):
        data = model_data(self.make_session())
        np.testing.assert_allclose(data.events["is_aud_target"], [0.2])
        np.testing.assert_allclose(data.events["is_vis_target"], [1.2])
        np.testing.assert_array_equal(data.trial_index[:10], 0)
        np.testing.assert_array_equal(data.trial_index[10:], 1)
        np.testing.assert_array_equal(data.trial_values["trial_context"], [-1, 1])

    def test_default_model_prepares_without_builder_functions(self):
        prepared = prepare(self.make_session(), DEFAULT_MODEL)
        self.assertEqual(set(prepared.base_blocks), set(DEFAULT_MODEL.predictor_names))
        self.assertEqual(prepared.base_blocks["is_hit"].shape, (20, 9))
        self.assertTrue(prepared.fit_mask.any())


if __name__ == "__main__":
    unittest.main()
