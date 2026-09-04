import unittest
from unittest import mock

import numpy as np
import polars as pl

from gain_glm import ModelSpec, Signal
from gain_glm.dynamic_routing import (
    DEFAULT_DROPOUTS,
    DEFAULT_MODEL,
    LATE_STIMULUS_PREDICTOR_NAMES,
    NO_FACE_MODEL,
    NO_HIT_LONG_STIM_MODEL,
    ONLY_BASELINE_MODEL,
    STIMULUS_EVENTS,
    load_session,
    prepare,
)


class DynamicRoutingAdapterTests(unittest.TestCase):
    def load_fake_session(self, *models):
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
        pupil_values = np.cos(sample_times)
        pupil_values[5] = 999
        pupil_bad_frames = np.zeros(sample_times.size, dtype=bool)
        pupil_bad_frames[5] = True
        tables = {
            "/processing/behavior/licks": pl.DataFrame({"timestamps": [10.4, 11.4]}),
            "/processing/behavior/rewards": pl.DataFrame({"timestamps": [10.5, 11.5]}),
            "/processing/behavior/running_speed": pl.DataFrame(
                {"timestamps": sample_times, "data": np.sin(sample_times)}
            ),
            "/processing/behavior/eye_tracking": pl.DataFrame(
                {
                    "timestamps": sample_times,
                    "pupil_area": pupil_values,
                    "pupil_is_bad_frame": pupil_bad_frames,
                }
            ),
            "/processing/behavior/lp_side_camera": pl.DataFrame(pose_columns),
            "/acquisition/frametimes_side_camera": pl.DataFrame(
                {"timestamps": sample_times}
            ),
        }
        scanned_paths = []

        def read_nwb(nwb_path, path):
            self.assertEqual(nwb_path, "fake.nwb")
            self.assertEqual(path, "/intervals/trials")
            return pl.DataFrame(trial_rows)

        def scan_nwb(nwb_path, path):
            self.assertEqual(nwb_path, "fake.nwb")
            scanned_paths.append(path)
            return tables[path].lazy()

        with (
            mock.patch(
                "gain_glm.dynamic_routing.lazynwb.read_nwb",
                side_effect=read_nwb,
            ),
            mock.patch(
                "gain_glm.dynamic_routing.lazynwb.scan_nwb",
                side_effect=scan_nwb,
            ),
        ):
            session = load_session("fake.nwb", *models)
        return session, scanned_paths

    def make_session(self):
        session, _ = self.load_fake_session(DEFAULT_MODEL)
        return session

    def test_named_inputs_are_task_relative(self):
        data = self.make_session().data
        np.testing.assert_allclose(data.events["is_aud_target"], [0.2])
        np.testing.assert_allclose(data.events["is_vis_target"], [1.2])
        np.testing.assert_array_equal(data.trial_index[:40], 0)
        np.testing.assert_array_equal(data.trial_index[40:], 1)
        np.testing.assert_array_equal(data.trial_values["trial_context"], [-1, 1])
        self.assertNotIn("is_miss", data.events)
        context_baseline = data.signals["context_baseline"]
        self.assertIsNone(context_baseline.times)
        np.testing.assert_array_equal(context_baseline.values[:40], -1)
        np.testing.assert_array_equal(context_baseline.values[40:], 1)
        self.assertNotIn(999, data.signals["pupil_area"].values)

    def test_no_face_model_does_not_load_or_process_pose(self):
        session, scanned_paths = self.load_fake_session(NO_FACE_MODEL)
        self.assertNotIn("/processing/behavior/lp_side_camera", scanned_paths)
        self.assertNotIn("/acquisition/frametimes_side_camera", scanned_paths)
        self.assertTrue(
            {"ear", "jaw", "nose", "whisker_pad"}.isdisjoint(session.data.signals)
        )

    def test_loader_reads_only_sources_required_by_model(self):
        running_model = ModelSpec(
            (Signal("running_speed", window=(0, 0), n_basis=1),),
            name="running_only",
            dt=DEFAULT_MODEL.dt,
        )
        session, scanned_paths = self.load_fake_session(running_model)
        self.assertEqual(
            scanned_paths,
            ["/processing/behavior/running_speed"],
        )
        self.assertEqual(set(session.data.signals), {"running_speed"})
        self.assertEqual(session.data.events, {})
        self.assertEqual(session.data.trial_values, {})

    def test_loader_uses_union_of_multiple_models(self):
        running_model = ModelSpec(
            (Signal("running_speed", window=(0, 0), n_basis=1),),
            name="running_only",
            dt=DEFAULT_MODEL.dt,
        )
        pupil_model = ModelSpec(
            (Signal("pupil_area", window=(0, 0), n_basis=1),),
            name="pupil_only",
            dt=DEFAULT_MODEL.dt,
        )
        session, scanned_paths = self.load_fake_session(running_model, pupil_model)
        self.assertEqual(
            scanned_paths,
            [
                "/processing/behavior/running_speed",
                "/processing/behavior/eye_tracking",
            ],
        )
        self.assertEqual(set(session.data.signals), {"running_speed", "pupil_area"})

    def test_loader_rejects_unsupported_sources_before_nwb_access(self):
        model = ModelSpec(
            (Signal("unknown", window=(0, 0), n_basis=1),),
            name="unknown",
            dt=DEFAULT_MODEL.dt,
        )
        with (
            mock.patch("gain_glm.dynamic_routing.lazynwb.read_nwb") as read_nwb,
            self.assertRaisesRegex(ValueError, "unsupported.*unknown"),
        ):
            load_session("fake.nwb", model)
        read_nwb.assert_not_called()

    def test_default_model_prepares_without_builder_functions(self):
        prepared = prepare(self.make_session(), DEFAULT_MODEL)
        self.assertEqual(set(prepared.base_blocks), set(DEFAULT_MODEL.predictor_names))
        self.assertEqual(prepared.base_blocks["is_hit"].shape, (80, 9))
        for source, name in zip(STIMULUS_EVENTS, LATE_STIMULUS_PREDICTOR_NAMES):
            predictor = DEFAULT_MODEL.predictor(name)
            self.assertEqual(predictor.source, source)
            self.assertEqual(predictor.window, (0.1, 1))
            self.assertEqual(predictor.n_basis, 9)
            self.assertEqual(predictor.gains, ("context",))
            self.assertEqual(prepared.base_blocks[name].shape, (80, 9))
        self.assertEqual(DEFAULT_MODEL.fit_window, (-0.5, 1.0))
        self.assertEqual(DEFAULT_MODEL.fit_events, STIMULUS_EVENTS)
        self.assertEqual(DEFAULT_MODEL.dropouts, DEFAULT_DROPOUTS)
        self.assertTrue(prepared.fit_mask.all())

        resolved = {
            dropout.name: dropout.resolve(prepared) for dropout in DEFAULT_DROPOUTS
        }
        self.assertEqual(
            resolved["early_stim_context_gain"].gain_terms,
            tuple(("context", name) for name in STIMULUS_EVENTS),
        )
        self.assertEqual(
            resolved["late_stim_context_gain"].gain_terms,
            tuple(("context", name) for name in LATE_STIMULUS_PREDICTOR_NAMES),
        )

        self.assertEqual(
            tuple(dropout.name for dropout in NO_HIT_LONG_STIM_MODEL.dropouts),
            ("context", "context_baseline"),
        )
        self.assertEqual(
            tuple(dropout.name for dropout in ONLY_BASELINE_MODEL.dropouts),
            ("context_baseline",),
        )
        for name in ("ear", "jaw", "nose", "whisker_pad"):
            self.assertEqual(
                ONLY_BASELINE_MODEL.predictor(name).orthogonalize_against,
                "context_baseline",
            )


if __name__ == "__main__":
    unittest.main()
