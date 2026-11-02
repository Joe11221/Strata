"""Experimental source-image admission: the same parallel requests as the MTP and no change to stock engines."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from strata_control import drafting
from tests.terminal.launch_support import HW_12, Home, catalog

ROOT = Path(__file__).resolve().parents[2]


class ServingChoice(unittest.TestCase):
    def test_only_matching_source_images_enable_serving(self):
        for marker, enabled in ((None, False), ("0", False), ("1", True)):
            env = dict(os.environ)
            env.pop("STRATA_DFLASH2_SERVE", None)
            if marker is not None:
                env["STRATA_DFLASH2_SERVE"] = marker
            result = subprocess.run(
                [sys.executable, "-c", "from strata_control.drafting import ENGINE_SERVES_DFLASH2; print(ENGINE_SERVES_DFLASH2)"],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), str(enabled))

    def test_every_drafter_keeps_fitted_parallelism(self):
        with Home():
            for served, mode in ((True, "dflash2"), (False, "dflash2"), (True, "mtp"), (True, None)):
                with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", served):
                    runner = Runner(S.LaunchSettings(family="qwen", model="Q2_0", drafting=mode), catalog=catalog(HW_12))
                    self.assertEqual(runner.catalog_parallel(), S.fit_parallel(12))

    def test_explicit_parallel_requests_are_allowed_with_dflash2(self):
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True):
            self.assertFalse(S.check_parallel(S.LaunchSettings(drafting="dflash2", parallel=2), vram_gb=24))
            self.assertFalse(S.check_parallel(S.LaunchSettings(drafting="mtp", parallel=2), vram_gb=24))

    def test_engine_restrictions_are_reported_as_mode_fallbacks(self):
        for reason in ("--batch", "--pipeline-windows (use 0)", "--prefill-head-alias"):
            log = f"strata generate: --draft dflash2 is off: it is not available with {reason} in this build; the MTP drafts instead"
            self.assertEqual(drafting.fallback_note(log), drafting.mtp_line(drafting.NOT_HERE))
        self.assertEqual(drafting.fallback_note("strata serve: DFlash2 drafter on CUDA5 (test.pack)"), "")


class DockerVariable(unittest.TestCase):
    DOCKER = {"STRATA_IN_DOCKER": "1"}

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data = Path(folder.name) / "data"
        pack = drafting.pack_path(self.data)
        pack.parent.mkdir(parents=True)
        with pack.open("wb") as out:
            out.truncate(2 << 20)
        # the layout of a hand-written profile: the packs live apart from the data folder that holds the drafter
        self.cfg = {"model_name": "Swift-IQ3-S", "parallel": 2, "drafting": "mtp",
                    "args": ["--pack", str(Path(folder.name) / "packs" / "swift-iq3_s"), "--mtp", str(self.data / "mtp" / "rt"), "--spec", "4"]}

    def test_it_does_nothing_outside_docker_or_when_empty(self):
        for env in ({"DRAFTING": "dflash2"}, self.DOCKER, {**self.DOCKER, "DRAFTING": " "}):
            before = {**self.cfg, "args": list(self.cfg["args"])}
            self.assertIsNone(drafting.apply_env(self.cfg, env))
            self.assertEqual(self.cfg, before)

    def test_dflash2_on_swift_keeps_its_parallel_requests_gets_serial_windows_and_finds_the_drafter(self):
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True):
            self.assertEqual(drafting.apply_env(self.cfg, {**self.DOCKER, "DRAFTING": "DFlash2"}), "dflash2")
            self.assertEqual((self.cfg["drafting"], self.cfg["parallel"]), ("dflash2", 2))
            self.assertEqual(self.cfg["args"][-2:], ["--pipeline-windows", "0"])
            d = drafting.apply(self.cfg, klass="other", data_dir=drafting.data_dir_of(self.cfg["args"]))
            self.assertEqual(d.flags, ("--draft", "dflash2", "--draft-model", str(drafting.pack_path(self.data))))

    def test_a_stock_engine_keeps_the_profile_settings(self):
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", False):
            drafting.apply_env(self.cfg, {**self.DOCKER, "DRAFTING": "dflash2"})
            self.assertEqual(self.cfg["parallel"], 2)
            self.assertNotIn("--pipeline-windows", self.cfg["args"])

    def test_mtp_overrides_a_profile_that_saved_dflash2(self):
        self.cfg["drafting"] = "dflash2"
        self.cfg["args"] += ["--draft", "dflash2", "--draft-model", "/x.pack"]
        with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True):
            drafting.apply_env(self.cfg, {**self.DOCKER, "DRAFTING": "mtp"})
            d = drafting.apply(self.cfg, klass="other", data_dir=drafting.data_dir_of(self.cfg["args"]))
        self.assertEqual((d.drafter, self.cfg["parallel"]), ("mtp", 2))
        self.assertNotIn("--draft", self.cfg["args"])

    def test_a_wrong_value_is_an_error(self):
        with self.assertRaises(ValueError):
            drafting.apply_env(self.cfg, {**self.DOCKER, "DRAFTING": "dflash3"})


if __name__ == "__main__":
    unittest.main()
