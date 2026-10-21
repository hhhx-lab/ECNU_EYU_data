"""Exercise the real Bash launcher using simulated training processes, without a GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


MODS = ("t1c", "t1n", "t2w", "t2f")


class TrainLauncherTests(unittest.TestCase):
    def setUp(self):
        self.bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not self.bash or not Path(self.bash).is_file():
            self.skipTest("Bash unavailable")
        self.temporary = tempfile.TemporaryDirectory(prefix="train-launcher-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.events = self.root / "events"
        self.events.mkdir()
        # Records "start <mod>" / "end <mod>" per GPU so tests can check ordering and overlap.
        # WAIT_FOR="a:b" makes modality a wait (bounded) until b has started, to prove concurrency.
        simulated_python = (
            '#!/usr/bin/env bash\n'
            'mod=""\n'
            'while [[ $# -gt 0 ]]; do\n'
            '  if [[ "$1" == "--modality" ]]; then mod="$2"; shift 2; else shift; fi\n'
            'done\n'
            'events="$EVENT_DIR/gpu_${CUDA_VISIBLE_DEVICES:-unset}.log"\n'
            'echo "start $mod" >> "$events"\n'
            'touch "$EVENT_DIR/started_$mod"\n'
            'if [[ -n "${WAIT_FOR:-}" && "${WAIT_FOR%%:*}" == "$mod" ]]; then\n'
            '  other="${WAIT_FOR#*:}"\n'
            '  for _ in $(seq 1 100); do [[ -e "$EVENT_DIR/started_$other" ]] && break; sleep 0.05; done\n'
            '  if [[ -e "$EVENT_DIR/started_$other" ]]; then r=met; else r=timeout; fi\n'
            '  echo "$r $other" >> "$EVENT_DIR/rendezvous.log"\n'
            'fi\n'
            'sleep "${JOB_SECONDS:-0}"\n'
            'echo "end $mod" >> "$events"\n'
            'echo "simulated training completed: $mod"\n'
            'if [[ "${FAIL_MODALITY:-}" == "$mod" || "${FAIL_MODALITY:-}" == "all" ]]; then exit 17; fi\n'
            'exit 0\n')
        with open(self.bin / "python", "w", encoding="utf-8", newline="\n") as stream:
            stream.write(simulated_python)
        self.launcher = Path(__file__).absolute().parents[1] / "scripts/train_4gpu.sh"

    def run_launcher(self, failing="", gpus="0 1 2 3", invocation='exec bash "$LAUNCHER"', **extra_env):
        env = {**os.environ, "MOCK_BIN": self.bin.as_posix(), "LAUNCHER": self.launcher.as_posix(),
               "FAIL_MODALITY": failing, "GPUS": gpus, "EVENT_DIR": self.events.as_posix(),
               **extra_env}
        return subprocess.run(
            [self.bash, "-c", 'mock_bin=$(cd "$MOCK_BIN" && pwd); '
             'export PATH="$mock_bin:$PATH"; chmod +x "$mock_bin/python"; ' + invocation],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=30)

    def wait_for_status_files(self, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if all((self.root / f"logs/train_{mod}.status").is_file() for mod in MODS):
                return
            time.sleep(0.1)

    def gpu_events(self, gpu):
        path = self.events / f"gpu_{gpu}.log"
        return path.read_text().split("\n")[:-1] if path.exists() else []

    def status(self, mod):
        return (self.root / f"logs/train_{mod}.status").read_text().strip()

    # ---- default layout: four GPUs, one modality each (unchanged behaviour) ----

    def test_all_four_succeed(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("All four finished successfully.", result.stdout)
        self.assertEqual(result.stdout.count("finished successfully (exit 0)"), 4)
        for gpu, mod in zip("0123", MODS):
            self.assertEqual(self.gpu_events(gpu), [f"start {mod}", f"end {mod}"])

    def test_one_failure_is_propagated_and_other_jobs_checked(self):
        for modality in ("t1c", "t2w", "t2f"):
            with self.subTest(modality=modality):
                result = self.run_launcher(modality)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"{modality} FAILED (exit 17)", result.stderr)
                self.assertEqual(result.stdout.count("finished successfully (exit 0)"), 3)
                self.assertNotIn("All four finished successfully.", result.stdout)
                for mod in MODS:
                    self.assertIn(f"completed: {mod}", (self.root / f"logs/train_{mod}.log").read_text())

    def test_all_failures_are_reported(self):
        result = self.run_launcher("all")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr.count("FAILED (exit 17)"), 4)

    def test_slots_on_different_gpus_run_concurrently(self):
        # t1c (GPU 0) waits for t1n (GPU 1) to start; only parallel slots let that happen in time.
        result = self.run_launcher(WAIT_FOR="t1c:t1n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.events / "rendezvous.log").read_text().strip(), "met t1n")

    # ---- serial layouts ----

    def test_single_gpu_trains_all_four_in_series(self):
        result = self.run_launcher(gpus="0", JOB_SECONDS="0.2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 parallel slot(s)", result.stdout)
        self.assertIn("slot 1 on GPU 0: t1c t1n t2w t2f", result.stdout)
        # No overlap on the single GPU, and the queue keeps modality order.
        self.assertEqual(self.gpu_events("0"),
                         [f"{edge} {mod}" for mod in MODS for edge in ("start", "end")])
        self.assertIn("All four finished successfully.", result.stdout)
        for mod in MODS:
            self.assertEqual(self.status(mod), "0")

    def test_two_gpus_split_round_robin_and_serialize_per_gpu(self):
        result = self.run_launcher(gpus="0 1", JOB_SECONDS="0.2", WAIT_FOR="t1c:t1n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("slot 1 on GPU 0: t1c t2w", result.stdout)
        self.assertIn("slot 2 on GPU 1: t1n t2f", result.stdout)
        self.assertEqual(self.gpu_events("0"), ["start t1c", "end t1c", "start t2w", "end t2w"])
        self.assertEqual(self.gpu_events("1"), ["start t1n", "end t1n", "start t2f", "end t2f"])
        # The two GPUs still work at the same time.
        self.assertEqual((self.events / "rendezvous.log").read_text().strip(), "met t1n")
        self.assertEqual(result.stdout.count("finished successfully (exit 0)"), 4)

    def test_serial_failure_does_not_stop_the_rest_of_the_queue(self):
        result = self.run_launcher("t1n", gpus="0")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("t1n FAILED (exit 17)", result.stderr)
        self.assertEqual(result.stdout.count("finished successfully (exit 0)"), 3)
        self.assertNotIn("All four finished successfully.", result.stdout)
        self.assertEqual(self.gpu_events("0"),
                         [f"{edge} {mod}" for mod in MODS for edge in ("start", "end")])
        self.assertEqual([self.status(mod) for mod in MODS], ["0", "17", "0", "0"])

    def test_two_slots_may_share_one_gpu(self):
        result = self.run_launcher(gpus="0 0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("slot 1 on GPU 0: t1c t2w", result.stdout)
        self.assertIn("slot 2 on GPU 0: t1n t2f", result.stdout)
        self.assertEqual(sorted(self.gpu_events("0")),
                         sorted(f"{edge} {mod}" for mod in MODS for edge in ("start", "end")))
        self.assertIn("All four finished successfully.", result.stdout)

    # ---- a serial queue must outlive the terminal it was launched from ----

    def test_serial_queue_survives_closed_output(self):
        # The reader stops once training has started, as a lost terminal would; every later
        # progress message from the queue then hits a closed pipe.
        reader = ('bash "$LAUNCHER" | { while IFS= read -r line; do '
                  '[[ $line == "started t1c"* ]] && break; done; }')
        self.run_launcher(gpus="0", JOB_SECONDS="0.2", invocation=reader)
        self.wait_for_status_files()
        self.assertEqual([self.status(mod) for mod in MODS], ["0", "0", "0", "0"])
        self.assertEqual(self.gpu_events("0"),
                         [f"{edge} {mod}" for mod in MODS for edge in ("start", "end")])

    def test_serial_queue_survives_hangup_of_its_process_group(self):
        # A terminal hangup signals the launcher's whole process group; job control gives the
        # launcher its own group so the signal reaches it, its slots and the training processes.
        hangup = (
            'set -m; bash "$LAUNCHER" > launcher.out 2>&1 & pg=$!; '
            'for _ in $(seq 1 100); do [[ -e "$EVENT_DIR/started_t1c" ]] && break; sleep 0.05; done; '
            'kill -HUP -- -"$pg"; wait "$pg"; exit 0')
        self.run_launcher(gpus="0", JOB_SECONDS="0.3", invocation=hangup)
        self.wait_for_status_files()
        self.assertEqual([self.status(mod) for mod in MODS], ["0", "0", "0", "0"])

    def test_stale_status_from_a_previous_run_is_not_reported(self):
        (self.root / "logs").mkdir()
        (self.root / "logs/train_t2f.status").write_text("0\n")
        result = self.run_launcher("t2f", gpus="0")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.status("t2f"), "17")

    def test_launcher_has_unix_line_endings(self):
        # A CR before each newline breaks the script under Linux bash (set: pipefail<CR>: invalid option name).
        self.assertNotIn(b"\r", self.launcher.read_bytes())

    # ---- invalid configuration ----

    def test_invalid_gpu_count_fails_before_starting_jobs(self):
        for gpus in (" ", "0 1 2 3 4"):
            with self.subTest(gpus=gpus):
                result = self.run_launcher(gpus=gpus)
                self.assertEqual(result.returncode, 2)
                self.assertIn("between 1 and 4 GPU IDs", result.stderr)
                self.assertNotIn("started", result.stdout)
                self.assertFalse((self.root / "logs").exists())


if __name__ == "__main__":
    unittest.main()
