import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.run_modal import command, host_environment, task_files, run_trial


class ModalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = self.root / "task"
        self.names = ["task.toml", "instruction.md", "tests/test.sh", "environment/Dockerfile"]
        for name in self.names:
            path = self.task / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.names))

    def test_explicit_files_only(self):
        (self.task / "environment/unlisted.txt").write_text("not uploaded")
        self.assertEqual(task_files(self.task, self.manifest), sorted(map(Path, self.names)))

    def test_unsafe_sources(self):
        for name in ["../secret", "/etc/passwd", "environment/.env", "environment/token.key", "other.txt", "environment/missing"]:
            with self.subTest(name=name):
                self.manifest.write_text(json.dumps(self.names + [name]))
                with self.assertRaises(ValueError):
                    task_files(self.task, self.manifest)

    def test_symlink_rejected(self):
        (self.task / "environment/link").symlink_to(self.task / "task.toml")
        self.manifest.write_text(json.dumps(self.names + ["environment/link"]))
        with self.assertRaises(ValueError):
            task_files(self.task, self.manifest)

    def test_missing_required(self):
        self.manifest.write_text('["instruction.md"]')
        with self.assertRaises(ValueError):
            task_files(self.task, self.manifest)

    def test_sdk_credentials_not_in_command(self):
        # Synthetic credentials exercise the actual dotenv loader, not a mock.
        env_file = self.root / "credentials.env"
        env_file.write_text("LLM_API_KEY=test-only\nUNRELATED_PASSWORD=excluded\n")
        child_env = dict(os.environ, LLM_API_KEY="test-only")
        code = ("import json; from pathlib import Path; "
                "from scripts.run_modal import host_environment; "
                f"print(json.dumps(host_environment(Path({str(env_file)!r}), 'openhands-sdk')))")
        result = subprocess.run([sys.executable, "-c", code], env=child_env,
                                capture_output=True, text=True, check=True)
        env = json.loads(result.stdout)
        self.assertEqual(env["LLM_API_KEY"], "test-only")
        self.assertNotIn("UNRELATED_PASSWORD", env)
        self.assertNotIn("LLM_API_KEY", host_environment(env_file, "oracle"))
        args = argparse.Namespace(output=self.root, agent="openhands-sdk", timeout=60,
                                  sandbox_timeout=300, app_name="test", model="openai/test",
                                  base_url="https://example.com/v1", max_iterations=2)
        cmd = command(args, self.task)
        self.assertNotIn("--ae", cmd)
        self.assertNotIn("test-only", " ".join(cmd))
        self.assertIn("max_iterations=2", cmd)
        self.assertIn("load_skills=false", cmd)
        self.assertEqual(cmd[cmd.index("-a") + 1], "openhands-sdk")

    def test_dotenv_modal_tokens(self):
        env_file = self.root / "credentials.env"
        env_file.write_text("MODAL_TOKEN_ID=local-id\nMODAL_TOKEN_SECRET=local-secret\n")
        env = host_environment(env_file, "oracle")
        self.assertEqual(env["MODAL_TOKEN_ID"], "local-id")
        self.assertEqual(env["MODAL_TOKEN_SECRET"], "local-secret")

    def test_real_child_failure_is_retained_and_redacted(self):
        output = self.root / "trial"
        output.mkdir()
        record = run_trial([sys.executable, "-c",
                            "print('test-sensitive'); raise SystemExit(3)"],
                           {"LLM_API_KEY": "test-sensitive"}, output,
                           output / "result.json")
        self.assertEqual(record["process_exit_code"], 3)
        self.assertEqual(record["status"], "infrastructure_failure")
        self.assertIsNone(record["verifier_result"])
        self.assertIn("[REDACTED]", (output / "runner.log").read_text())
        self.assertNotIn("test-sensitive", (output / "runner.log").read_text())
        self.assertEqual(json.loads((output / "summary.json").read_text()), record)

    def test_zero_reward_is_completed_not_infrastructure_failure(self):
        output = self.root / "trial"
        output.mkdir()
        result = output / "result.json"
        result.write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0}},
                                      "exception_info": None}))
        record = run_trial([sys.executable, "-c", "pass"], {}, output, result)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["verifier_result"]["rewards"]["reward"], 0)

    def test_missing_result_and_spawn_failure(self):
        for cmd in ([sys.executable, "-c", "pass"], ["/nonexistent/harbor"]):
            record = run_trial(cmd, {}, self.root, self.root / "absent.json")
            self.assertEqual(record["status"], "infrastructure_failure")

    def test_explicit_trial_names(self):
        args = argparse.Namespace(output=self.root, agent="oracle", timeout=60,
                                  sandbox_timeout=300, app_name="test")
        first = command(args, self.task, "trial-0001")
        second = command(args, self.task, "trial-0002")
        self.assertEqual(second[second.index("--trial-name") + 1], "trial-0002")
        self.assertNotEqual(first, second)

    def test_real_dry_run(self):
        out = self.root / "output"
        result = subprocess.run([sys.executable, "scripts/run_modal.py", "--task", str(self.task),
                                 "--manifest", str(self.manifest), "--output", str(out),
                                 "--env-file", "/nonexistent", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(json.loads(result.stdout)), set(self.names))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
