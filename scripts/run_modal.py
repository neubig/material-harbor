"""Run independent trials of one fixed, explicitly allowlisted Harbor task on Modal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from dotenv import dotenv_values

CREDENTIALS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "LLM_API_KEY")


def task_files(task: Path, manifest: Path) -> list[Path]:
    names = json.loads(manifest.read_text())
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError("manifest must be a nonempty JSON list of file paths")
    paths = []
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("manifest paths must stay inside the task")
        if any(part.startswith(".") or part.lower() in {"credentials", "id_rsa", "id_ed25519"}
               for part in path.parts) or path.suffix.lower() in {".pem", ".key", ".env"}:
            raise ValueError("hidden or credential-like files are not allowed")
        if path.parts[0] not in {"task.toml", "instruction.md", "environment", "tests", "solution"}:
            raise ValueError("file outside Harbor task source allowlist")
        if any((task / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            raise ValueError("symlinks are not allowed")
        if not (task / path).is_file():
            raise ValueError("manifest entries must be regular files")
        paths.append(path)
    if not {Path("task.toml"), Path("instruction.md"), Path("tests/test.sh")} <= set(paths):
        raise ValueError("manifest must include task.toml, instruction.md, tests/test.sh")
    return sorted(set(paths))


def host_environment(env_file: Path | None, agent: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR", "SSL_CERT_FILE") if k in os.environ}
    if env_file is not None and not env_file.is_file():
        raise ValueError("credential file does not exist")
    values = dotenv_values(env_file, interpolate=False) if env_file else {}
    for key in CREDENTIALS:
        value = values.get(key) if key.startswith("MODAL_") else os.environ.get(key)
        if value and (agent != "oracle" or key != "LLM_API_KEY"):
            env[key] = value
    if agent != "oracle":
        if not env.get("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY is required for model trials")
    return env


def command(args, staged: Path, trial_name: str = "smoke") -> list[str]:
    cmd = [str(Path(sys.executable).with_name("harbor")), "trial", "start",
           "-p", str(staged), "-e", "modal", "-a", args.agent,
           "--trials-dir", str(args.output / "trials"), "--trial-name", trial_name,
           "--delete", "--agent-timeout", str(args.timeout),
           "--environment-kwarg", f"app_name={args.app_name}",
           "--environment-kwarg", f"sandbox_timeout_secs={args.sandbox_timeout}",
           "--environment-kwarg", f"sandbox_idle_timeout_secs={args.sandbox_timeout}"]
    if args.agent != "oracle":
        cmd += ["-m", args.model, "--agent-kwarg", "load_skills=false",
                "--agent-kwarg", f"max_iterations={args.max_iterations}",
                "--agent-kwarg", "temperature=0"]
    return cmd


def run_trial(cmd: list[str], env: dict[str, str], output: Path, result_path: Path) -> dict:
    (output / "command.json").write_text(json.dumps(cmd, indent=2))
    secrets = [env[k] for k in CREDENTIALS if env.get(k)]
    record = {"status": "infrastructure_failure", "process_exit_code": None,
              "result": str(result_path), "verifier_result": None, "exception_type": None}
    with (output / "runner.log").open("w") as log:
        try:
            with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                try:
                    for line in process.stdout:
                        for secret in secrets:
                            line = line.replace(secret, "[REDACTED]")
                        log.write(line)
                        log.flush()
                    record["process_exit_code"] = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    record["status"] = "interrupted"
                    record["process_exit_code"] = 130
        except OSError as exc:
            record["exception_type"] = type(exc).__name__
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
            record["verifier_result"] = result.get("verifier_result")
            record["exception_type"] = (result.get("exception_info") or {}).get("exception_type")
            if record["process_exit_code"] == 0 and not result.get("exception_info"):
                record["status"] = "completed"
        except (ValueError, OSError, AttributeError) as exc:
            record["exception_type"] = type(exc).__name__
    (output / "summary.json").write_text(json.dumps(record, indent=2))
    return record


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh local output directory")
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".env", help="read only Modal tokens; LLM_API_KEY comes from environment; never upload")
    parser.add_argument("--agent", choices=("oracle", "openhands-sdk"), default="openhands-sdk")
    parser.add_argument("--model", default="openai/deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://llm-proxy.app.all-hands.dev/v1")
    parser.add_argument("--app-name", default="material-harbor")
    parser.add_argument("--timeout", type=positive, default=180)
    parser.add_argument("--sandbox-timeout", type=positive, default=900)
    parser.add_argument("--max-iterations", type=positive, default=20)
    parser.add_argument("--repetitions", type=positive, default=1)
    parser.add_argument("--modal-environment", help="explicitly authorized Modal environment; no automatic fallback")
    parser.add_argument("--dry-run", action="store_true", help="validate sources without credentials or cloud calls")
    args = parser.parse_args()
    args.task = args.task.absolute()
    args.output = args.output.resolve()
    try:
        if args.task.is_symlink():
            raise ValueError("task root must not be a symlink")
        files = task_files(args.task, args.manifest)
        tomllib.loads((args.task / "task.toml").read_text())
        if args.output.exists() or args.output.is_relative_to(args.task.resolve()):
            raise ValueError("output must be fresh and outside source task")
        if args.dry_run:
            print(json.dumps([str(p) for p in files], indent=2))
            return 0
        env = host_environment(args.env_file, args.agent)
        if not all(env.get(key) for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")):
            raise ValueError("both Modal tokens are required in the credential file")
        if args.modal_environment:
            env["MODAL_ENVIRONMENT"] = args.modal_environment
        if args.agent != "oracle":
            env["LLM_BASE_URL"] = args.base_url
        args.output.mkdir(parents=True, mode=0o700)
        staged = args.output / "task"
        for path in files:
            target = staged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.task / path, target)
        (args.output / "manifest.json").write_text(json.dumps([str(p) for p in files], indent=2))
        (args.output / "source-hashes.json").write_text(json.dumps({
            str(path): hashlib.sha256((staged / path).read_bytes()).hexdigest()
            for path in files}, indent=2))
        summary = {"requested_trials": args.repetitions, "agent": args.agent,
                   "model": args.model if args.agent != "oracle" else None,
                   "base_url": args.base_url if args.agent != "oracle" else None,
                   "modal_environment": args.modal_environment,
                   "max_iterations": args.max_iterations, "load_skills": False,
                   "temperature": 0, "trials": []}
        for index in range(args.repetitions):
            name = "smoke" if args.repetitions == 1 else f"trial-{index + 1:04d}"
            output = args.output if args.repetitions == 1 else args.output / name
            output.mkdir(exist_ok=True, mode=0o700)
            record = run_trial(command(args, staged, name), env, output,
                               args.output / "trials" / name / "result.json")
            summary["trials"].append(record)
            (args.output / "batch-summary.json").write_text(json.dumps(summary, indent=2))
            if record["status"] != "completed":
                print("Stopped on infrastructure failure or interruption; retained all outputs", file=sys.stderr)
                return 130 if record["status"] == "interrupted" else 1
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Runner validation failed ({type(exc).__name__}); check paths, manifest, credentials", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
