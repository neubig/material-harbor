"""Allowlisted native Harbor CSMBench smoke; no credentials in task uploads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import uuid

import run_modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/measurements/csmbench'
FILES = ['environment/Dockerfile', 'environment/data/image.jpg',
         'environment/data/question.json', 'instruction.md', 'solution/solve.sh',
         'task.toml', 'tests/label.json', 'tests/test.sh', 'tests/verify.py']
MODEL = 'deepseek/deepseek-v4-flash'


def smoke(agent):
    name = f'csm-{agent}-{uuid.uuid4().hex[:12]}'
    output = OUT / name
    output.mkdir(parents=True, mode=0o700)
    source = ROOT / 'csmbench/tasks/pilot10/csmbench-mcqa-0023'
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps(FILES, indent=2))
    for path in run_modal.task_files(source, manifest):
        target = output / 'task' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, target)
    args = argparse.Namespace(output=output, agent=agent, model=MODEL, timeout=600,
                              sandbox_timeout=1800, max_iterations=20,
                              app_name='material-harbor')
    env = run_modal.host_environment(Path.home() / '.env', agent)
    env['MODAL_ENVIRONMENT'] = 'paper2rlenv'
    env['LLM_BASE_URL'] = 'https://llm-proxy.app.all-hands.dev/v1'
    cmd = run_modal.command(args, output / 'task', name)
    if agent != 'oracle':
        cmd += ['--agent-kwarg', 'version=1.50.1']
    (output / 'pid').write_text(str(os.getpid()))
    print(str(output), flush=True)
    result = run_modal.run_trial(cmd, env, output, output / 'trials' / name / 'result.json')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('agent', choices=['oracle', 'openhands-sdk'])
    smoke(parser.parse_args().agent)
