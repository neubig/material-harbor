"""Generate agent-only build contexts and isolated hidden verifier contexts."""
import json
import shutil
from pathlib import Path

from adapter import ROOT, digest

INSTRUCTION = '''# Unrelaxed crystal stability prediction

Predict whether the supplied initial (unrelaxed) periodic crystal will have a
DFT-relaxed MP2020-corrected energy on or below the fixed Materials Project DFT
reference convex hull. Stable means signed distance <= 0 eV/atom, including
negative distances. This is not stability of the initial geometry itself.

Inputs: `/data/structure.json` (lattice, species, fractional coordinates) and
`/data/reference.json` (fixed MP-only DFT hull reference for this composition).
A pinned CHGNet tool is installed: use `/opt/science/bin/python /opt/tools/predict.py
--help` for its arguments. CHGNet 0.3.0 energies already include MP2020 corrections;
do not apply those corrections a second time. Relax the initial structure before
comparing predicted total energy/atom to the fixed DFT hull total energy/atom.
Never build a hull from predicted energies. Tool nonconvergence is diagnostic,
not proof of either stability class. You may inspect the structure and tool code.

Write `/output/answer.json` as a JSON object with exactly one boolean field:
`{"stable": true}` or `{"stable": false}`. No extra fields or markdown.
The file must be a regular file, not a symlink. Missing/invalid answers score zero.
Do not retrieve benchmark labels/predictions, use source IDs, or call auxiliary
language models. Network access is restricted. You have a working budget of
40 turns. This task belongs to a balanced finite qualification, not the original
class-distribution accuracy or the official leaderboard.
'''


def generate():
    science = ROOT / 'science'
    required = [science / 'predict.py', science / 'Dockerfile', science / 'interface.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Scientific environment not frozen: {path}')
    interface = json.loads((science / 'interface.json').read_text())
    if not interface.get('ready_for_tasks'):
        raise ValueError('Science interface must explicitly declare ready_for_tasks')
    sample = json.loads((ROOT / 'private/sample.json').read_text())
    gold = json.loads((ROOT / 'private/gold.json').read_text())
    tasks = ROOT / 'tasks'
    if tasks.exists():
        raise FileExistsError('Tasks already exist; refusing to overwrite frozen tasks')
    for item in sample:
        if not (science / 'references' / f"{item['task_id']}.json").is_file():
            raise FileNotFoundError(f"Missing fixed DFT reference: {item['task_id']}")
    tasks.mkdir()
    manifest = {}
    for item in sample:
        task_id = item['task_id']
        task = tasks / task_id
        env = task / 'environment'
        (env / 'data').mkdir(parents=True)
        (task / 'tests').mkdir()
        (task / 'solution').mkdir()
        shutil.copy2(ROOT / 'private/inputs' / f'{task_id}.json', env / 'data/structure.json')
        shutil.copy2(science / 'references' / f'{task_id}.json', env / 'data/reference.json')
        for filename in interface['build_files']:
            source = science / filename
            if Path(filename).is_absolute() or not source.is_file() or source.is_symlink() or '..' in Path(filename).parts:
                raise ValueError(f'Unsafe build file: {filename}')
            destination = env / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (task / 'instruction.md').write_text(INSTRUCTION)
        (task / 'task.toml').write_text(f'''schema_version = "1.0"
artifacts = ["/output"]
[task]
name = "matbench-discovery/{task_id}"
[metadata]
dataset = "Matbench Discovery balanced qualification v1"
[agent]
timeout_sec = 3600.0
user = "agent"
network_mode = "allowlist"
allowed_hosts = ["llm-proxy.app.all-hands.dev"]
[environment]
cpus = 4
memory_mb = 8192
storage_mb = 16384
build_timeout_sec = 1800.0
network_mode = "allowlist"
allowed_hosts = ["llm-proxy.app.all-hands.dev"]
[verifier]
timeout_sec = 120.0
user = "root"
environment_mode = "separate"
network_mode = "no-network"
[verifier.environment]
cpus = 1
memory_mb = 512
storage_mb = 1024
build_timeout_sec = 600.0
network_mode = "no-network"
''')
        shutil.copy2(ROOT / 'verify.py', task / 'tests/verify.py')
        (task / 'tests/gold.json').write_text(json.dumps({'stable': gold[task_id]['stable']}))
        (task / 'tests/Dockerfile').write_text('FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c\nCOPY . /tests/\nRUN mkdir -p /output /logs/verifier\n')
        (task / 'tests/test.sh').write_text('#!/bin/sh\nexec /usr/local/bin/python -I /tests/verify.py\n')
        answer = json.dumps({'stable': gold[task_id]['stable']})
        (task / 'solution/solve.sh').write_text(f"#!/bin/sh\nmkdir -p /output\nprintf '%s\\n' '{answer}' > /output/answer.json\n")
        for path in (task / 'tests/test.sh', task / 'solution/solve.sh'):
            path.chmod(0o755)
        manifest[task_id] = {str(p.relative_to(task)): digest(p) for p in sorted(task.rglob('*')) if p.is_file()}
    (ROOT / 'task-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return len(manifest)


if __name__ == '__main__':
    print(generate())
