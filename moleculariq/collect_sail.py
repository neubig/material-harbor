"""Record only this adapter's Sailbox identities, never auth headers."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
names = {p.name for p in (ROOT / 'jobs').glob('*/*') if p.is_dir()}
boxes = []
for offset in range(0, 2000, 100):
    page = json.loads(subprocess.check_output([
        '/home/gneubig/.sail/bin/sail', 'box', 'list', '--json',
        '--search', 'task_000', '--limit', '100', '--offset', str(offset)]))['data']
    boxes.extend(page)
    if len(page) < 100:
        break

path = ROOT / 'sail-boxes.json'
existing = json.loads(path.read_text()) if path.exists() else {}
for box in boxes:
    name = box.get('name', '')
    if not any(name.startswith(n + '__') or name == n for n in names):
        continue
    fields = ('sailbox_id', 'id', 'name', 'app_id', 'app_name', 'image_id', 'status', 'created_at',
              'started_at', 'updated_at', 'error_message', 'vcpu_count', 'memory_mib', 'egress_policy')
    safe = {k: box[k] for k in fields if k in box}
    existing[name] = safe
path.write_text(json.dumps(existing, indent=2))
print('Recorded', len(existing), 'adapter Sailboxes')
