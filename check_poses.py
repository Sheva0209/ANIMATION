import sys
sys.path.insert(0, 'src')
from script_generator import load_config
from renderer import load_character

cfg = load_config()
poses = cfg['pose_map']
print(f'Total poses in config: {len(poses)}')
print()

ok, fail = [], []
for name, data in poses.items():
    if data.get('is_blink_frame'):
        continue
    ptype = data.get("type", "?")
    try:
        img = load_character(name, cfg, 1080)
        ok.append(f'  OK   [{ptype:20s}] {name}  ({img.size[0]}x{img.size[1]})')
    except FileNotFoundError:
        fail.append(f'  MISS [{ptype:20s}] {name}')

for x in sorted(ok): print(x)
if fail:
    print()
    for x in sorted(fail): print(x)

print(f'\nResult: {len(ok)} OK, {len(fail)} missing')
