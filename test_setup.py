import sys
sys.path.insert(0, 'src')

print('Testing imports...')

# Test config loading
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
n_poses = len(cfg['pose_map'])
n_bgs = len(cfg['background_map'])
print(f'  [OK] config.yaml loaded -- {n_poses} poses, {n_bgs} backgrounds')

# Test module imports
from script_generator import load_config, build_system_prompt
print('  [OK] script_generator imports OK')

from tts_generator import load_config as lc2
print('  [OK] tts_generator imports OK')

from transcriber import parse_srt, wrap_srt_lines
print('  [OK] transcriber imports OK')

from renderer import load_config as lc3, load_font, load_background, load_character
print('  [OK] renderer imports OK')

# Test font loading
from PIL import ImageFont
font = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 52)
print(f'  [OK] Font loaded OK')

# Test background loading
bg_land = load_background('kamar', cfg, 1920, 1080)
print(f'  [OK] Background landscape: {bg_land.size}')

bg_port = load_background('kampus', cfg, 1080, 1920)
print(f'  [OK] Background portrait: {bg_port.size}')

# Test character loading
char = load_character('sheva_waving and say hi', cfg, 1080)
print(f'  [OK] Character loaded: {char.size}')

char2 = load_character('halfbody explaining', cfg, 1080)
print(f'  [OK] Character halfbody: {char2.size}')

print()
print('All checks passed! System is ready.')
print('Run with: python main.py "topik video kamu"')
