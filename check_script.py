import json, sys

with open("scripts/tips_nggak_burnout_saat_uts.json", encoding="utf-8") as f:
    s = json.load(f)

total = sum(sc["estimated_duration"] for sc in s["scenes"])
total_frames = int(total * 30)
print(f"Scenes: {len(s['scenes'])}, Total: ~{total:.0f}s, ~{total_frames} frames per orientation")
print()
for sc in s["scenes"]:
    emotion = sc.get("subtitle_emotion", "?")
    has_gen = bool(sc.get("generated_scene"))
    narr = sc["narration"][:55]
    print(f"  Scene {sc['id']:2d}: emotion={emotion!r:12s} | ai_gen={has_gen} | {narr}...")
