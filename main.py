"""
main.py — Entry point for Sheva Animation System

Usage:
    python main.py "topik video kamu"
    python main.py "tips nggak burnout saat UTS"
    python main.py "5 cara keluar dari comfort zone" --skip-tts --audio audio/existing.mp3
    python main.py --script scripts/existing.json
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

from pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sheva Animation Automation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "tips nggak burnout saat UTS"
  python main.py "cara manage waktu" --skip-tts --audio audio/cara_manage_waktu_full.mp3
  python main.py --script scripts/my_topic.json --skip-tts --audio audio/existing.mp3
        """,
    )

    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic for the animation video (in Bahasa Indonesia)",
    )

    # Optional overrides
    parser.add_argument(
        "--script",
        metavar="PATH",
        help="Use existing script JSON instead of generating new one",
    )
    parser.add_argument(
        "--audio",
        metavar="PATH",
        help="Use existing audio MP3 instead of generating TTS",
    )
    parser.add_argument(
        "--srt",
        metavar="PATH",
        help="Use existing SRT subtitle file",
    )

    # Skip flags
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS generation (requires --audio)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine topic
    topic = args.topic
    if not topic and not args.script:
        print("[ERROR] Provide a topic or --script path")
        print("   Example: python main.py \"tips nggak burnout saat UTS\"")
        sys.exit(1)

    if not topic:
        # Infer topic from script filename
        topic = Path(args.script).stem.replace("_", " ")

    # Validation
    if args.skip_tts and not args.audio and not args.script:
        print("[ERROR] --skip-tts requires --audio <path>")
        sys.exit(1)

    print(f"[Sheva] Starting Animation Pipeline")
    print(f"   Topic: {topic}")

    try:
        result = run_pipeline(
            topic=topic,
            skip_tts=args.skip_tts,
            script_path=args.script,
            audio_path=args.audio,
            srt_path=args.srt,
        )
        print("\nDone! Check your output folder:")
        print(f"   Landscape: {result['landscape']}")
        print(f"   Portrait:  {result['portrait']}")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
