"""CLI entrypoint for CutClipper and Auto Hype Clip."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.pipeline.jobs import JobManager
from app.schemas import ClipSettings, HypeMode, JobMode, JobStatus, LayoutMode


def main() -> None:
    parser = argparse.ArgumentParser(description="CutClipper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hype_parser = subparsers.add_parser("hype", help="Auto Hype Clip from live stream or VOD")
    hype_parser.add_argument("--url", required=True, help="YouTube live stream or VOD URL")
    hype_parser.add_argument(
        "--mode",
        choices=["record", "clip_now", "vod"],
        default="vod",
        help="Hype mode (record X minutes, clip now, or analyze VOD)",
    )
    hype_parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Recording duration in minutes (for record mode)",
    )
    hype_parser.add_argument(
        "--min-clip",
        type=float,
        default=30.0,
        help="Minimum clip length in seconds (default: 30)",
    )
    hype_parser.add_argument(
        "--max-clip",
        type=float,
        default=60.0,
        help="Maximum clip length in seconds (default: 60)",
    )
    hype_parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="Number of top hype moments to detect (default: 1)",
    )
    hype_parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Which ranked peak to export (default: 1)",
    )
    hype_parser.add_argument(
        "--layout",
        choices=["default", "facecam_top"],
        default="default",
        help="Layout mode: default (9:16 full-bleed) or facecam_top (Webcam top + Gameplay bottom)",
    )
    hype_parser.add_argument(
        "--webcam-region",
        type=str,
        default=None,
        help="Manual webcam overlay region as x,y,w,h (e.g. 1500,50,400,300)",
    )
    hype_parser.add_argument(
        "--webcam-height",
        type=int,
        default=600,
        help="Height in pixels for top webcam panel in 1080x1920 output (default: 600)",
    )
    hype_parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Vertical gap in pixels between webcam and gameplay panels (default: 0)",
    )
    hype_parser.add_argument(
        "--divider",
        action="store_true",
        help="Draw a divider line between webcam and gameplay panels",
    )
    hype_parser.add_argument(
        "--remove-webcam",
        action="store_true",
        help="Shift gameplay crop away from webcam overlay if overlapping",
    )
    hype_parser.add_argument(
        "--smart-center",
        action="store_true",
        help="Shift gameplay crop toward region with highest visual motion",
    )
    hype_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination directory to copy exported clip and JSON breakdown",
    )

    args = parser.parse_args()
    if args.command == "hype":
        asyncio.run(
            _run_cli_hype(
                url=args.url,
                mode=HypeMode(args.mode),
                duration=args.duration,
                min_clip=args.min_clip,
                max_clip=args.max_clip,
                top_n=args.top,
                rank=args.rank,
                layout=LayoutMode(args.layout),
                webcam_region=args.webcam_region,
                webcam_height=args.webcam_height,
                gap=args.gap,
                divider=args.divider,
                remove_webcam=args.remove_webcam,
                smart_center=args.smart_center,
                out_dir=args.out,
            )
        )


async def _run_cli_hype(
    *,
    url: str,
    mode: HypeMode,
    duration: float,
    min_clip: float,
    max_clip: float,
    top_n: int,
    rank: int,
    layout: LayoutMode = LayoutMode.DEFAULT,
    webcam_region: str | None = None,
    webcam_height: int = 600,
    gap: int = 0,
    divider: bool = False,
    remove_webcam: bool = False,
    smart_center: bool = False,
    out_dir: Path | None = None,
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    manager = JobManager(settings)
    await manager.start()

    print(f"🎬 Starting Auto Hype Clip for: {url}")
    print(
        f"   Mode: {mode.value} | Min: {min_clip}s | Max: {max_clip}s | Top N: {top_n} | Layout: {layout.value}"
    )

    clip_settings = ClipSettings()
    clip_settings.min_clip_seconds = int(min_clip)
    clip_settings.max_clip_seconds = int(max_clip)
    clip_settings.layout_mode = layout
    clip_settings.webcam_height = webcam_height
    clip_settings.gap = gap
    clip_settings.divider = divider
    clip_settings.remove_webcam_from_gameplay = remove_webcam
    clip_settings.smart_center = smart_center

    if webcam_region:
        from app.services.layout_facecam import parse_webcam_region

        parsed_box = parse_webcam_region(webcam_region)
        if parsed_box:
            clip_settings.webcam_region = list(parsed_box)
            clip_settings.auto_detect = False

    record = await manager.create_job(
        url,
        clip_settings,
        mode=JobMode.HYPE,
        hype_mode=mode,
        hype_record_minutes=duration,
        hype_top_n=top_n,
        hype_selected_rank=rank,
    )

    last_stage = None
    try:
        async for raw in manager.event_stream(record.id):
            if not raw.startswith("data: "):
                continue
            data = json.loads(raw[6:])
            stage = data.get("stage_label")
            progress = data.get("progress", 0)
            msg = data.get("message", "")
            if stage != last_stage or int(progress) % 15 == 0:
                print(f"[{progress:.0f}%] {stage}: {msg}")
                last_stage = stage

            if data.get("status") in {"completed", "failed", "cancelled"}:
                break

        final = manager.snapshot(record.id)
        if final.status == JobStatus.COMPLETED:
            print("\n✅ Auto Hype Clip Completed Successfully!")
            if final.clips:
                clip = final.clips[0]
                clip_rec = manager.clip(record.id, clip.id)
                print(f"   Video: {clip_rec.video_path}")
                print(f"   Duration: {clip.duration}s | Score: {clip.score}/100")

                outputs = record.job_dir / "outputs"
                json_files = list(outputs.glob("*_hype_*.json"))
                if json_files:
                    print(f"   Score Breakdown JSON: {json_files[0]}")
                    with open(json_files[0], encoding="utf-8") as f:
                        print(f.read())

                if out_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    import shutil

                    dest_video = out_dir / clip_rec.video_path.name
                    shutil.copy2(clip_rec.video_path, dest_video)
                    print(f"   Copied to: {dest_video}")
                    if json_files:
                        dest_json = out_dir / json_files[0].name
                        shutil.copy2(json_files[0], dest_json)
                        print(f"   Copied JSON to: {dest_json}")
        else:
            print(f"\n❌ Job failed: {final.error_message or final.message}")
            sys.exit(1)
    finally:
        await manager.stop()


if __name__ == "__main__":
    main()
