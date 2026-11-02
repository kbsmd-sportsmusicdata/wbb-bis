"""Command entry point; importing/help performs no collection."""
import argparse
from pathlib import Path

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Recurring NCAA WBB scouting: NET top 50 plus ACC")
    result.add_argument("command", choices=("audit", "refresh", "build", "pilot"))
    result.add_argument("--season", type=int, default=None)
    result.add_argument("--repository", default="kbsmd-sportsmusicdata/wbb-bis")
    result.add_argument("--state-dir", type=Path, default=Path("data/scouting"))
    result.add_argument("--now", help="Explicit ISO timestamp for reproducible verification")
    result.add_argument("--trigger", choices=("manual", "primary", "retry"), default="manual")
    result.add_argument("--publish", action="store_true", help="Publish verified run to GitHub Releases")
    result.add_argument("--local-input", type=Path, help="Use explicitly supplied local snapshots")
    result.add_argument("--pilot-config", type=Path, help="Explicit six-team historical pilot configuration")
    result.add_argument("--net-dir", type=Path, help="Reviewed official NET snapshot folder")
    return result

def main(argv=None) -> int:
    command_parser=parser()
    args = command_parser.parse_args(argv)
    args.season=args.season or (2026 if args.command=="pilot" else 2027)
    if args.command=="pilot":
        if args.publish:command_parser.error("The historical pilot cannot publish into production")
        from .pilot import run_pilot_command
        return run_pilot_command(args)
    from .runner import run_command
    return run_command(args)
