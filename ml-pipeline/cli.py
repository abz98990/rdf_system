"""Single front door for the whole pipeline. Every underlying script still
works standalone (python -m src.data_ingestion, etc. — see README) — this
just wraps them so there's one command to remember.

    python cli.py download-data   # fetch the M5 dataset (no Kaggle needed)
    python cli.py train           # ingest -> features -> baseline -> LSTM
    python cli.py evaluate        # like-for-like scoring of all models
    python cli.py report          # regenerate plots/metrics from the last run
    python cli.py serve           # start the FastAPI prediction server
    python cli.py demo            # live terminal simulation (see README)
    python cli.py test            # run the test suite
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def cmd_download_data(args):
    from scripts.download_m5 import download_m5

    download_m5()


def cmd_train(args):
    from src.train_pipeline import run

    run()


def cmd_evaluate(args):
    from scripts.final_evaluation import run

    run()


def cmd_report(args):
    from scripts.generate_report_assets import run

    run()


def cmd_serve(args):
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=args.port, reload=args.reload)


def cmd_demo(args):
    from scripts.live_demo import run_demo

    run_demo(n_items=args.items, speed=args.speed, port=args.port, days=args.days)


def cmd_test(args):
    import pytest

    raise SystemExit(pytest.main([str(ROOT_DIR / "tests"), "-v"]))


def main():
    parser = argparse.ArgumentParser(prog="cli.py", description="M5 retail demand forecasting pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download-data", help="Download the M5 dataset (Kaggle-free GitHub mirror).").set_defaults(func=cmd_download_data)
    sub.add_parser("train", help="Run the full pipeline and save trained models.").set_defaults(func=cmd_train)
    sub.add_parser("evaluate", help="Score every model on identical held-out rows (like-for-like).").set_defaults(func=cmd_evaluate)
    sub.add_parser("report", help="Regenerate report plots/metrics from the last training run.").set_defaults(func=cmd_report)

    serve_p = sub.add_parser("serve", help="Start the FastAPI prediction server.")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only).")
    serve_p.set_defaults(func=cmd_serve)

    demo_p = sub.add_parser("demo", help="Live terminal simulation against real held-out test-period data.")
    demo_p.add_argument("--items", type=int, default=3, help="How many products to simulate.")
    demo_p.add_argument("--speed", type=float, default=0.6, help="Seconds paused between simulated days.")
    demo_p.add_argument("--port", type=int, default=8000)
    demo_p.add_argument("--days", type=int, default=None, help="Trailing days to simulate (default: TEST_DAYS).")
    demo_p.set_defaults(func=cmd_demo)

    sub.add_parser("test", help="Run the test suite.").set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # Python already puts this script's own directory (ml-pipeline/) at the
    # front of sys.path when run directly, so `src`/`api`/`scripts`/`config`
    # import correctly regardless of the caller's working directory.
    main()
