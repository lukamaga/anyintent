"""The console script anyintent-serve. It is installed with the library even without the 'server' extra, so it imports
the HTTP server only when it runs: without FastAPI and uvicorn it prints the install line and exits with code 2 instead
of a traceback."""
import sys


def main(argv=None):
    try:
        from .server import main as run
    except ImportError as e:
        print(f"anyintent-serve: {e}", file=sys.stderr)
        return 2
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
