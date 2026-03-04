import sys

from sklearn.linear_model import LogisticRegression  # FIX 6: train needs a fresh model

from cli import Args, Record, Review, Run, Train, parse_command_line
from recording import record
from trainer import train


def main() -> int:
    """
    Entry point for the echosafe CLI.
    Returns an exit code: 0 on success, 1 on handled error.
    """
    # FIX 5: top-level exception handler — all errors surface as clean messages
    try:
        return _run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


def _run() -> int:
    args: Args = parse_command_line()

    # Review doesn't need a source or output stream — handle before open_source()
    # so we don't accidentally try to open a serial port that isn't needed
    if isinstance(args.command, Review):
        from sounds_review import review
        review(args.command.directory, args.command.export)
        return 0

    # FIX 3 & 4: initialise to None so the finally block can safely check
    # whether open_source() / open_output() succeeded before closing them
    source = None
    output = None
    can_close_output = False

    try:
        source = args.open_source()
        output, can_close_output = args.open_output()

        match args.command:

            case command if isinstance(command, Train):
                # FIX 6: train command builds a fresh model — initialize_model()
                # loads an existing model for inference and is wrong here
                model = LogisticRegression(max_iter=1000)
                train(
                    model,
                    output,
                    command.window_size,
                    command.feature_count,
                    source.iterator,
                )

            case command if isinstance(command, Record):
                # FIX 1: record() is now a proper callable in recording.py
                # Pass port from CLI args so --port override is respected
                record(
                    port=args.port,
                    baud=args.baud,
                    duration_seconds=command.seconds,
                )

            case command if isinstance(command, Run):
                # FIX 2: pass --port and --verbose through to listener so they
                # are not silently dropped when run via the CLI
                from listener import main as run_listener
                argv_override = ["listener", "--port", args.port, "--baud", str(args.baud)]
                if args.verbose:
                    argv_override.append("--verbose")
                sys.argv = argv_override
                run_listener()

            case _:
                raise ValueError(f"Unknown command: {args.command!r}")

    finally:
        # FIX 3 & 4: guard against NameError if open_source/open_output raised
        if source is not None:
            source.close()
        if output is not None and can_close_output:
            output.close()

    return 0  # FIX 7: explicit success exit code


# FIX 8: propagate exit code from main() to the shell
if __name__ == "__main__":
    sys.exit(main())