from cli import Args, Record, Run, parse_command_line
from recording import record
from trainer import initialize_model, train


def main():
    args: Args = parse_command_line()
    source = args.open_source()
    output, can_close_output = args.open_output()
    match args.command:
        case command if isinstance(command, Run):
            interpreter = initialize_model(args.model)

            train(
                interpreter,
                output,
                command.window_size,
                command.feature_count,
                source.iterator,
            )
            # open source Iterator[DataEntry]
            # pass to training function
            # finish
        case command if isinstance(command, Record):
            record(output, source.iterator, command.seconds)
            pass
            # pass to recorder function
            # finish
    source.close()
    if can_close_output:
        output.close()
    # interpretor = initialize_model(args.model)
