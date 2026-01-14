import csv
from collections.abc import Iterator

# from _typeshed import SupportsWrite
from source import MIC_VALUE_LABEL, QUANTITY_LABEL, TIME_LABEL, DataEntry
from utils import Writeable, success

# def record(path: str, data: Iterator[DataEntry], seconds: int) -> None:
#     with open(path, "w") as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow([TIME_LABEL, MIC_VALUE_LABEL, QUANTITY_LABEL])
#         print(f"recording for {seconds} seconds")
#         for entry in data:
#             writer.writerow([entry.time, entry.microphone, entry.quantity])
#         success(f"finished recording to {path}")


def record(output: Writeable, data: Iterator[DataEntry], seconds: int) -> None:
    writer = csv.writer(output)  # type: ignore stupid python


# def record_serial(path: str, com_port: str, seconds: int) -> None:
#     with initiate_serial_connection(com_port) as serial_connection:
#         record(path, open_serial_data(serial_connection), seconds)
