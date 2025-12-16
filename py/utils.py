import sys
from collections.abc import Buffer
from typing import NoReturn, Protocol

import colored
from _typeshed import SupportsWrite

from py.cli import parse_file_path, parse_serial_path
from py.source import initiate_serial_connection

ERROR_RETURN = 2


def error(msg: str) -> NoReturn:
    print(colored.stylize(msg, colored.fore("red")))
    exit(ERROR_RETURN)


def warning(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("yellow")))


def success(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("green")))


def subtext(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("dark_gray")))


class Writeable(Protocol):
    def write(self, data: Buffer, /) -> int | None: ...


class WriteableAndCloseable(SupportsWrite, Protocol):
    def close(self) -> None: ...
