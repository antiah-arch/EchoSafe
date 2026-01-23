# py/utils.py
from typing import NoReturn
import colored

ERROR_RETURN = 2

def error(msg: str) -> NoReturn:
    print(colored.stylize(msg, colored.fore("red")))
    raise SystemExit(ERROR_RETURN)

def warning(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("yellow")))

def success(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("green")))

def subtext(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("dark_gray")))
