# py/utils.py
from typing import NoReturn
import colored

def error(msg: str) -> NoReturn:
    raise RuntimeError(msg)

def warning(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("yellow")))

def success(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("green")))

def subtext(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("dark_gray")))