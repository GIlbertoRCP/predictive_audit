"""Type stubs for the nanobind extension module built from native/python/module.cpp."""

from typing import TypedDict

class BuildInfo(TypedDict):
    version: str
    compiler: str
    build_type: str
    cplusplus: int
    hardware_threads: int

def version() -> str: ...
def build_info() -> BuildInfo: ...
