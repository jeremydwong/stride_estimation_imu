"""Live (streaming) processing of APDM Opal foot IMUs.

Everything live lives in this folder: README.md (design report + how to run),
bridge/ (Java bridge to the APDM SDK), scripts/ (command-line entry points).
"""
from .mechanize import LiveFoot, LiveFootfall, StrideBlock, run_live
from .sources import SampleBlock, Event, ReplaySource, BridgeSource, bridge_command, to_body
from .recorder import HdfRecorder
from .pipeline import LivePipeline
