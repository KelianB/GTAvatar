import sys
sys.path.insert(1, "submodules")

from .types import AvatarOutput, RenderSettings
from .arguments import create_parser, parse_args
from .avatar import Avatar
