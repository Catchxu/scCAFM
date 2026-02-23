from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("scCAFM")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Founctions
from .load import load_resources
from .trainer import sfm_trainer

# Modules
from .models import SFM
from .tokenizer import TomeTokenizer
from .loss import SFMLoss
