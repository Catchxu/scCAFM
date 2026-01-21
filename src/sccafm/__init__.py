from importlib.metadata import version as _version
__version__ = _version("scCAFM")

# Founctions
from .load import load_resources
from .trainer import sfm_trainer

# Modules
from .models import SFM
from .tokenizer import TomeTokenizer
from .loss import SFMLoss