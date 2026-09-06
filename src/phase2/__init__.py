"""Phase 2: frozen-expert token bridges, reranking, and dense teaching.

Phase 2 consumes the immutable Phase 1.5 selection.  It deliberately contains
no Phase 3 router or Phase 4 controller.
"""

from .bridge_model import FrozenPairBridge
from .cross_attention import TextToVisionBridge
from .schemas import RunStatus, TokenBatch

__all__ = ["FrozenPairBridge", "RunStatus", "TextToVisionBridge", "TokenBatch"]
