from __future__ import annotations

from .common import context, parser
from src.phase2.token_validation import validate_token_interfaces


def main() -> None:
    args = parser("Validate selected token interfaces", "configs/phase2/token_interfaces.yaml").parse_args()
    root, config = context(args)
    print(validate_token_interfaces(root, config))


if __name__ == "__main__": main()
