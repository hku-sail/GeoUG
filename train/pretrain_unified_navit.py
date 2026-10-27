"""Compatibility entry point for the former SFT training command.

New integrations should run ``python -m train.sft.train``.
"""

from train.sft.train import main


if __name__ == "__main__":
    main()
