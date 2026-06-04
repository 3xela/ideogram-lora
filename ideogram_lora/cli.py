"""``ilora <command>`` dispatches to a subcommand module's ``main()``.

Kept import-light so ``ilora --help`` is instant; the heavy module (torch,
ideogram4) is imported lazily only once a real command runs.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "inspect":  ("ideogram_lora.inspect_model", "Load the model; print structure + LoRA targets"),
    "dataset":  ("ideogram_lora.make_dataset", "Generate a claymation-penguin training set"),
    "train":    ("ideogram_lora.train_lora", "Train a LoRA adapter on an image+caption folder"),
    "sample":   ("ideogram_lora.sample", "Generate an image, optionally with a LoRA (--compare)"),
    "eval":     ("ideogram_lora.eval_lora", "Grid of base vs each checkpoint on two prompts"),
    "grid":     ("ideogram_lora.make_grid", "Stitch images into a labelled comparison grid"),
    "selftest": ("ideogram_lora.selftest", "CPU correctness checks (no weights needed)"),
}


def _usage() -> None:
    print("usage: ilora <command> [options]   (or: python main.py <command> ...)\n")
    print("commands:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<9} {desc}")
    print("\nrun 'ilora <command> --help' for that command's options")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"ilora: unknown command '{cmd}'\n")
        _usage()
        sys.exit(2)
    module = importlib.import_module(COMMANDS[cmd][0])
    # Hand the remaining argv to the module's own argparse.
    sys.argv = [f"ilora {cmd}", *rest]
    module.main()


if __name__ == "__main__":
    main()
