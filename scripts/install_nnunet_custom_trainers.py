"""
Copy the files in nnunet/custom_trainers/ into the nnU-Net environment so
nnU-Net can find the study-specific trainer classes by name.

nnU-Net is a plain pip install, so custom trainers must live inside the package
tree. This keeps the project copy as the source of truth and refreshes the venv
copy. Run from the repository root with the nnU-Net environment activated:

  python scripts/install_nnunet_custom_trainers.py
"""

from pathlib import Path
import shutil
import importlib.util

CUSTOM = Path(__file__).resolve().parents[1] / "nnunet" / "custom_trainers"
SRCS = ["nnUNetTrainerCF.py", "nnUNetTrainerCLDice.py"]   # all custom-loss trainer files

spec = importlib.util.find_spec("nnunetv2")
if spec is None:
    raise SystemExit("nnunetv2 not importable — run this with the nnU-Net venv's python.")
pkg = Path(spec.submodule_search_locations[0])
dst_dir = pkg / "training" / "nnUNetTrainer" / "variants" / "loss"

for name in SRCS:
    shutil.copyfile(CUSTOM / name, dst_dir / name)
    print(f"copied {name} -> {dst_dir / name}")

# Confirm nnU-Net can actually resolve the classes by name.
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
base = pkg / "training" / "nnUNetTrainer"
required = (
    "nnUNetTrainerCFv_100epochs",
    "nnUNetTrainerCFb_100epochs",
    "nnUNetTrainerCFvb_100epochs",
    "nnUNetTrainerDiceLoss_100epochs",
    "nnUNetTrainerCELoss_100epochs",
    "nnUNetTrainerCLDice_100epochs",
)
missing = []
for name in required:
    cls = recursive_find_python_class(str(base), name, "nnunetv2.training.nnUNetTrainer")
    print(f"  discover {name}: {'OK' if cls is not None else 'NOT FOUND'}")
    if cls is None:
        missing.append(name)

if missing:
    raise SystemExit("Custom trainer installation failed; not found: " + ", ".join(missing))

