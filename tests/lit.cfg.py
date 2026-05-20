import os
from pathlib import Path

from lit.formats import ShTest

config.name = "mlir_lab"
config.test_format = ShTest()
config.suffixes = [".mlir"]

# Augment PATH so RUN: lines can reference tools by name without absolute paths.
#
# Under Bzlmod the llvm-project repo has a canonical name like
# "+llvm_configure+llvm-project" rather than plain "llvm-project". We match
# on the substring so this works regardless of the exact canonical prefix.
runfiles_dir = Path(os.environ["RUNFILES_DIR"])

tool_dirs = []
for subdir in runfiles_dir.iterdir():
    if "llvm-project" in subdir.name:
        for tool_subdir in ["mlir", "llvm"]:
            p = subdir / tool_subdir
            if p.exists():
                tool_dirs.append(str(p))

# Custom project tools (mlir-lab-opt and future additions).
tools_dir = runfiles_dir / "_main" / "tools"
if tools_dir.exists():
    tool_dirs.append(str(tools_dir))

config.environment["PATH"] = ":".join(tool_dirs) + ":" + os.environ["PATH"]

# %project_source_dir expands to the root of the project's runfiles tree.
# Use it in RUN: lines to reference source files by a stable path.
config.substitutions.append(
    ("%project_source_dir", str(runfiles_dir / "_main"))
)
