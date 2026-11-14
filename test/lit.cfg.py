import os

import lit.formats
import lit.llvm

lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config

config.name = "TileFlow"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.tileflow_binary_dir, "test")

config.environment["PATH"] = os.pathsep.join(
    [config.tileflow_tools_dir, config.llvm_tools_dir, config.environment.get("PATH", "")]
)

llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    [
        "tileflow-opt",
        "tileflow-tune",
        "mlir-opt",
        "mlir-translate",
        "FileCheck",
        "count",
        "not",
    ],
    [config.tileflow_tools_dir, config.llvm_tools_dir],
)
