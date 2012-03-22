#@+leo-ver=5-thin
#@+node:michael.20120322193448.2263: * @thin ./setup.py
#@+others
#@+node:michael.20120322193448.2264: ** cx_freeze config
from cx_Freeze import setup, Executable
import sys

#@+others
#@+node:michael.20120322193448.2338: *3* project_config
project_version = '1.0.0'
project_name = 'pyLTEVisualizer'
project_owner = "Author: Ling Duo, Email: duo.ling.cn@gmail.com"
project_weblink = "http://code.google.com/p/pyltevisualizer/"
#@-others

EXCUTABLE_FILES = ( #"./Simulation/PCFICH/pcfich_gen_detect.py",
                                #"./Simulation/CSRS/csrs_gen_channel_estimation.py",
                                #"./Simulation/PBCH/pbch_gen_detect.py",
                                #"./Simulation/PSS/PSS_gen_detect.py",
                                #"./Simulation/SSS/sss_gen_detect.py"
                                "pyLTESimulator.py",
    )
INCLUDES = ['matplotlib', 'matplotlib.backends', 'matplotlib.backends.backend_qt4agg']
# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os"], "excludes": ["tkinter"], "include_files": [], "optimize":2, "includes":INCLUDES}
bdist_msi_options = {}
# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

#include-files = ( "config.txt" ,)
for script_name in EXCUTABLE_FILES:
    setup(
        name = "pyLTESimulator",
        version = project_version,
        description = "LTE PHY Simulator written in Python",
        options = {"build_exe": build_exe_options},
        executables = [Executable(script_name, base=base)]
        )

# setup(
	# name = "pyLTEVisualizer",
	# version = "1.0",
	# description = "LTE PHY Uu Visualizer written in Python",
	# options = {"bdist_msi": bdist_msi_options},
	# executables = [Executable("pyLTEVisualizer.py", base=base)]
	# )
#@-others
#@-leo
