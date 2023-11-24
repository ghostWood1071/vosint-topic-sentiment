import os
import sys

def get_running_path():
    script_path = os.path.abspath(sys.argv[0])
    script_directory = os.path.dirname(script_path)
    return script_directory