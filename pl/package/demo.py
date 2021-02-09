
import os
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

print(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import lib
from lib.m1 import ops