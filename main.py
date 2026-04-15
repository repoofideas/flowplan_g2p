#!/usr/bin/env python3
"""
FlowPlan-G2P Main Script

Alternative entry point that delegates to the package CLI.
Usage: python main.py <args>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flowplan_g2p.cli import main

if __name__ == "__main__":
    main()
