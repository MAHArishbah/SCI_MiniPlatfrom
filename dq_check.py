"""Standalone data quality check against the raw CSV files.

Usage:
    python dq_check.py --input ./data/raw --output ./artifacts/dq_report.json [--fail-on-critical]
"""

import sys

from sci.dq.runner import main

if __name__ == "__main__":
    sys.exit(main())
