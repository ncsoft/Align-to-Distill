# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3 -u
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""

from fairseq_cli.train import cli_main


if __name__ == "__main__":
    cli_main()
