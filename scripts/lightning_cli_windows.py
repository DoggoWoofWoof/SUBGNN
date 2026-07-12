"""Windows shim for Lightning SDK CLI.

The current `lightning-sdk` CLI imports `simple_term_menu`, which depends on
`termios` and crashes on native Windows before non-interactive commands can run.
This shim provides the tiny object the SDK imports so CLI commands that do not
need an interactive menu can run from PowerShell.
"""

from __future__ import annotations

import sys
import types


class _TerminalMenu:
    def __init__(self, entries, *args, **kwargs):
        self.entries = list(entries)
        self.chosen_menu_index = 0

    def show(self):
        self.chosen_menu_index = 0
        return 0 if self.entries else None


fake_menu = types.ModuleType("simple_term_menu")
fake_menu.TerminalMenu = _TerminalMenu
sys.modules.setdefault("simple_term_menu", fake_menu)

from lightning_sdk.cli.entrypoint import main_cli


if __name__ == "__main__":
    main_cli()
