#!/usr/bin/env bash
# Compatibility shim — Standard TPE launchers now live under standard_tpe/scripts/
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../standard_tpe/scripts/write_all_8gen_reports.sh" "$@"
