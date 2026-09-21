"""Small, lazy command router: native commands never import reference tools."""
import argparse
from importlib import import_module
import sys

COMMANDS = {
    'fgm': 'kflame.fgm.generate',
    'export': 'kflame.fgm.export',
    'plot': 'kflame.fgm.plot',
    'refine-table': 'kflame.fgm.refine',
    'validate-table': 'kflame.fgm.validate',
    'soret': 'kflame.benchmarks.soret',
    'compare': 'kflame.reference.compare',
    'reference-fgm': 'kflame.reference.generate_fgm',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='KFLAME native flame solver and FGM tools')
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Use COMMAND --help for its options')
    args = parser.parse_args(argv)
    original = sys.argv
    try:
        sys.argv = ['kflame ' + args.command, *args.arguments]
        import_module(COMMANDS[args.command]).main()
    finally:
        sys.argv = original
