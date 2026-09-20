"""Small, lazy command router: native commands never import reference tools."""
import argparse
from importlib import import_module
import sys

COMMANDS = {
    'fgm': 'kava.fgm.generate',
    'export': 'kava.fgm.export',
    'plot': 'kava.fgm.plot',
    'refine-table': 'kava.fgm.refine',
    'validate-table': 'kava.fgm.validate',
    'soret': 'kava.benchmarks.soret',
    'compare': 'kava.reference.compare',
    'reference-fgm': 'kava.reference.generate_fgm',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='KAVA native flame solver and FGM tools')
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Use COMMAND --help for its options')
    args = parser.parse_args(argv)
    original = sys.argv
    try:
        sys.argv = ['kava ' + args.command, *args.arguments]
        import_module(COMMANDS[args.command]).main()
    finally:
        sys.argv = original
