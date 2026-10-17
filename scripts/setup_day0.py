#!/usr/bin/env python3
"""Clone the reviewed upstream revision and apply the minimal optional-import patch."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / 'third_party' / 'day0_support_resistance'
REVISION = 'fdd4a7f0fdccf6c4007da79fd6f3879164462c3c'

def main():
    if not TARGET.exists():
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/day0market/support_resistance.git', str(TARGET)], check=True)
        subprocess.run(['git', '-C', str(TARGET), 'checkout', '--detach', REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(TARGET), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f'Expected upstream {REVISION}, got {actual}; choose revision explicitly before updating')
    source = TARGET / 'pricelevels' / '_abstract.py'
    text = source.read_text()
    if '\nfrom zigzag import peak_valley_pivots\n' in text:
        text = text.replace('\nfrom zigzag import peak_valley_pivots\n', '\n')
        text = text.replace('        pivots = peak_valley_pivots(', '        from zigzag import peak_valley_pivots\n        pivots = peak_valley_pivots(')
        source.write_text(text)
    print(f'day0 ready: {TARGET} @ {REVISION}')

if __name__ == '__main__': main()
