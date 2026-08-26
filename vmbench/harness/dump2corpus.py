#!/usr/bin/env python3
"""Split a LIVEDUMP block into a wolfram/corpus/<case>/ directory.

  dump2corpus.py <dump.txt> <corpus-dir>

The dump carries @entry/@cfg/@defs/@uses/@phis/@liveout sections in exactly
the shapes ImportCFG reads; this just splits them into the six files.
"""
import os
import sys


def main(src, outdir):
    sections = {}
    cur = None
    with open(src) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('LIVEDUMP-BEGIN') or line.startswith('LIVEDUMP-END'):
                continue
            if line.startswith('@'):
                cur = line[1:]
                sections[cur] = []
                continue
            if cur is not None:
                sections[cur].append(line)
    want = ['entry', 'cfg', 'defs', 'uses', 'phis', 'liveout']
    missing = [w for w in want if w not in sections]
    if missing:
        sys.exit('missing sections: %s' % missing)
    os.makedirs(outdir, exist_ok=True)
    for w in want:
        with open(os.path.join(outdir, w + '.txt'), 'w') as f:
            f.write('\n'.join(sections[w]))
            if sections[w]:
                f.write('\n')
    print('wrote %s: %s' % (outdir,
          {w: len(sections[w]) for w in want}))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
