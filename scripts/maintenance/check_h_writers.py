"""Read /proc for H users; pipe this source to python -B - <Linux project root>."""
import errno
import json
import os
from pathlib import Path
import posixpath
import re
import shlex
import sys


SCOPES = ('scripts/budget_free', 'configs/budget_free',
          'experiments/budget_free/value_v1', 'experiments/budget_free/value_diagnostics_v1',
          'logs/budget_free')


def match_path(value, scopes):
    if value.endswith(' (deleted)'):
        value = value[:-10]
    if not value.startswith('/'):
        return []
    value = posixpath.normpath(value)
    return [name for name, aliases in scopes.items()
            if any(value == alias or value.startswith(alias + '/') for alias in aliases)]


def command_matches(arguments, cwd, scopes):
    matches = []
    for argument in arguments:
        pieces = [argument]
        try:
            pieces.extend(shlex.split(argument))
        except ValueError:
            pass
        for piece in pieces:
            candidate = piece.split('=', 1)[-1]
            if cwd and not candidate.startswith('/') and '/' in candidate:
                candidate = posixpath.join(cwd, candidate)
            for scope in match_path(candidate, scopes):
                matches.append(dict(scope=scope, kind='cmdline', value=argument))
        for name, aliases in scopes.items():
            if any(re.search(r'(^|[\s=:\x22\x27(])' + re.escape(alias)
                             + r'(?=/|$|[\s\x22\x27)])', argument) for alias in aliases):
                matches.append(dict(scope=name, kind='cmdline', value=argument))
    return matches


def main():
    result = dict(status='BLOCKED', probe_pid=os.getpid(), hits=[], errors=[], vanished=[],
                  proc_scanned=0, ignored_pids=[os.getpid()])
    if len(sys.argv) != 2 or not sys.argv[1].startswith('/') or not Path('/proc').is_dir():
        result['errors'].append('Expected Linux /proc and one absolute project-root argument')
        print(json.dumps(result))
        return 1
    root = os.path.normpath(sys.argv[1])
    if root == '/' or not Path(root).is_dir():
        result['errors'].append('Project root is missing or is the filesystem root')
        print(json.dumps(result))
        return 1
    roots = {root, os.path.realpath(root)}
    scopes = {name: sorted(os.path.join(alias, name) for alias in roots) for name in SCOPES}
    result['scopes'] = scopes
    try:
        pids = sorted(path for path in Path('/proc').iterdir() if path.name.isdigit())
    except OSError as exc:
        result['errors'].append(dict(source='/proc', error=str(exc)))
        print(json.dumps(result))
        return 1
    for proc in pids:
        pid = int(proc.name)
        if pid == os.getpid():
            continue
        matches, arguments, cwd = [], [], None
        result['proc_scanned'] += 1

        def read(source, operation):
            try:
                return operation()
            except OSError as exc:
                if exc.errno in (errno.ENOENT, errno.ESRCH):
                    result['vanished'].append(dict(pid=pid, source=source))
                else:
                    result['errors'].append(dict(pid=pid, source=source, error=str(exc)))
                return None

        raw = read('cmdline', lambda: (proc / 'cmdline').read_bytes())
        if raw is not None:
            arguments = [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]
        cwd = read('cwd', lambda: os.readlink(proc / 'cwd'))
        if cwd:
            matches.extend(dict(scope=name, kind='cwd', value=cwd) for name in match_path(cwd, scopes))
        matches.extend(command_matches(arguments, cwd, scopes))
        descriptors = read('fd', lambda: list((proc / 'fd').iterdir()))
        for descriptor in descriptors or []:
            value = read('fd/' + descriptor.name, lambda: os.readlink(descriptor))
            if value:
                matches.extend(dict(scope=name, kind='fd/' + descriptor.name, value=value)
                               for name in match_path(value, scopes))
        maps = read('maps', lambda: (proc / 'maps').read_text(encoding='utf-8', errors='replace'))
        for line in (maps or '').splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6:
                value = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), fields[5])
                matches.extend(dict(scope=name, kind='maps', value=value)
                               for name in match_path(value, scopes))
        if matches:
            unique = {json.dumps(item, sort_keys=True): item for item in matches}
            result['hits'].append(dict(pid=pid, cmdline=arguments, cwd=cwd,
                                       matches=list(unique.values())))
    result['status'] = 'PASS' if not result['hits'] and not result['errors'] else 'BLOCKED'
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
