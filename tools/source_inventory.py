"""Source inventory shared by artifact collectors after library/CLI separation."""
from pathlib import Path


def source_files(root):
    root = Path(root)
    paths = [root/'CMakeLists.txt', root/'sources.lock.json', root/'tokenizer/schema3_contract.json']
    paths.extend(p for p in (root/'CMakePresets.json',) if p.is_file())
    paths.extend(p for p in (root/'.github/workflows').glob('*')
                 if p.is_file() and p.suffix in ('.yml', '.yaml'))
    for directory in ('include', 'cli', 'src', 'cmake', 'probes', 'tests', 'tools', 'tokenizer'):
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file()
                     and not {'target', '__pycache__'}.intersection(p.parts)
                     and (p.name == 'CMakeLists.txt' or p.suffix in
                          ('.cpp', '.h', '.in', '.cmake', '.py', '.rs', '.toml', '.lock', '.param', '.jinja')))
    return sorted(set(paths))
