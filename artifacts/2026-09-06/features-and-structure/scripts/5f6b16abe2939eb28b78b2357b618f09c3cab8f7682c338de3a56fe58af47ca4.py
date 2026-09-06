"""UTF-8 prompt-file convention shared by conversion/reference tools and CLI."""
from pathlib import Path


def read_prompt(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Cannot open prompt file: {path}')
    with path.open('rb') as stream:
        data = stream.read(1024*1024 + 1)
    if len(data) > 1024*1024:
        raise ValueError('Prompt file exceeds 1 MiB')
    text = data.decode('utf-8-sig')
    if '\0' in text:
        raise ValueError('Prompt file contains NUL; require UTF-8 text')
    return text
