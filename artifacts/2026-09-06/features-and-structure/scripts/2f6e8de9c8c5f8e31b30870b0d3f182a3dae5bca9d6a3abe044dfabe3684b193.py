"""Model-free integrity contract for the optional native prompt-enhancer package."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(layers=26, hidden_size=3072, vocabulary=131072, capacity=4096, tokens_per_call=1)
TEMPLATE_SHA256 = '0c859484eecf01db103acd02c332610163ee425cd46866d6cb126ee1bee974ea'


def runtime_files():
    return sorted(['pe.cfg', 'embeddings.bf16', 'rope-inv-freq.f32', 'head.ncnn.param', 'head.ncnn.bin',
                   'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'tokenizer/chat_template.jinja'] +
                  [f'block-{i:02d}/pe.ncnn.{suffix}' for i in range(26) for suffix in ('param', 'bin')])


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def verify_pe_package(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    lock = json.loads((ROOT/'sources.lock.json').read_text())
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1
            or manifest.get('kind') != 'prompt_enhancer'
            or manifest.get('official_model_revision') != lock['official_model']['revision']
            or manifest.get('ncnn_revision') != lock['ncnn']['revision']):
        raise ValueError('PE package version differs')
    if type(manifest.get('portable')) is not bool:
        raise ValueError('Missing PE portable flag')
    names = set(runtime_files())
    if set(manifest.get('files', {})) != names or set(manifest.get('file_sizes', {})) != names:
        raise ValueError('PE runtime file inventory differs')
    cfg = manifest.get('config', {})
    if cfg != CONFIG or any(type(x) is not int for x in cfg.values()):
        raise ValueError('PE configuration differs')
    for name in sorted(names):
        path = root/name
        if manifest['portable']:
            current = root
            for part in Path(name).parts:
                current = current/part
                if current.is_symlink(): raise ValueError('Portable PE package contains a symlink')
        expected = manifest['file_sizes'][name]
        if not path.is_file() or type(expected) is not int or path.stat().st_size != expected:
            raise ValueError(f'PE file size differs: {name}')
        if sha256(path) != manifest['files'][name]:
            raise ValueError(f'PE file checksum differs: {name}')
    if sha256(root/'tokenizer/chat_template.jinja') != TEMPLATE_SHA256:
        raise ValueError('PE chat template differs from native formatter')
    words = (root/'pe.cfg').read_text().split()
    if len(words) != 10 or len(set(words[::2])) != 5:
        raise ValueError('Malformed pe.cfg')
    if dict(zip(words[::2], map(int, words[1::2]))) != CONFIG:
        raise ValueError('PE configuration and manifest disagree')
    return manifest
