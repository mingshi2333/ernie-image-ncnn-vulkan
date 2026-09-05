#!/usr/bin/env python3
"""Extract a pinned safetensors component using bounded HTTP Range requests."""
import argparse
import hashlib
import http.client
import json
import math
import re
import struct
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_HEADER = 16 * 1024 * 1024
WIDTHS = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
RANGE_WINDOW = 32 * 1024 * 1024


def request_range(url, start, end):
    # A distinct resolver URL avoids caches that ignore Range in their cache key.
    req = urllib.request.Request(f'{url}?download=true&range={start}-{end}',
                                 headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'ernie-image-ncnn-component/0.1'})
    response = urllib.request.urlopen(req, timeout=60)
    expected = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
    if response.status != 206 or not expected or (int(expected[1]), int(expected[2])) != (start, end):
        response.close()
        raise RuntimeError('Server did not honor the exact byte range; refusing a full-shard download')
    return response


def read_header(url):
    with request_range(url, 0, 7) as response:
        size_data = response.read(8)
    if len(size_data) != 8:
        raise RuntimeError('Truncated safetensors size header')
    size = struct.unpack('<Q', size_data)[0]
    if not 2 <= size <= MAX_HEADER:
        raise RuntimeError(f'Unexpected safetensors header length: {size}')
    with request_range(url, 8, 7 + size) as response:
        data = response.read(size + 1)
    if len(data) != size:
        raise RuntimeError('Truncated safetensors JSON header')
    return json.loads(data), size + 8, hashlib.sha256(data).hexdigest()


def bounded_ranges(url, start, end, window=RANGE_WINDOW):
    """Yield only fully received, exact-size windows; retry transient truncation."""
    if window < 1 or start < 0 or end < start:
        raise ValueError('Invalid bounded range')
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + window - 1)
        count = stop - cursor + 1
        for attempt in range(3):
            try:
                with request_range(url, cursor, stop) as response:
                    data = response.read(count + 1)
                if len(data) != count:
                    raise RuntimeError(f'Truncated range window: expected {count}, received {len(data)}')
                break
            except (OSError, RuntimeError, http.client.HTTPException) as error:
                print(json.dumps({'status': 'range_retry', 'start': cursor, 'end': stop,
                                  'attempt': attempt + 1, 'error': str(error)}), flush=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        yield data
        cursor = stop + 1


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_component(prefix, output, subfolder='transformer', index_name='diffusion_pytorch_model.safetensors.index.json', single_file=None):
    lock = json.loads((ROOT / 'sources.lock.json').read_text())['official_model']
    revision = lock['revision']
    base = f"{lock['url']}/resolve/{revision}/{subfolder}/"
    headers = {}
    if single_file is not None:
        if Path(single_file).name != single_file:
            raise ValueError('Unexpected single-file path')
        headers[single_file] = read_header(base + single_file)
        weight_map = {name: single_file for name in headers[single_file][0] if name != '__metadata__'}
        source_info = {'single_file_url': base + single_file,
                       'source_header_sha256': headers[single_file][2]}
    else:
        with urllib.request.urlopen(base + index_name, timeout=30) as response:
            index_data = response.read(MAX_HEADER + 1)
        if len(index_data) > MAX_HEADER:
            raise RuntimeError('Index unexpectedly large')
        weight_map = json.loads(index_data)['weight_map']
        source_info = {'index_url': base + index_name, 'index_sha256': hashlib.sha256(index_data).hexdigest()}
    selected = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not selected:
        raise ValueError(f'No tensors match {prefix!r}')
    manifest_path = output.with_suffix('.manifest.json')
    if output.exists():
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if (manifest['revision'] == revision and manifest['prefix'] == prefix
                    and manifest['sha256'] == sha256(output)
                    and all(manifest.get(key) == value for key, value in source_info.items())):
                print(json.dumps({'status': 'reused', 'file': str(output), 'sha256': manifest['sha256']}), flush=True)
                return manifest
        raise FileExistsError('Existing component has no matching valid manifest; use a new output path')

    records = []
    for shard in sorted(set(selected.values())):
        if Path(shard).name != shard:
            raise ValueError('Unexpected shard path')
        header, data_start, header_hash = headers[shard] if shard in headers else read_header(base + shard)
        for name in selected:
            if selected[name] != shard:
                continue
            item = header[name]
            begin, end = item['data_offsets']
            if item['dtype'] not in WIDTHS or end - begin != math.prod(item['shape']) * WIDTHS[item['dtype']]:
                raise ValueError(f'Invalid tensor layout: {name}')
            records.append({'name': name, 'shard': shard, 'dtype': item['dtype'], 'shape': item['shape'],
                            'source_offsets': [data_start + begin, data_start + end], 'header_sha256': header_hash})
    records.sort(key=lambda item: (item['shard'], item['source_offsets'][0]))
    target_header = {'__metadata__': {'repository': lock['url'], 'revision': revision, 'prefix': prefix}}
    cursor = 0
    for item in records:
        size = item['source_offsets'][1] - item['source_offsets'][0]
        target_header[item['name']] = {'dtype': item['dtype'], 'shape': item['shape'], 'data_offsets': [cursor, cursor + size]}
        cursor += size
    header_bytes = json.dumps(target_header, separators=(',', ':')).encode()
    header_bytes += b' ' * ((-len(header_bytes)) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(f'Partial output exists: {temporary}; preserve or remove it before retrying')
    print(json.dumps({'status': 'download', 'tensors': len(records), 'payload_bytes': cursor, 'prefix': prefix}), flush=True)
    # Adjacent tensors share bounded requests, with independent tensor hashes.
    groups = []
    for item in records:
        if groups and groups[-1][-1]['shard'] == item['shard'] and groups[-1][-1]['source_offsets'][1] == item['source_offsets'][0]:
            groups[-1].append(item)
        else:
            groups.append([item])
    with temporary.open('xb') as target:
        target.write(struct.pack('<Q', len(header_bytes)))
        target.write(header_bytes)
        for group in groups:
            chunks = iter(bounded_ranges(base + group[0]['shard'], group[0]['source_offsets'][0], group[-1]['source_offsets'][1] - 1))
            data, offset = b'', 0
            for item in group:
                remaining = item['source_offsets'][1] - item['source_offsets'][0]
                h = hashlib.sha256()
                while remaining:
                    if offset == len(data):
                        data, offset = next(chunks), 0
                    take = min(remaining, len(data) - offset)
                    part = memoryview(data)[offset:offset + take]
                    target.write(part)
                    h.update(part)
                    remaining -= take
                    offset += take
                item['sha256'] = h.hexdigest()
                print(json.dumps({'tensor': item['name'], 'status': 'downloaded'}), flush=True)
            if offset != len(data) or next(chunks, None) is not None:
                raise RuntimeError('Response exceeds requested component range')
    temporary.rename(output)
    manifest = {'schema_version': 1, 'repository': lock['url'], 'revision': revision, 'prefix': prefix,
                **source_info,
                'sha256': sha256(output), 'payload_bytes': cursor, 'tensors': records,
                'range_window_bytes': RANGE_WINDOW, 'downloader_sha256': sha256(Path(__file__)),
                'verification': 'HTTPS from pinned revision, exact range and length checks; local per-tensor and component hashes. Full upstream shard hash not checked.'}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'file': str(output), 'sha256': manifest['sha256']}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 <= args.block < 36:
        parser.error('block must be in 0..35')
    fetch_component(f'layers.{args.block}.', args.output or ROOT / 'models/official' / f'dit-block-{args.block:02d}.safetensors')


if __name__ == '__main__':
    main()
