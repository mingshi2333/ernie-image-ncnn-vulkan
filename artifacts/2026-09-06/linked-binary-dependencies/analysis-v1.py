#!/usr/bin/env python3
"""Recompute the fixed D3 GNU ld / ELF64 link-map supplement; no model calls.

Archive selection, positive input-section contributions and SHF_ALLOC intersections
are separate observations. These are not redistribution or license conclusions.
"""
import collections
import hashlib
import json
import re
import struct
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_member_hashes(path, selected):
    """Read GNU regular ar members without extracting or resolving thin paths."""
    found = {}
    names = b''
    with Path(path).open('rb') as stream:
        assert stream.read(8) == b'!<arch>\n', 'Not a regular ar archive'
        while header := stream.read(60):
            assert len(header) == 60 and header[58:60] == b'`\n', 'Truncated ar header'
            size = int(header[48:58])
            assert size >= 0
            start = stream.tell()
            assert start + size <= Path(path).stat().st_size, 'Truncated ar member'
            name = header[:16].decode('ascii').strip()
            if name == '//':
                assert size < 8 * 1024 * 1024
                names = stream.read(size)
            elif name not in ('/', '/SYM64/'):
                if name.startswith('/'):
                    offset = int(name[1:])
                    assert 0 <= offset < len(names)
                    end = names.index(b'/\n', offset)
                    name = names[offset:end].decode('utf-8')
                else:
                    assert not name.startswith('#1/'), 'BSD names outside fixed format'
                    name = name.removesuffix('/')
                if name in selected:
                    assert name not in found, 'Ambiguous duplicate member'
                    digest = hashlib.sha256()
                    remaining = size
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        assert block
                        digest.update(block)
                        remaining -= len(block)
                    found[name] = {'sha256': digest.hexdigest(), 'size_bytes': size}
            stream.seek(start + size + (size & 1))
        assert found.keys() == selected, 'Missing selected archive members'
    return found


def alloc_sections(path):
    data = Path(path).read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01', 'Only ELF64 little-endian is accepted'
    assert struct.unpack_from('<HH', data, 16) == (2, 62), 'Expected x86-64 ET_EXEC'
    offset = struct.unpack_from('<Q', data, 40)[0]
    entry_size, count, names_index = struct.unpack_from('<HHH', data, 58)
    assert entry_size == 64 and count and names_index < count
    assert offset + count * entry_size <= len(data)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * 64)
                for i in range(count)]
    strings = sections[names_index]
    names = data[strings[4]:strings[4] + strings[5]]
    result = []
    for s in sections:
        if s[2] & 2 and s[5]:  # SHF_ALLOC; file and NOBITS tracked separately.
            name = names[s[0]:names.index(b'\0', s[0])].decode('ascii')
            result.append({'name': name, 'type': s[1], 'flags': s[2],
                           'address': s[3], 'size': s[5]})
    return result


def parse_map(text):
    markers = ['Archive member included to satisfy reference by file (symbol)',
               'Discarded input sections', 'Memory Configuration',
               'Linker script and memory map', 'Cross Reference Table']
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    included = re.findall(r'^([^\s]+\.a)\(([^)]+)\)', text[:positions[1]], re.M)
    assert included and len(set(included)) == len(included)
    rows = []
    pending = None
    expression = re.compile(r'^\s+(?:(\.[^\s]+)\s+)?(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(.+?)\s*$')
    for line in text[positions[3]:positions[4]].splitlines():
        match = expression.match(line)
        if match:
            name, address, size, origin = match.groups()
            assert re.fullmatch(r'[^\s]+\.a\([^)]+\)|[^\s]+\.o', origin), origin
            rows.append({'section': name or pending, 'address': int(address, 16),
                         'size': int(size, 16), 'origin': origin})
            assert rows[-1]['section'], 'Unidentified continuation input section'
            pending = None
        else:
            name = re.fullmatch(r' (\.[^\s]+)\s*', line)
            pending = name[1] if name else None
    assert rows, 'Empty final input-section denominator'
    return included, rows


def main():
    worktree = Path.cwd()
    output = worktree / 'outputs/d2-linkmap-v1'
    plan = json.loads((output / 'plan.json').read_text())
    result = json.loads((output / 'result.json').read_text())
    assert sha(output / 'plan.json') == 'c47f4272dfebf89543afabfc1a2dd7205e4b156a2f0f25fc050bd1461c24349d'
    assert result['status'] == 'passed_identical_binary_link_map'
    assert result['return_code'] == 0 and result['failure'] is None
    assert sha(output / 'worker.py') == plan['worker_sha256']
    for path, record in plan['inputs'].items():
        assert Path(path).stat().st_size == record['size_bytes']
        assert sha(path) == record['sha256'] == result['post_sha256'][path]
    for name, record in result['outputs'].items():
        assert (output / name).stat().st_size == record['size_bytes']
        assert sha(output / name) == record['sha256']
    binary = output / 'ernie-image.relinked'
    assert sha(binary) == plan['original_binary_sha256']
    cwd = Path(plan['cwd'])
    included, rows = parse_map((output / 'link.map').read_text())
    alloc = alloc_sections(binary)
    by_archive = collections.defaultdict(set)
    for archive, member in included:
        by_archive[archive].add(member)
    input_paths = {Path(path).resolve() for path in plan['inputs']}
    members = {}
    for archive, selected in by_archive.items():
        path = (cwd / archive).resolve()
        assert path in input_paths
        for member, record in selected_member_hashes(path, selected).items():
            members[f'{archive}({member})'] = {
                'archive': str(path), 'member': member, **record}
    source_totals = {}
    for row in rows:
        if not row['size']:
            continue
        origin = row['origin']
        if origin not in source_totals:
            if origin in members:
                identity = members[origin]
            else:
                path = (cwd / origin).resolve()
                assert path in input_paths and origin.endswith('.o')
                identity = {'object': str(path), 'sha256': sha(path),
                            'size_bytes': path.stat().st_size}
            source_totals[origin] = {
                **identity, 'positive_section_count': 0, 'positive_map_bytes': 0,
                'alloc_overlap_bytes': 0, 'alloc_file_overlap_bytes': 0,
                'alloc_nobits_overlap_bytes': 0, 'input_sections': set()}
        total = source_totals[origin]
        total['positive_section_count'] += 1
        total['positive_map_bytes'] += row['size']
        total['input_sections'].add(row['section'])
        for section in alloc:
            overlap = max(0, min(row['address'] + row['size'],
                                 section['address'] + section['size'])
                          - max(row['address'], section['address']))
            total['alloc_overlap_bytes'] += overlap
            total['alloc_nobits_overlap_bytes' if section['type'] == 8
                  else 'alloc_file_overlap_bytes'] += overlap
    for total in source_totals.values():
        total['input_sections'] = sorted(total['input_sections'])
    assert set(members) <= set(source_totals), 'Included archive without positive contribution'
    old_path = worktree / 'outputs/delivery-dependencies-v4/dependencies.json'
    old_sha = sha(old_path)
    assert old_sha == '2b40e0fe16d7a5e7cf927b9782768ed126b29ef1688340b5e19d46e7aac39e64'
    old = json.loads(old_path.read_text())
    bridge = cwd / 'cargo/release/libernie_tokenizer_bridge.a'
    assert sha(bridge) == old['rust_archive']['sha256']
    project_crates = {(x['name'].replace('-', '_'), x['artifact']): x
                      for x in old['rust_archive']['matched_project_crates']}
    rust_associations = []
    for key, member in members.items():
        if Path(member['archive']) != bridge:
            continue
        name = member['member']
        record = {'origin': key, 'member_sha256': member['sha256']}
        crate = re.match(r'^([A-Za-z0-9_]+)-([0-9a-f]{16})\.', name)
        if crate and crate.groups() in project_crates:
            package = project_crates[crate.groups()]
            notice = old['notices'][package['id']]
            for item in notice['notices']:
                assert sha(old_path.parent / item['path']) == item['sha256']
            record.update(kind='previously_authenticated_project_crate',
                          package=package, notice_evidence=notice)
        else:
            associations = [a for a in old['runtime']['archive_associations']
                            if a['identical_members'].get(name) == member['sha256']]
            if associations:
                record.update(kind='previously_authenticated_runtime_member',
                              source_archives=[{'path': a['source_archive'], 'sha256': a['sha256']}
                                               for a in associations])
            else:
                # Archive provenance covers the two local bridge objects; no
                # third-party crate association is invented from name alone.
                record.update(kind='unassigned_by_previous_dependency_supplement')
        rust_associations.append(record)
    assert len(rust_associations) == len(by_archive['cargo/release/libernie_tokenizer_bridge.a'])
    for path, record in plan['inputs'].items():
        assert sha(path) == record['sha256'], 'Link input changed during analysis'
    payload = {
        'status': 'actual_identical_binary_link_map_analyzed',
        'distributable': False, 'licenses_complete': False,
        'binary_sha256': sha(binary), 'map_sha256': sha(output / 'link.map'),
        'plan_sha256': sha(output / 'plan.json'), 'result_sha256': sha(output / 'result.json'),
        'analysis_script_sha256': sha(Path(__file__)),
        'input_files_reverified_before_and_after': len(plan['inputs']),
        'archive_selected_members': len(included), 'final_input_section_rows': len(rows),
        'positive_contribution_sources': len(source_totals),
        'positive_alloc_overlap_sources': sum(t['alloc_overlap_bytes'] > 0 for t in source_totals.values()),
        'selected_members_per_archive': {key: len(value) for key, value in sorted(by_archive.items())},
        'allocated_elf_sections': alloc, 'contributions': dict(sorted(source_totals.items())),
        'rust_associations': rust_associations,
        'reused_dependency_evidence': {'path': str(old_path.resolve()), 'sha256': old_sha,
                                     'same_rust_archive_sha256': sha(bridge)},
        'limitations': [
            'This GNU ld map and ELF64 analysis applies only to the authenticated D3 executable.',
            'Positive map sizes are linker input-section attribution, not unique ELF byte totals or runtime memory.',
            'SHF_ALLOC intersections separate file/NOBITS overlap, not symbol-level ownership or license obligations.',
            'Existing archive associations are reused only because the complete Rust archive bytes are identical.',
            'The two local bridge objects remain unassigned by the older third-party dependency supplement.',
            'Fine-grained toolchain/vendor notices, external Vulkan runtime prerequisites and other platforms remain open.',
        ]}
    destination = output / 'analysis.json'
    with destination.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: payload[key] for key in [
        'status', 'archive_selected_members', 'final_input_section_rows',
        'positive_contribution_sources', 'positive_alloc_overlap_sources',
        'selected_members_per_archive']}, indent=2))
    print('rust association counts', dict(collections.Counter(x['kind'] for x in rust_associations)))


if __name__ == '__main__':
    main()
