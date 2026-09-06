"""Named weight roles in the fixed peer graphs, independently of value lookup.

This checks graph connections and matrix axes. It does not certify the numerical
semantics of ncnn kernels, runtime conditioning, or complete model parity.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path


GRAPH_PINS = {
    'dit/chunks.ncnn.param': '74ccdb234ef4e7844a0857942adaf1338163ad093562f6033f7da53c437e7abe',
    'text_encoder/language_model_encoder.ncnn.param': 'f4a7e07f9d6d8eea5851f952ee9089c94a903ad16a886ea624367d7da7590314',
    'pe/decoder.ncnn.param': '0dd27e936dd5c1a5da820e282a6a183248893aba6cad7ec78f16548550cc1ac4',
    'pe/embed_tokens.ncnn.param': '95f0b1f34b606b23e71b3c60ff8394faf5d8e69b7011c38f9f822ddb6c1067dc',
    'pe/lm_head.ncnn.param': '9ebbb183b7b527d5be13f2e6a5967cf8c0bb99947518367b165344a5ce283b2e',
    'text_encoder/language_model_embed_tokens.ncnn.param': 'd280feb1ae24d787d5f59966252caec80fce4068fb79cb0ee69ae97431b807b3',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Graph:
    def __init__(self, text):
        lines = text.splitlines()
        require(len(lines) >= 2 and lines[0] == '7767517', 'Invalid graph header')
        count, blobs = map(int, lines[1].split())
        self.nodes, self.names, self.producers = [], {}, {}
        for line in lines[2:]:
            fields = line.split()
            require(len(fields) >= 4, 'Truncated graph layer')
            kind, name = fields[:2]
            ni, no = map(int, fields[2:4])
            require(ni >= 0 and no > 0 and len(fields) >= 4 + ni + no, 'Invalid graph arity')
            inputs, outputs = fields[4:4 + ni], fields[4 + ni:4 + ni + no]
            pairs = [x.split('=', 1) for x in fields[4 + ni + no:]]
            require(all(len(x) == 2 for x in pairs), 'Invalid graph parameter')
            params = dict(pairs)
            require(len(params) == len(pairs), 'Duplicate graph parameter')
            require(name not in self.names, 'Duplicate graph layer')
            require(all(x in self.producers for x in inputs), 'Graph input precedes its producer')
            require(len(set(outputs)) == len(outputs) and not set(outputs) & self.producers.keys(), 'Duplicate graph blob')
            node = dict(kind=kind, name=name, inputs=inputs, outputs=outputs, params=params)
            self.nodes.append(node)
            self.names[name] = node
            self.producers.update((x, node) for x in outputs)
        require(len(self.nodes) == count and len(self.producers) <= blobs, 'Graph denominator mismatch')
        # The pinned text graph allocates two unused blob slots. No edge may
        # refer to an undeclared producer, but spare allocation slots are legal.
        self.unused_blob_slots = blobs - len(self.producers)

    def unsplit(self, blob):
        while self.producers[blob]['kind'] == 'Split':
            node = self.producers[blob]
            require(len(node['inputs']) == 1, 'Split has multiple inputs')
            blob = node['inputs'][0]
        return blob

    def ancestor(self, blob, kind, allowed):
        while True:
            node = self.producers[self.unsplit(blob)]
            if node['kind'] == kind:
                return node
            require(node['kind'] in allowed and node['inputs'], 'Unexpected path to ' + kind)
            blob = node['inputs'][0]

    def binary(self, code, inputs):
        wanted = Counter(self.unsplit(x) for x in inputs)
        matches = [n for n in self.nodes if n['kind'] == 'BinaryOp'
                   and n['params'] == {'0': str(code)}
                   and Counter(self.unsplit(x) for x in n['inputs']) == wanted]
        require(len(matches) == 1, 'Missing or ambiguous binary connection')
        return matches[0]['outputs'][0]

    def modulation(self, blob, normalized, scale, shift):
        add = self.producers[self.unsplit(blob)]
        require(add['kind'] == 'BinaryOp' and add['params'] == {'0': '0'}
                and len(add['inputs']) == 2, 'Modulation shift addition')
        operands = [self.unsplit(x) for x in add['inputs']]
        require(operands.count(self.unsplit(shift)) == 1, 'Modulation shift source')
        product = self.producers[operands[1 - operands.index(self.unsplit(shift))]]
        require(product['kind'] == 'BinaryOp' and product['params'] == {'0': '2'}
                and len(product['inputs']) == 2, 'Modulation scale multiplication')
        operands = [self.unsplit(x) for x in product['inputs']]
        require(operands.count(self.unsplit(normalized)) == 1, 'Modulation normalization source')
        scalar = self.producers[operands[1 - operands.index(self.unsplit(normalized))]]
        require(scalar['kind'] == 'BinaryOp' and scalar['params'].get('0') == '0'
                and scalar['params'].get('1') == '1' and float(scalar['params'].get('2', 'nan')) == 1.0
                and len(scalar['inputs']) == 1 and self.unsplit(scalar['inputs'][0]) == self.unsplit(scale),
                'Modulation scale must add one')
        return self.unsplit(blob)


def _out(node):
    return node['outputs'][0]


def block_roles(graph, family, count):
    """Assign roles from fixed block order and verified dataflow, never hashes."""
    require(family in ('dit', 'text', 'pe') and count > 0, 'Invalid graph family')
    dit = family == 'dit'
    gemms = [n for n in graph.nodes if n['kind'] == 'Gemm']
    norms = [n for n in graph.nodes if n['kind'] == 'RMSNorm']
    attention = [n for n in graph.nodes if n['kind'] == 'SDPA']
    require(len(gemms) == count * 7 and len(attention) == count, 'Block projection/attention denominator')
    require(len(norms) == count * (4 if dit else 2) + (family == 'pe'), 'Block norm denominator')
    result, state = {}, 'in0'
    for i in range(count):
        q, k, v, o, first, second, down = gemms[i * 7:i * 7 + 7]
        ns = norms[i * (4 if dit else 2):(i + 1) * (4 if dit else 2)]
        before, after = ns[0], ns[-1]
        attn = attention[i]
        require(graph.unsplit(before['inputs'][0]) == graph.unsplit(state), 'Block residual order')
        require(len(attn['inputs']) == (6 if family == 'pe' else 4), 'Attention input count')
        if family == 'pe':
            require(attn['inputs'][4:] == [f'cache_k{i}', f'cache_v{i}']
                    and attn['outputs'][1:] == [f'out_cache_k{i}', f'out_cache_v{i}'], 'PE cache layer correspondence')
        path = {'Reshape', 'Permute', 'RotaryEmbed', 'ErnieImageRoPE', 'RMSNorm'}
        for slot, projection in enumerate((q, k, v)):
            require(graph.ancestor(attn['inputs'][slot], 'Gemm', path)['name'] == projection['name'], 'Q/K/V role connection')
        require(graph.ancestor(o['inputs'][0], 'SDPA', {'Reshape', 'Permute'})['name'] == attn['name'], 'Attention output projection')
        condition = _out(before)
        if dit:
            for slot, norm in enumerate(ns[1:3]):
                require(graph.ancestor(attn['inputs'][slot], 'RMSNorm', path - {'RMSNorm'})['name'] == norm['name'], 'Q/K normalization role')
            condition = graph.modulation(q['inputs'][0], condition, 'in2', 'in1')
        require(all(graph.unsplit(n['inputs'][0]) == graph.unsplit(condition) for n in (q, k, v)), 'Projection input conditioning')
        projected = graph.binary(2, ['in3', _out(o)]) if dit else _out(o)
        residual = graph.binary(0, [state, projected])
        require(graph.unsplit(after['inputs'][0]) == graph.unsplit(residual), 'Post-attention normalization')
        condition = _out(after)
        if dit:
            condition = graph.modulation(first['inputs'][0], condition, 'in5', 'in4')
        require(all(graph.unsplit(n['inputs'][0]) == graph.unsplit(condition) for n in (first, second)), 'MLP input conditioning')
        gate, up = (second, first) if dit else (first, second)
        nonlinear = [n for n in graph.nodes if n['kind'] == ('GELU' if dit else 'Swish')
                     and len(n['inputs']) == 1 and graph.unsplit(n['inputs'][0]) == _out(gate)]
        require(len(nonlinear) == 1, 'MLP gate projection role')
        product = graph.binary(2, [_out(up), _out(nonlinear[0])])
        require(graph.unsplit(down['inputs'][0]) == graph.unsplit(product), 'MLP down projection role')
        projected = graph.binary(2, ['in6', _out(down)]) if dit else _out(down)
        state = graph.binary(0, [residual, projected])
        if dit:
            roles = [(before, 'adaLN_sa_ln.weight'), (q, 'self_attention.to_q.weight'),
                     (k, 'self_attention.to_k.weight'), (v, 'self_attention.to_v.weight'),
                     (ns[1], 'self_attention.norm_q.weight'), (ns[2], 'self_attention.norm_k.weight'),
                     (o, 'self_attention.to_out.0.weight'), (after, 'adaLN_mlp_ln.weight'),
                     (up, 'mlp.up_proj.weight'), (gate, 'mlp.gate_proj.weight'), (down, 'mlp.linear_fc2.weight')]
            prefix = f'layers.{i}.'
        else:
            roles = [(before, 'input_layernorm.weight'), (q, 'self_attn.q_proj.weight'),
                     (k, 'self_attn.k_proj.weight'), (v, 'self_attn.v_proj.weight'),
                     (o, 'self_attn.o_proj.weight'), (after, 'post_attention_layernorm.weight'),
                     (gate, 'mlp.gate_proj.weight'), (up, 'mlp.up_proj.weight'), (down, 'mlp.down_proj.weight')]
            prefix = ('language_model.model.' if family == 'text' else 'model.') + f'layers.{i}.'
        for node, role in roles:
            result[node['name']] = (f'{family}-block-{i:02d}.safetensors', prefix + role)
    if family == 'pe':
        require(graph.unsplit(norms[-1]['inputs'][0]) == graph.unsplit(state), 'PE final normalization input')
        state = _out(norms[-1])
        result[norms[-1]['name']] = ('pe-norm.safetensors', 'model.norm.weight')
    require(graph.unsplit(state) == graph.unsplit('out0'), 'Final graph output')
    return result


def weight_shape(node):
    p = node['params']
    if node['kind'] == 'Gemm':
        require(all(p.get(k) == v for k, v in {'2': '0', '3': '1', '4': '0', '5': '1', '6': '1', '10': '-1'}.items()), 'Unreviewed matrix storage/layout')
        return [int(p['8']), int(p['9'])], 'B'
    if node['kind'] == 'RMSNorm':
        require(p.get('2') == '1', 'Non-affine normalization has no learned weight')
        return [int(p['0'])], 'gamma'
    if node['kind'] == 'Embed':
        require(p.get('2') == '0', 'Unreviewed embedding bias')
        return [int(p['1']), int(p['0'])], 'weight'
    require(node['kind'] == 'InnerProduct' and p.get('1') == '0', 'Unreviewed tied output projection')
    require(int(p['2']) % int(p['0']) == 0, 'Invalid output projection dimensions')
    return [int(p['0']), int(p['2']) // int(p['0'])], 'weight'


def graph_weight_mappings(source, peer_rows, official_rows):
    targets = {(r['file'], r['name']): r for r in official_rows}
    require(len(targets) == len(official_rows), 'Duplicate official logical tensor')
    mapped, gaps, reviewed, spare_slots = [], [], [], {}
    specs = {
        'dit/chunks.ncnn.param': ('dit', 36),
        'text_encoder/language_model_encoder.ncnn.param': ('text', 25),
        'pe/decoder.ncnn.param': ('pe', 26),
    }
    singles = {
        'pe/embed_tokens.ncnn.param': ('Embed', 'pe-embed.safetensors', 'model.embed_tokens.weight'),
        'pe/lm_head.ncnn.param': ('InnerProduct', 'pe-lm-head.safetensors', 'lm_head.weight'),
        'text_encoder/language_model_embed_tokens.ncnn.param': ('Embed', 'text-embed.safetensors', 'language_model.model.embed_tokens.weight'),
    }
    for relative, pin in GRAPH_PINS.items():
        binary = str(Path(relative).with_suffix('.bin'))
        rows = [r for r in peer_rows if r['file'] == binary]
        if not rows:
            continue  # Partial component scans do not imply coverage of missing graphs.
        try:
            path = Path(source) / relative
            require(hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'Graph source identity mismatch')
            graph = Graph(path.read_text())
            if relative in specs:
                roles = block_roles(graph, *specs[relative])
            else:
                kind, file, name = singles[relative]
                require(len(graph.nodes) == 2 and graph.nodes[0]['kind'] == 'Input', 'Embedding graph denominator')
                node = graph.nodes[1]
                require(node['kind'] == kind and node['inputs'] == ['in0'] and node['outputs'] == ['out0'], 'Embedding/LM role connection')
                roles = {node['name']: (file, name)}
            require(len(rows) == len(roles) and len({r['layer'] for r in rows}) == len(rows), 'Learned weight denominator')
            checked = []
            for row in rows:
                require(row['layer'] in roles and row['param_sha256'] == pin, 'Serialized row graph binding')
                node = graph.names[row['layer']]
                shape, role = weight_shape(node)
                target_key = roles[row['layer']]
                require(target_key in targets, 'Official named component missing')
                target = targets[target_key]
                require(row['kind'] == node['kind'] and row['role'] == role and row['count'] == math.prod(shape), 'Serialized weight role/size')
                require(target['shape'] == shape and target['canonical_sha256'] == row['canonical_sha256'], 'Named official weight shape/value mismatch')
                checked.append(dict(file=binary, layer=row['layer'], role=role, logical_file=target_key[0],
                                    logical_name=target_key[1], logical_shape=shape, graph_sha256=pin,
                                    canonical_sha256=row['canonical_sha256'], status='named_graph_weight_match'))
            mapped.extend(checked)
            reviewed.append(relative)
            spare_slots[relative] = graph.unused_blob_slots
        except (ValueError, KeyError, OSError, ZeroDivisionError) as exc:
            gaps.append(dict(file=relative, reason=str(exc)))
    return dict(scope='Fixed graph weight roles and serialized axes; runtime/kernel semantics and end-to-end parity not proven',
                reviewed_graphs=reviewed, unused_blob_slots=spare_slots, mapped_tensor_count=len(mapped), mappings=mapped, gaps=gaps,
                expected_complete_mapped_tensor_count=859, allowed_to_close_S=False)
