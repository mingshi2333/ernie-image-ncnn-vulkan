import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.port_graph_contract import Graph, block_roles, graph_weight_mappings, weight_shape


def text_graph(count=1):
    """Small logical graph, no model data or downloaded fixtures."""
    rows = []

    def add(kind, name, inputs, output, params=''):
        rows.append(f'{kind} {name} {len(inputs)} 1 ' + ' '.join([*inputs, output]) + (' ' + params if params else ''))
        return output

    add('Input', 'input', [], 'in0')
    add('Input', 'mask', [], 'mask')
    state = 'in0'
    gemm = '2=0 3=1 4=0 5=1 6=1 8=2 9=2 10=-1'
    for i in range(count):
        prefix = f'b{i}_'

        def node(kind, name, inputs, params=''):
            return add(kind, prefix + name, inputs, prefix + name + '_out', params)

        norm = node('RMSNorm', 'pre', [state], '0=2 1=0.00001 2=1')
        q, k, v = [node('Gemm', name, [norm], gemm) for name in ('q', 'k', 'v')]
        attention = node('SDPA', 'attention', [q, k, v, 'mask'], '5=1')
        projected = node('Gemm', 'o', [attention], gemm)
        residual = node('BinaryOp', 'attention_residual', [state, projected], '0=0')
        norm = node('RMSNorm', 'post', [residual], '0=2 1=0.00001 2=1')
        gate = node('Gemm', 'gate', [norm], gemm)
        active = node('Swish', 'activation', [gate])
        up = node('Gemm', 'up', [norm], gemm)
        product = node('BinaryOp', 'product', [up, active], '0=2')
        down = node('Gemm', 'down', [product], gemm)
        state = add('BinaryOp', prefix + 'mlp_residual', [residual, down],
                    'out0' if i == count - 1 else prefix + 'state', '0=0')
    return f'7767517\n{len(rows)} {len(rows)}\n' + '\n'.join(rows) + '\n'


class GraphRoleTests(unittest.TestCase):
    def test_roles_follow_connections_without_weight_values(self):
        graph = Graph(text_graph(2))
        roles = block_roles(graph, 'text', 2)
        self.assertEqual(len(roles), 18)
        self.assertEqual(roles['b1_k'], ('text-block-01.safetensors', 'language_model.model.layers.1.self_attn.k_proj.weight'))
        self.assertEqual(roles['b0_gate'][1], 'language_model.model.layers.0.mlp.gate_proj.weight')

    def test_qk_gate_and_residual_rewiring_is_rejected(self):
        text = text_graph()
        mutations = [
            ('b0_q_out b0_k_out b0_v_out mask', 'b0_k_out b0_q_out b0_v_out mask'),
            ('Swish b0_activation 1 1 b0_gate_out', 'Swish b0_activation 1 1 b0_up_out'),
            ('in0 b0_o_out b0_attention_residual_out', 'b0_pre_out b0_o_out b0_attention_residual_out'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaises(ValueError):
                block_roles(Graph(text.replace(before, after)), 'text', 1)
        # This replacement still uses an already defined producer, so rejection
        # must come from the MLP role check, not the generic topological parser.
        changed = text.replace('Swish b0_activation 1 1 b0_gate_out', 'Swish b0_activation 1 1 b0_q_out')
        with self.assertRaisesRegex(ValueError, 'MLP gate projection role'):
            block_roles(Graph(changed), 'text', 1)

    def test_modulation_selects_the_actual_branch_among_repeated_constants(self):
        rows = ['Input scale 0 1 scale', 'Input shift 0 1 shift',
                'Input norm0 0 1 norm0', 'Input norm1 0 1 norm1']
        for i in range(2):
            rows.extend([f'BinaryOp plus{i} 1 1 scale one{i} 0=0 1=1 2=1.0',
                         f'BinaryOp mul{i} 2 1 norm{i} one{i} product{i} 0=2',
                         f'BinaryOp add{i} 2 1 product{i} shift out{i} 0=0'])
        text = f'7767517\n{len(rows)} {len(rows)}\n' + '\n'.join(rows)
        graph = Graph(text)
        for i in range(2):
            self.assertEqual(graph.modulation(f'out{i}', f'norm{i}', 'scale', 'shift'), f'out{i}')
        with self.assertRaisesRegex(ValueError, 'normalization source'):
            graph.modulation('out0', 'norm1', 'scale', 'shift')
        with self.assertRaisesRegex(ValueError, 'scale must add one'):
            Graph(text.replace('2=1.0', '2=0.0')).modulation('out0', 'norm0', 'scale', 'shift')

    def test_missing_block_and_unbound_blob_fail(self):
        with self.assertRaises(ValueError):
            block_roles(Graph(text_graph()), 'text', 2)
        with self.assertRaises(ValueError):
            Graph(text_graph().replace('b0_q_out b0_k_out', 'never_defined b0_k_out'))
        lines = text_graph().splitlines()
        count, blobs = map(int, lines[1].split())
        lines[1] = f'{count} {blobs + 2}'
        self.assertEqual(Graph('\n'.join(lines)).unused_blob_slots, 2)

    def test_matrix_axis_and_affine_checks(self):
        node = Graph(text_graph()).names['b0_q']
        self.assertEqual(weight_shape(node), ([2, 2], 'B'))
        for key, value in [('3', '0'), ('4', '1'), ('10', '4')]:
            changed = {**node, 'params': {**node['params'], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                weight_shape(changed)

    def mapping_fixture(self, root):
        relative = 'text_encoder/language_model_encoder.ncnn.param'
        path = root / relative
        path.parent.mkdir()
        path.write_text(text_graph(25))
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        graph = Graph(path.read_text())
        roles = block_roles(graph, 'text', 25)
        rows, official = [], []
        for layer, (file, name) in roles.items():
            shape, role = weight_shape(graph.names[layer])
            digest = hashlib.sha256(layer.encode()).hexdigest()
            rows.append(dict(file=str(Path(relative).with_suffix('.bin')), layer=layer,
                             kind=graph.names[layer]['kind'], role=role, count=math.prod(shape),
                             param_sha256=pin, canonical_sha256=digest))
            official.append(dict(file=file, name=name, shape=shape, canonical_sha256=digest))
        return relative, path, pin, rows, official

    def test_named_mapping_rejects_forged_shape_value_or_denominator(self):
        for mutation in ('none', 'shape', 'value', 'denominator', 'source'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                relative, path, pin, rows, official = self.mapping_fixture(root)
                if mutation == 'shape':
                    official[1]['shape'] = [1, 4]
                elif mutation == 'value':
                    official[1]['canonical_sha256'] = '0' * 64
                elif mutation == 'denominator':
                    rows.pop()
                elif mutation == 'source':
                    path.write_text(path.read_text() + '\n')
                with patch('tools.port_graph_contract.GRAPH_PINS', {relative: pin}):
                    result = graph_weight_mappings(root, rows, official)
                self.assertEqual(result['mapped_tensor_count'], 225 if mutation == 'none' else 0)
                self.assertEqual(bool(result['gaps']), mutation != 'none')
                self.assertFalse(result['allowed_to_close_S'])


if __name__ == '__main__':
    unittest.main()
