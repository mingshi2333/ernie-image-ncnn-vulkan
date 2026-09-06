"""Bounded decoder-only shape contract; no production shape registration."""
FIXED_LATENT_HW = (96, 172)

def validate_decoder_shape(height, width, *, fixed=False, reference_only=True):
    if type(height) is not int or type(width) is not int:
        raise ValueError('Decoder dimensions must be integers')
    if fixed:
        if (height, width) != FIXED_LATENT_HW:
            raise ValueError('Fixed 1376x768 requires latent height=96 width=172')
        if not reference_only:
            raise ValueError('Fixed 1376x768 is reference-only; whole-VAE pnnx is forbidden')
    elif not (1 <= height <= 128 and 1 <= width <= 128):
        raise ValueError('Ordinary latent dimensions remain in [1,128]')
    if height * width > 4096 and not reference_only:
        raise ValueError('Large whole-VAE pnnx is forbidden; use reference-only and specialization')


def specialize_decoder_graph(graph, height, width, *, fixed=False):
    """Caller authenticates the full template SHA; replace exactly two known nodes."""
    validate_decoder_shape(height, width, fixed=fixed)
    expected = {'reshape_99': ['0=64', '1=512'],
                'reshape_100': ['0=8', '1=8', '2=512']}
    seen = set(); lines = []; changes = []
    for line in graph.splitlines():
        fields = line.split()
        if fields and fields[0] == 'Reshape':
            if (len(fields) < 6 or fields[1] not in expected or fields[1] in seen or
                    fields[2:4] != ['1', '1'] or fields[6:] != expected[fields[1]]):
                raise ValueError('Unreviewed or duplicate spatial reshape')
            name = fields[1]; seen.add(name)
            values = ([f'0={height*width}', '1=512'] if name == 'reshape_99' else
                      [f'0={width}', f'1={height}', '2=512'])
            updated = ' '.join(fields[:6] + values)
            changes.append({'before': line, 'after': updated}); line = updated
        lines.append(line)
    if seen != set(expected):
        raise ValueError('Expected precisely two spatial reshapes')
    return '\n'.join(lines) + '\n', changes
