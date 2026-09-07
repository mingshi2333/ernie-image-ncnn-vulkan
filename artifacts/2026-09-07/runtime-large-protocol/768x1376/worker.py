import json
from pathlib import Path
import sys
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'source/tools'))
import torch
import validate_pipeline
from validate_pipeline import reference
original_load_block = validate_pipeline.load_block
def load_block_with_memory(path, index):
    block, manifest = original_load_block(path, index)
    def observe(module, inputs, output):
        print(json.dumps({'cuda_memory_after_block': index, 'allocated_bytes': torch.cuda.memory_allocated(),
                          'reserved_bytes': torch.cuda.memory_reserved(),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved()}), flush=True)
    block.register_forward_hook(observe)
    return block, manifest
validate_pipeline.load_block = load_block_with_memory
request=json.loads((base/'request.json').read_text())
torch.set_num_threads(2)
torch.set_grad_enabled(False)
fixture=reference(Path(request['package']),request['prompt'],base/'official/reference',request['steps'],device=request['reference_device'],
    initial_path=Path(request['initial']),runtime_size=tuple(request['runtime_size']),source_weights_package=Path(request['source_weights_package']))
print(json.dumps({'complete':fixture['complete'],'config':fixture['config'],'ids':fixture['ids']}),flush=True)
