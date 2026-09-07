import json
from pathlib import Path
import sys
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'source/tools'))
import torch
from validate_pipeline import reference
request=json.loads((base/'request.json').read_text())
torch.set_num_threads(2)
torch.set_grad_enabled(False)
fixture=reference(Path(request['package']),request['prompt'],base/'official/reference',request['steps'],device=request['reference_device'],
    initial_path=Path(request['initial']),runtime_size=tuple(request['runtime_size']),source_weights_package=Path(request['source_weights_package']))
print(json.dumps({'complete':fixture['complete'],'config':fixture['config'],'ids':fixture['ids']}),flush=True)
