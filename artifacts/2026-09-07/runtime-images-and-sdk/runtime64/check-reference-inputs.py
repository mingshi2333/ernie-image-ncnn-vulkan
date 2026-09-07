import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'source/tools'))
import validate_pipeline as validator
request=json.loads((base/'request.json').read_text())
source=Path(request['package']);donor=Path(request['source_weights_package'])
class BeforeForward(Exception):pass
rows=[]
with tempfile.TemporaryDirectory(prefix='ernie-reference-inputs-') as temp:
 temp=Path(temp)
 damaged=temp/'damaged';damaged.mkdir()
 data=json.loads((donor/'manifest.json').read_text());data['config']['text_bucket']=32
 (damaged/'manifest.json').write_text(json.dumps(data))
 cases=[('pinned-source-with-identical-assets',source,dict(runtime_size=(64,64),source_weights_package=donor),32),
        ('missing-weight-provenance',source,dict(runtime_size=(64,64)),None),
        ('changed-donor-identity',source,dict(runtime_size=(64,64),source_weights_package=damaged),None),
        ('changed-source-identity',damaged,dict(runtime_size=(64,64),source_weights_package=donor),None),
        ('nonmultiple-dimension',source,dict(runtime_size=(65,64),source_weights_package=donor),None),
        ('area-overflow-contract',source,dict(runtime_size=(2048,2048),source_weights_package=donor),None),
        ('legacy-fixed-reference',donor,{},64)]
 for name,package,options,bucket in cases:
  with patch.object(validator,'real_reference',side_effect=BeforeForward) as forward:
   try:validator.reference(package,'test',temp/name,8,**options)
   except BeforeForward:
    assert bucket is not None and forward.call_args.args[-1]==bucket,name
    rows.append({'case':name,'result':'correct source selected before model forward'})
   except ValueError as error:
    assert bucket is None and not forward.called,name
    rows.append({'case':name,'result':'rejected before model forward','error':str(error)})
   else:raise AssertionError(name)
(base/'reference-input-checks.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'passed':len(rows),'model_forwards':0}))
