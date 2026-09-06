#!/usr/bin/env python3
"""Specialize the two reviewed spatial reshapes in the VAE encoder graph."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
try:
 from package_model import sha256
 from validate_img2img_encoder import verify
except ImportError:
 from tools.package_model import sha256
 from tools.validate_img2img_encoder import verify

BASE_PARAM='75d493995616b451e51ddecc0dc352a3f200557baab3f98d742cc374ed6d0977'
BASE_BIN='7fa2441a94886d9a1d44dbafe4fbac9211190e342b1cac171acb94c0faf517ce'

SHAPE_REPLACEMENTS={(512,384):{'reshape_77':['0=3072','1=512'],'reshape_78':['0=64','1=48','2=512']},
                    (1024,1024):{'reshape_77':['0=16384','1=512'],'reshape_78':['0=128','1=128','2=512']}}

def specialize_lines(lines,width,height):
 replacements=SHAPE_REPLACEMENTS.get((width,height))
 if replacements is None:raise ValueError('Encoder specialization shape is not registered')
 result=list(lines);changes=[];expected={'reshape_77':['0=16','1=512'],'reshape_78':['0=4','1=4','2=512']}
 for i,line in enumerate(result):
  parts=line.split()
  if parts and parts[0]=='Reshape':
   if parts[1] not in expected or parts[6:]!=expected[parts[1]]:raise ValueError('Unreviewed encoder reshape')
   changed=' '.join(parts[:6]+replacements[parts[1]]);changes.append({'before':line,'after':changed});result[i]=changed
 if len(changes)!=2:raise ValueError('Expected exactly two encoder reshapes')
 return result,changes

def reviewed_dimensions(width,height):
 if (width,height)!=(512,384):raise ValueError('Reviewed large encoder specialization is pinned to 512x384')

def specialize_candidate(template,reference,output,width,height):
 template,reference,output=map(Path,(template,reference,output))
 if output.exists():raise ValueError('Use a new output directory')
 base=verify(template);ref=json.loads((reference/'fixture.json').read_text())
 if sha256(template/'head.ncnn.param')!=BASE_PARAM or sha256(template/'head.ncnn.bin')!=BASE_BIN:raise ValueError('Unreviewed encoder template')
 if (ref.get('width'),ref.get('height'))!=(width,height) or any(ref.get(k)!=base.get(k) for k in ['component','weights','official_revision','vae_config_sha256','official_source_sha256','distribution_source_sha256','posterior','packing','encoder_bn','decoder_inverse_bn_eps']):raise ValueError('Large reference identity differs')
 for entry in [ref['rgb'],*ref['inputs'].values(),*ref['expected'].values()]:
  path=reference/entry['file'];expected=int(np.prod(entry['shape']))*(1 if entry is ref['rgb'] else 4)
  if Path(entry['file']).name!=entry['file'] or path.stat().st_size!=expected or sha256(path)!=entry['sha256']:raise ValueError('Large reference tensor identity differs')
 lines,changes=specialize_lines((template/'head.ncnn.param').read_text().splitlines(),width,height)
 output.mkdir();(output/'head.ncnn.param').write_text('\n'.join(lines)+'\n');(output/'head.ncnn.bin').symlink_to((template/'head.ncnn.bin').resolve())
 for name in ['fixture.json','input.rgb','in0.f32','out0.f32','out1.f32','out2.f32']:(output/name).symlink_to((reference/name).resolve())
 conversion={'method':'reviewed_encoder_spatial_reshape_specialization','width':width,'height':height,'changes':changes,'template_param_sha256':BASE_PARAM,'template_bin_sha256':BASE_BIN,'reference_fixture_sha256':sha256(reference/'fixture.json')}
 (output/'conversion.json').write_text(json.dumps(conversion,indent=2)+'\n')
 return {'output':str(output),'param_sha256':sha256(output/'head.ncnn.param'),'changes':len(changes)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--template',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 ref=json.loads((a.reference/'fixture.json').read_text());reviewed_dimensions(ref.get('width'),ref.get('height'))
 print(json.dumps(specialize_candidate(a.template,a.reference,a.output,512,384)))
if __name__=='__main__':main()
