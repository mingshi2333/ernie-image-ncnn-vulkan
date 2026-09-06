#!/usr/bin/env python3
"""Freeze and verify exact cross-port inputs; never run generation or evaluation.

Paths are relative to the corpus root. Image shape is WH; packed noise is
little-endian FP32 contiguous CHW [128,H/16,W/16]. Seed records provenance only.
Verification authenticates the manifest-listed inputs only. Additional reports or
other unlisted files are not authenticated and must never be consumed as inputs.
The committed summary pins the manifest hash; self hashes detect accidents,
not an adversary able to replace both the manifest and its trusted anchor.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

REQUIRED = ('id','split','prompt_source','shape','steps','pe','dtype_by_stage','model_identity')
SPLITS = {'formal','development','performance','img2img'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _validate_case(c):
    for key in REQUIRED:
        if key not in c or c[key] is None:
            raise ValueError(f'missing {key}')
    if not isinstance(c['id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', c['id']):
        raise ValueError('unsafe case id')
    if c['split'] not in SPLITS:
        raise ValueError('unknown split')
    shape=c['shape']
    if not isinstance(shape,list) or len(shape)!=2 or any(type(x)!=int or x<16 or x>2048 or x%16 for x in shape) or math.prod(shape)>2097152:
        raise ValueError('invalid image dimensions')
    if type(c['steps'])!=int or c['steps']<=0:
        raise ValueError('invalid steps')
    if not isinstance(c['pe'],dict) or type(c['pe'].get('enabled'))!=bool:
        raise ValueError('explicit PE enabled required')
    if c['pe']['enabled']:
        for key in ('sampling','max_input_tokens','max_new_tokens','template_sha256','temperature','top_p',
                    'stop_at_eos','add_generation_prompt','template_source'):
            if key not in c['pe']:raise ValueError(f'missing PE {key}')
        pe=c['pe']
        if pe['sampling']!='greedy':raise ValueError('paired PE must use greedy')
        for key in ('max_input_tokens','max_new_tokens'):
            if type(pe[key]) is not int or not 1<=pe[key]<=2048:
                raise ValueError(f'PE {key} must be an integer in [1,2048]')
        for key in ('stop_at_eos','add_generation_prompt'):
            if type(pe[key]) is not bool:raise ValueError(f'PE {key} must be boolean')
        if not isinstance(pe['template_source'],str) or not pe['template_source'].strip():
            raise ValueError('PE template_source must be nonempty text')
        if not isinstance(pe['template_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',pe['template_sha256']):
            raise ValueError('PE template_sha256 must be a lowercase SHA256')
        for key,expected in (('temperature',0),('top_p',1)):
            if type(pe[key]) not in (int,float) or not math.isfinite(pe[key]) or pe[key]!=expected:
                raise ValueError(f'PE greedy {key} must explicitly equal {expected}')
    if not isinstance(c['dtype_by_stage'],dict) or any(c['dtype_by_stage'].get(k) not in ('fp32','fp16','bf16') for k in ('text','dit','vae','scheduler')):
        raise ValueError('explicit stage precision required')
    if not c['model_identity'] or not c['prompt_source']:raise ValueError('empty identity')
    if c.get('mode')=='img2img' or c['split']=='img2img':
        if not isinstance(c.get('strength'),(float,int)) or isinstance(c.get('strength'),bool) or not 0<=c['strength']<=1:
            raise ValueError('invalid or missing strength')
        if not c.get('resize_policy') or not c.get('image_source'):raise ValueError('explicit image source/resize policy required')


def _safe_file(root, relative):
    if not isinstance(relative,str):raise ValueError('invalid file path')
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('unsafe input path')
    current=root
    for part in p.parts:
        current=current/part
        if current.is_symlink():raise ValueError('input symlinks forbidden')
    if not current.is_file():raise ValueError(f'missing input {relative}')
    return current


def _inventory(root):
    return [{'path':p.relative_to(root).as_posix(),'sha256':digest(p.read_bytes()),'size_bytes':p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json']


def freeze_inputs(spec:dict, output:Path)->dict:
    """Create a new corpus in an empty directory, refusing overwrite and bad IDs."""
    import numpy as np
    output=Path(output)
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError('freeze requires a new or empty output directory')
    cases=copy.deepcopy(spec.get('cases'))
    if not isinstance(cases,list) or not cases:raise ValueError('nonempty cases required')
    seen=set()
    for c in cases:
        _validate_case(c)
        if c['id'] in seen:raise ValueError('duplicate case id')
        seen.add(c['id'])
        if ('seed' in c)+('noise_file' in c)!=1:
            raise ValueError('exactly one noise source required: seed or noise_file')
        if ('prompt' in c)+('prompt_file' in c)!=1:
            raise ValueError('exactly one prompt source required: prompt or prompt_file')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Stage next to destination; invalid input never leaves a half-frozen corpus.
    with tempfile.TemporaryDirectory(prefix='.corpus-freeze-',dir=output.parent) as temp:
        stage=Path(temp)
        for c in cases:
            target=stage/c['id'];target.mkdir()
            data=Path(c.pop('prompt_file')).read_bytes() if 'prompt_file' in c else c.pop('prompt').encode('utf-8')
            data.decode('utf-8',errors='strict')
            (target/'prompt.txt').write_bytes(data)
            c.update(prompt_path=f"{c['id']}/prompt.txt",prompt_sha256=digest(data),prompt_size_bytes=len(data),shape_order='WH')
            w,h=c['shape'];shape=[128,h//16,w//16]
            if 'noise_file' in c:
                noise=Path(c.pop('noise_file')).read_bytes();origin='saved_bytes'
            else:
                noise=np.random.Generator(np.random.PCG64(c['seed'])).standard_normal(shape).astype('<f4').tobytes();origin='numpy.PCG64.standard_normal.float64_cast_float32'
            if len(noise)!=math.prod(shape)*4 or not np.isfinite(np.frombuffer(noise,dtype='<f4')).all():raise ValueError('invalid FP32 noise')
            (target/'initial.f32').write_bytes(noise)
            c.update(noise_path=f"{c['id']}/initial.f32",noise_sha256=digest(noise),noise_size_bytes=len(noise),noise_shape=shape,noise_layout='CHW',noise_dtype='<f4',noise_generator=origin)
            if c.get('mode')=='img2img' or c['split']=='img2img':
                from PIL import Image
                source=Path(c.pop('input_image_file'));raw=source.read_bytes()
                (target/'input-image').write_bytes(raw)
                with Image.open(source) as im:
                    im=im.convert('RGB');rgb=im.tobytes();rgb_shape=[im.height,im.width,3]
                (target/'decoded.rgb').write_bytes(rgb)
                c.update(input_image_path=f"{c['id']}/input-image",input_image_sha256=digest(raw),input_image_size_bytes=len(raw),decoded_rgb_path=f"{c['id']}/decoded.rgb",decoded_rgb_sha256=digest(rgb),decoded_rgb_size_bytes=len(rgb),decoded_rgb_shape=rgb_shape,decoded_rgb_layout='HWC',decoded_rgb_dtype='uint8')
        for entry in spec.get('attachments',[]):
            rel=Path(entry['path'])
            if rel.is_absolute() or '..' in rel.parts or rel.name=='manifest.json' or (stage/rel).exists():raise ValueError('unsafe/duplicate attachment')
            (stage/rel).parent.mkdir(parents=True,exist_ok=True)
            (stage/rel).write_bytes(Path(entry['source']).read_bytes())
        (stage/'protocol.json').write_bytes(canonical(spec.get('protocol',{'status':'inputs_only_no_evaluation'}))+b'\n')
        m=dict(schema_version=1,status='frozen_inputs_no_formal_results',cases=cases,files=_inventory(stage))
        m['manifest_sha256']=digest(canonical(m))
        (stage/'manifest.json').write_bytes(canonical(m)+b'\n')
        verify_inputs(m,stage)
        if output.exists():output.rmdir()
        stage.rename(output)
    return m


def verify_inputs(manifest:dict,root:Path)->None:
    """Authenticate listed inputs; reject mutation, input symlinks and duplicate IDs.

    Unlisted files (for example later reports) are outside this identity contract.
    Consumers must resolve inputs exclusively from verified manifest entries.
    """
    import numpy as np
    root=Path(root)
    if manifest.get('schema_version')!=1:raise ValueError('unsupported schema')
    m=copy.deepcopy(manifest);expected=m.pop('manifest_sha256',None)
    if digest(canonical(m))!=expected:raise ValueError('manifest metadata checksum mismatch')
    seen=set();files={}
    for entry in manifest.get('files',[]):
        name=entry['path']
        if name in files:raise ValueError('duplicate file path')
        p=_safe_file(root,name);data=p.read_bytes()
        if len(data)!=entry['size_bytes'] or digest(data)!=entry['sha256']:raise ValueError(f'changed input {name}')
        files[name]=entry
    if 'protocol.json' not in files:raise ValueError('missing protocol')
    for c in manifest['cases']:
        _validate_case(c)
        if c['id'] in seen:raise ValueError('duplicate case id')
        seen.add(c['id'])
        w,h=c['shape']
        if c.get('shape_order')!='WH' or c.get('noise_shape')!=[128,h//16,w//16] or c.get('noise_layout')!='CHW' or c.get('noise_dtype')!='<f4':raise ValueError('noise contract mismatch')
        keys=['prompt','noise']
        if c.get('mode')=='img2img' or c['split']=='img2img':keys+=['input_image','decoded_rgb']
        for key in keys:
            rel=c.get(key+'_path');entry=files.get(rel)
            if not entry or entry['sha256']!=c.get(key+'_sha256') or entry['size_bytes']!=c.get(key+'_size_bytes'):raise ValueError('missing/inconsistent input identity')
        if c['noise_size_bytes']!=128*(h//16)*(w//16)*4:raise ValueError('wrong noise bytes')
        if not np.isfinite(np.frombuffer(_safe_file(root,c['noise_path']).read_bytes(),'<f4')).all():raise ValueError('nonfinite noise')
        _safe_file(root,c['prompt_path']).read_bytes().decode('utf-8')
        if 'decoded_rgb' in keys:
            from PIL import Image
            with Image.open(_safe_file(root,c['input_image_path'])) as im:
                im=im.convert('RGB')
                if c.get('decoded_rgb_shape')!=[im.height,im.width,3] or im.tobytes()!=_safe_file(root,c['decoded_rgb_path']).read_bytes():raise ValueError('decoded RGB mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',type=Path);p.add_argument('--spec',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.verify:
        verify_inputs(json.loads((a.verify/'manifest.json').read_text()),a.verify);print('All frozen input identities verified')
    elif a.spec and a.output:
        m=freeze_inputs(json.loads(a.spec.read_text()),a.output);print(m['manifest_sha256'])
    else:p.error('provide --verify ROOT or --spec JSON --output ROOT')

if __name__=='__main__':main()

# Corpus authoring is deliberately separate from freeze_inputs: no tokenizer or
# network dependency is needed to verify an already frozen corpus.
def build_port_corpus(project:Path, output:Path, summary_path:Path):
    import subprocess
    import urllib.request
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
    from transformers import AutoTokenizer
    project=project.resolve()
    lock=json.loads((project/'sources.lock.json').read_text())
    acceptance=json.loads((project/'docs/superpowers/plans/surpass-acceptance.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(project/'models/tokenizer',local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='corpus-author-') as tmp:
        work=Path(tmp);attachments=[]
        def attach(path, data):
            p=work/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            attachments.append(dict(path=path,source=str(p)));return p
        def fetch(url):
            with urllib.request.urlopen(url,timeout=60) as response:return response.read()
        attach('provenance/sources.lock.json',(project/'sources.lock.json').read_bytes())
        attach('provenance/acceptance.json',(project/'docs/superpowers/plans/surpass-acceptance.json').read_bytes())
        attach('provenance/acceptance_manifest.py',Path(__file__).read_bytes())
        runner=attach('provenance/ernie-tokenize.snapshot',(project/'build/ernie-tokenize').read_bytes());runner.chmod(0o755)
        model_manifest=json.loads((project/'models/tokenizer/manifest.json').read_text())
        if model_manifest['revision']!=lock['official_model']['revision']:raise ValueError('tokenizer revision differs')
        for name,sha in model_manifest['files'].items():
            data=(project/'models/tokenizer'/name).read_bytes()
            if digest(data)!=sha:raise ValueError('tokenizer hash mismatch')
        attach('provenance/tokenizer-manifest.json',(project/'models/tokenizer/manifest.json').read_bytes())
        def tokens(prompt,path):
            p=attach(path,prompt.encode('utf-8'))
            raw=tokenizer(prompt,add_special_tokens=True,truncation=False,padding=False)['input_ids']
            official=tokenizer(prompt,add_special_tokens=True,truncation=True,padding=False)['input_ids']
            run=subprocess.run([str(runner),str(project/'models/tokenizer'),str(p)],capture_output=True,text=True,timeout=30)
            native=json.loads(run.stdout) if run.returncode==0 else None
            if native!=official:raise ValueError(f'token mismatch {path}: {run.stderr}')
            return dict(untruncated_token_ids=raw,token_ids=official,token_count=len(official),untruncated_token_count=len(raw),bos_id=tokenizer.bos_token_id,add_special_tokens=True,official_truncation=True,native_return_code=run.returncode,native_token_ids=native,native_stderr=run.stderr,capacity_policy='truncate_to_2048' if len(raw)>2048 else 'no_truncation',prompt_path=path)
        prompts=[]
        hist=[('apple','outputs/text-prompt-apple-v1/reference/fixture.json',15),('english40','outputs/pipeline1024-long-s64-v1/reference/text/fixture.json',40),('chinese32','outputs/pipeline1024-chinese-s64-fp16-v1/reference/text/fixture.json',32)]
        for label,path,count in hist:
            d=json.loads((project/path).read_text());prompt=d['prompt'];e=tokens(prompt,f'prompts/{label}.txt')
            if e['token_count']!=count or e['token_ids']!=d['ids']:raise ValueError('historical prompt changed')
            prompts.append(dict(name=label,prompt=prompt,prompt_source=path,cohort='historical_regression',tokenizer=e))
        for i in range(1,6):
            name='prompt'+(str(i) if i>1 else '')+'.txt';url=f"https://raw.githubusercontent.com/futz12/ernie-image-ncnn-vulkan/{lock['historical_port']['revision']}/assets/{name}"
            raw=fetch(url);prompt=raw.decode('utf-8');e=tokens(prompt,f'prompts/reference-{i}.txt')
            # Never strip BOM, CRLF, whitespace, or trailing newlines.
            if (work/e['prompt_path']).read_bytes()!=raw:raise ValueError('raw prompt changed')
            prompts.append(dict(name=f'reference-{i}',prompt=prompt,prompt_source=url,cohort='public_regression',source_sha256=digest(raw),tokenizer=e))
        independent=[
('chinese-bookshop','一张正面平视的写实街景照片：一家只有一个入口的小书店，木质招牌清晰写着“山海书店”四个字。门左边是一盆绿色龟背竹，门右边摆着两把红椅子。橱窗内整齐陈列三本封面分别为蓝色、黄色、白色的书。招牌下方另有一张白纸，黑字写“今日营业 09:00–18:00”。下午柔和自然光，没有路人，没有其他商店招牌。'),
('english-menu','Front-facing product photograph of one cream cafe menu on a dark green wall. The large heading reads "HARBOR CAFE". Below it are exactly three clearly separated rows: "ESPRESSO 3", "LATTE 5", and "TEA 4". One thin gold line separates the heading from the rows. A single white cup sits at the bottom left on a wooden counter; two brown coffee beans lie to its right. Clean readable black lettering, even natural light, no extra words or prices.'),
('japanese-station','日本の小さな駅の構内を正面から撮影した写真。白い案内板が一枚あり、上段に「さくら駅」、中段に「出口 →」、下段に「1番線 ←」と大きく読みやすく書かれている。案内板の左には青いベンチが一台、右には丸い時計が一個あり、時計は三時を示す。人は二人だけで、一人は黄色い傘、もう一人は赤い鞄を持つ。昼の柔らかい光。'),
('three-drinks','Studio photograph of exactly three identical transparent cylindrical glasses in one horizontal row on a white table. From left to right the drinks are red strawberry juice, yellow lemonade, and blue soda. Each glass has one white straw, exactly two visible ice cubes, and the same liquid level. No other glasses, bottles, fruit, text, or reflections that resemble additional glasses. Soft frontal lighting and pale gray background.'),
('left-right','A straight-on still life with exactly one red wooden cube on the left, one small white sphere in the center, and one blue ceramic pyramid on the right. The sphere is separated from both other objects by equal gaps. All three objects rest on one matte beige table and cast shadows toward the back right. A small card below the cube reads "LEFT"; a card below the pyramid reads "RIGHT". No overlapping objects.'),
('occlusion','A realistic tabletop photograph of exactly three objects: a large green vase in the back, a red apple in front of the vase, and a small yellow cup in front of the apple but shifted right. The apple visibly covers the lower left quarter of the vase; the cup covers only the lower right edge of the apple. The vase rim, apple stem, and entire cup handle remain visible. Neutral gray background, soft side light, no text.'),
('four-panel','A four-panel comic arranged as exactly two columns and two rows, with clear black gutters. The same young adult woman with short black hair, round glasses, a yellow jacket, and a red backpack appears once in every panel. Top left: she finds one blue umbrella on a bench. Top right: she opens that umbrella in rain. Bottom left: she shares the umbrella with one small white dog. Bottom right: she and the dog stand beside a doorway as sunlight returns. No captions, no speech bubbles, consistent clothing and face.'),
('room-plan','Orthographic top-down illustration of one rectangular room with the north wall at the top. One blue two-seat sofa is centered against the north wall. A round wooden table stands in the center with exactly three white chairs around it. One single bed with a green blanket occupies the southeast corner. The door is in the southwest corner; two windows are evenly spaced on the west wall. A red rectangular rug lies only under the table. Label the top wall "NORTH". No other furniture.'),
('night-street','Low-contrast realistic night photograph of one narrow wet street. Exactly two pedestrians in dark gray coats walk away in the distance. One dim amber streetlamp stands on the left; a single blue shop window glows on the right. Its sign reads "OPEN". The wet asphalt has faint amber and blue reflections, but no headlights, bright neon, or blown highlights. Retain subtle detail in the shadowed brick walls and folds of both coats. Quiet overcast sky.'),
('metal-glass','High-detail still life on a dark slate tabletop: one polished stainless steel kettle at left, one empty transparent glass tumbler at center, and one brushed copper sphere at right. The kettle has a single curved black handle and reflects a rectangular softbox. The tumbler has a thick base, a visible rear rim, and clear refraction of one white stripe painted on the table. The copper sphere shows a warm gradient without mirror-like reflections. No text, no extra vessels.'),
('natural-portrait','Natural-window-light portrait photograph of one adult woman with brown eyes, short curly dark hair, light freckles, and a calm closed-mouth smile. She wears a plain blue cotton shirt with exactly three visible white buttons. Her left hand rests on one closed red book, and her right hand rests on her lap; both hands have five fingers. A single window is outside the frame on the left, casting soft light across her face. Neutral beige background, realistic skin texture, no jewelry or lettering.'),
('motion-pose','Sports photograph of exactly one adult male dancer mid-leap in an empty studio. His left knee is bent forward, right leg extends backward, left arm reaches upward, and right arm extends sideways. He wears one red T-shirt, black shorts, and two white sneakers. Show the whole body with both hands and both feet inside the frame, five fingers on each hand, believable joints, and a faint shadow below. One horizontal wooden barre crosses the background, no mirrors or text.'),
('line-architecture','Precise black ink architectural line drawing on white paper of a two-story corner library. The front facade has exactly three tall windows on the upper floor and one centered double door below. The side facade has exactly two windows on each floor. A small sign over the door reads "LIBRARY". One bicycle leans against the left corner. Two-point perspective, consistent straight edges, varied line weight, no color, gray wash, people, or additional buildings.'),
('watercolor-flowers','Watercolor illustration on textured off-white paper of exactly five flowers in one blue ceramic vase: two red poppies, two yellow daisies, and one purple iris centered above them. There are exactly four visible green leaves. Transparent pigments, soft bleeding edges, small areas of unpainted paper, and a faint lavender shadow to the right. The vase sits on one pale wooden shelf. No extra buds, insects, lettering, or background objects.'),
('bilingual-poster','一张竖版双语信息海报，米白色背景，深蓝色清晰文字。顶部中文标题为“城市读书日”，下一行英文为“CITY READING DAY”。中间画一本打开的蓝色书和一盏黄色台灯，只有这两个图形。下方分三行依次写“9月21日 / SEPTEMBER 21”、“10:00–16:00”、“中央图书馆 / CENTRAL LIBRARY”。底部写“免费入场 / FREE ENTRY”。各行独立居中，边距均匀，不添加二维码、徽标或其他文字。'),
]
        room_sections=[
'The north wall contains three equally spaced windows with white wooden frames. Each window has six rectangular panes arranged in two columns and three rows. All windows show the same overcast garden, and no window contains a reflected person.',
'Against the west wall stands one walnut bookcase with four shelves. The top shelf holds exactly five blue books, the next holds four red books, the third holds three green books, and the bottom holds two white storage boxes with brass handles.',
'At the center is one oval oak table with four legs. Exactly six chairs surround it, three on the north side and three on the south side. Every chair has a dark green upholstered seat, a wooden back, and four visible supports where perspective permits.',
'The table holds one white ceramic teapot, two white cups on matching saucers, one closed red notebook, and one transparent vase containing exactly three yellow tulips. Keep the notebook near the west end and the flowers near the east end, without duplicating any object.',
'In the southeast corner place one gray two-seat sofa with exactly two blue cushions and one folded cream blanket draped over its right arm. Its left arm remains uncovered. Behind it hangs one rectangular painting showing two white sailboats on a blue lake.',
'The southwest corner contains one floor lamp with a cylindrical white shade and a round black base. A small circular side table stands beside it and carries one analog clock. The clock face is white, its numerals are black, and its hands indicate ten past ten.',
'The east wall has one closed wooden door painted muted red with a brass handle on the left side. Above the door a small white plaque clearly reads "READING ROOM" in black uppercase letters. A second plaque below it reads "静阅室" with the three Chinese characters intact.',
'The floor uses pale oak planks running east to west. One rectangular cream rug lies beneath the central table and all six chairs. It has exactly two thin blue border lines and no central pattern. The rug must not extend under the sofa or the bookcase.',
'A single adult librarian stands between the bookcase and the north windows. She wears a plain navy cardigan over a white shirt, gray trousers, and brown flat shoes. She holds one open green book in both hands; show natural wrists and five fingers on each hand.',
'One orange cat sleeps on the left sofa cushion, curled into a compact shape with one visible tail and two visible ears. Keep it clearly separate from the blue cushion behind it. No other animals appear in the room, the garden, the painting, or reflections.',
'One square wall calendar hangs to the left of the red door. Its large heading reads "SEPTEMBER" and the number "21" is circled once in red. Render the calendar as a flat sheet attached at its upper edge with one small black clip.',
'Place one cylindrical terracotta pot on the floor under the middle north window. The plant has one central stem and exactly seven broad green leaves. The pot casts a soft short shadow toward the southeast and has no decorative text or duplicated rim.',
'The ceiling is white with one central pendant lamp composed of a single frosted glass globe on a black cord. The globe is switched off because daylight provides all illumination. Avoid multiple ceiling lights, recessed spotlights, dramatic beams, or glowing furniture.',
'Use a realistic wide-angle interior photograph from the southwest doorway at adult eye level. Vertical wall edges stay straight. The camera sees the north and east walls, most of the west bookcase, the complete table, and the sofa without impossible perspective.',
'The palette combines pale oak, muted red, navy blue, dark green, white ceramics, and warm gray upholstery. Preserve the specifically assigned colors of the books, flowers, cushions, door, notebook, and plant pot even where objects overlap in the perspective.',
'Daylight enters only through the three north windows. Shadows remain soft and point generally southward. Preserve readable surface texture in the wood grain, cloth fibers, paper edges, and frosted glass without oversharpening, grain filters, excessive bloom, or artificial lens flare.',
'The table surface is clean and dry. The transparent vase visibly contains clear water up to one third of its height, with three distinct flower stems extending below the waterline. The teapot lid is closed and its spout points toward the two cups.',
'Keep generous walking space between the table and bookcase and between the table and sofa. Neither the door nor its handle is blocked. All furniture legs contact the floor naturally, and objects resting on surfaces cast consistent contact shadows.',
'There are no electronic screens, phones, computers, cables, power strips, bottles, food, posters, extra rugs, extra doors, stairs, or mirrors. Do not introduce people into the garden. The librarian is the only human figure and the cat is the only animal.',
'Text appears only on the two doorway plaques and the calendar. Preserve the exact requested words and characters. All book spines are solid colors without readable titles, the notebook has no logo, and the cups and teapot have no painted markings.',
'Every object has a coherent outline and a plausible material. Glass remains transparent, brass reflects warm highlights, wood has subtle directional grain, ceramic appears opaque, and cotton stays matte. Avoid melting edges, repeated patterns, duplicate handles, and floating fragments.',
'Balance the composition so the oval table is the central focal point, with the librarian toward the left third and the red doorway toward the right third. The painting is fully inside the frame. Include the floor in the foreground and a narrow band of ceiling.',
]
        long='Create one detailed realistic reading-room interior. Follow every numbered constraint in the same single image.\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(room_sections))
        independent.append(('long-interior',long))
        for name,prompt in independent:
            e=tokens(prompt,f'prompts/{name}.txt');prompts.append(dict(name=name,prompt=prompt,prompt_source='self-authored:port-corpus-v1/'+name,cohort='independent_formal',tokenizer=e))
        if not 1025<=prompts[-1]['tokenizer']['untruncated_token_count']<=2048:raise ValueError(f"long prompt length {prompts[-1]['tokenizer']['untruncated_token_count']}")
        boundaries=[]
        for n in [31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049]:
            # Determine the raw length with the actual tokenizer, not characters.
            text=' '.join(['word']*(n-1));e=tokens(text,f'boundaries/tokens-{n}.txt')
            if e['untruncated_token_count']!=n:raise ValueError(f'boundary length mismatch {n}')
            boundaries.append(dict(target_tokens=n,**e))
        attach('boundaries/results.json',canonical(dict(scope='official tokenizer call and original native encode() runner; no encoder/model execution',cases=boundaries)))
        # Photograph identity and permission are independent of generation results.
        tag=json.loads(fetch('https://api.github.com/repos/scikit-image/scikit-image/git/tags/23308914c880574fe7c716e2ce9aa698ab0d69b4'))
        rev=tag['object']['sha'];photo_url=f'https://raw.githubusercontent.com/scikit-image/scikit-image/{rev}/skimage/data/coffee.png'
        photo=attach('images/coffee.png',fetch(photo_url))
        license_url=f'https://raw.githubusercontent.com/scikit-image/scikit-image/{rev}/skimage/data/_fetchers.py'
        attach('provenance/photo-license-source.py',fetch(license_url))
        photo_source=dict(kind='photograph',author='Rachel Michetti',license='CC0-1.0',description='Coffee photograph courtesy of Pikolo Espresso Bar',url=photo_url,revision=rev,license_evidence=license_url,documentation='https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.coffee')
        images=[('photo',photo,photo_source,'A realistic still life of a white coffee cup and saucer on a wooden cafe table. Preserve the cup position and make the ambient light softly warm.')]
        for kind in ['typography','geometry']:
            im=Image.new('RGB',(1024,1024),'#f4f0e8');draw=ImageDraw.Draw(im)
            if kind=='typography':
                font=ImageFont.load_default(size=74)
                for xy,text in [((100,160),'CITY LIBRARY'),((100,350),'READ 21'),((100,540),'OPEN 10:00')]:draw.text(xy,text,fill='#132f54',font=font)
                draw.rectangle((90,720,930,735),fill='#c53b35')
                prompt='A clean cream information poster. Preserve exactly three dark blue text lines: "CITY LIBRARY", "READ 21", and "OPEN 10:00", with one red horizontal rule below them.'
            else:
                draw.rectangle((100,120,390,410),fill='#e63946');draw.ellipse((590,120,880,410),fill='#277da1');draw.polygon([(250,600),(100,890),(400,890)],fill='#f9c74f');draw.rectangle((600,600,890,890),fill='#43aa8b')
                prompt='A flat cream graphic with exactly four shapes in a two by two grid: red square upper left, blue circle upper right, yellow triangle lower left, green square lower right. Preserve the layout and solid colors.'
            p=work/f'{kind}.png';im.save(p)
            fixture=attach(f'images/{kind}.png',p.read_bytes())
            images.append((kind,fixture,dict(kind='self-created',license='CC0-1.0',generator='provenance/acceptance_manifest.py',pillow_version=pillow_version),prompt))
        attach('images/sources.json',canonical([dict(id=x[0],source=x[2]) for x in images]))
        pe=dict(enabled=True,sampling='greedy',temperature=0.0,top_p=1.0,max_input_tokens=2048,max_new_tokens=2048,template_sha256=digest((project/'models/pe-tokenizer/chat_template.jinja').read_bytes()),template_source='official Turbo prompt_enhancer/tokenizer/chat_template.jinja',add_generation_prompt=False,stop_at_eos=True)
        attach('provenance/pe-chat-template.jinja',(project/'models/pe-tokenizer/chat_template.jinja').read_bytes())
        model=dict(official_model=lock['official_model'],reference_converted=lock['historical_converted_model'],weight_equivalence_status='unverified_requires_B2')
        cases=[]
        def case(identifier,split,prompt,source,shape,seed=42,**extra):
            return dict(id=identifier,split=split,prompt=prompt,prompt_source=source,shape=shape,seed=seed,steps=8,cfg=1,pe={'enabled':False},dtype_by_stage=dict(text='fp32',dit='fp32',vae='fp32',scheduler='fp32'),model_identity=model,**extra)
        shapes=[[512,512],[1024,1024],[1376,768]]
        for i,p in enumerate(prompts):
            for j,seed in enumerate([42,1234,2026]):
                cases.append(case(f'formal-{i:02d}-{seed}','formal',p['prompt'],p['prompt_source'],shapes[(i+j)%3],seed,prompt_index=i,seed_index=j,cohort=p['cohort'],tokenizer=p['tokenizer']))
        for i in range(3):
            p=prompts[i];cases.append(case(f'dev-{p["name"]}','development',p['prompt'],p['prompt_source'],[1024,1024],tokenizer=p['tokenizer']))
        d=json.loads((project/'outputs/text1080-s2048-v1/reference/fixture.json').read_text());e=tokens(d['prompt'],'prompts/development-1080.txt')
        if e['token_count']!=1080:raise ValueError('1080 fixture changed')
        cases.append(case('dev-1080','development',d['prompt'],'outputs/text1080-s2048-v1/reference/fixture.json',[512,384],tokenizer=e))
        c=case('dev-pe-apple','development','A red apple on a wooden table.','outputs/pipeline512x384-pe-fp32-v1/input-prompt.txt',[512,384]);c['pe']=pe;cases.append(c)
        for n in [33,65,1025]:
            e=next(x for x in boundaries if x['target_tokens']==n);text=(work/e['prompt_path']).read_text()
            cases.append(case(f'dev-boundary-{n}','development',text,f'actual-token-boundary:{n}',[512,384],tokenizer=e))
        perf=[(0,[512,512]),(0,[1024,1024]),(2,[1024,1024]),(23,[1376,768]),(0,[512,512]),(0,[1024,1024])]
        for i,(pi,shape) in enumerate(perf):
            p=prompts[pi];c=case(f'performance-{i}','performance',p['prompt'],p['prompt_source'],shape,tokenizer=p['tokenizer'])
            if i==4:c['pe']=pe
            if i==5:
                kind,img,source,prompt=images[0];c.update(prompt=prompt,prompt_source='self-authored:img2img-photo',mode='img2img',input_image_file=str(img),image_source=source,strength=.5,resize_policy={'mode':'stretch','width':1024,'height':1024,'filter':'bilinear','coordinate_transform':'half_pixel','antialias':False});c.pop('tokenizer')
            cases.append(c)
        for kind,img,source,prompt in images:
            for strength in [0,.25,.5,.75,1]:
                cases.append(case(f'img2img-{kind}-{int(strength*100):03d}','img2img',prompt,f'self-authored:img2img-{kind}',[1024,1024],mode='img2img',input_image_file=str(img),image_source=source,strength=strength,resize_policy={'mode':'stretch','width':1024,'height':1024,'filter':'bilinear','coordinate_transform':'half_pixel','antialias':False}))
        # Every case, including PE source text and image-conditioning prompts,
        # has explicit native and official image-tokenizer observations.
        for c in cases:
            if 'tokenizer' not in c:c['tokenizer']=tokens(c['prompt'],f'prompts/{c["id"]}.txt')
        protocol=dict(schema_version=1,status='frozen_inputs_no_results',acceptance=acceptance,shape_order='WH',noise_contract={'shape':'[128,H/16,W/16]','layout':'CHW','dtype':'<f4','distribution':'standard normal','seed_is_identity':False,'numpy_version':np.__version__},formal_prompt_count=24,formal_case_count=72,formal_independent_cases=48,formal_regression_cases=24,development_cases=8,performance_cases=6,img2img_cases=15,boundary_cases=21,formal_results_revealed=False,precision_contract='FP32 stages for initial paired baseline; other precision requires separately labeled matched experiments',performance={'warmups_per_port':1,'measured_pairs':5,'memory_pairs':3,'process_scope':'new_process','clock':'monotonic_ns','scope':'process_start_through_png_written','trace':False,'order':['AB','BA','AB','BA','AB'],'first_observation':'separate_not_OS_cold_cache','invalid_pair_policy':'invalidate entire pair for predeclared interference and replace entire pair; preserve invalid records','calibration_status':'pending_B2_no_model_run_in_B1'},img2img={'strengths':[0,.25,.5,.75,1],'zero_strength':'VAE encode/decode; no noise or denoise','positive_steps':'min(steps,max(1,floor(steps*strength+0.5)))','initial':'sigma[start]*saved_noise+(1-sigma[start])*encoded','encoder_distribution':'pending_official_and_reference_audit_in_P3','resizing':'explicit per case; decoded RGB is original resolution before resizing'},tokenizer={'official_manifest_sha256':digest((project/'models/tokenizer/manifest.json').read_bytes()),'native_runner_sha256':digest(runner.read_bytes()),'boundary_2049':'full raw text and raw IDs retained; both tested tokenizer calls truncate to2048; no full-model capacity claim'})
        m=freeze_inputs(dict(cases=cases,attachments=attachments,protocol=protocol),output)
        summary=dict(schema_version=1,corpus_root='outputs/port-corpus-v1',manifest_sha256=digest((output/'manifest.json').read_bytes()),manifest_content_sha256=m['manifest_sha256'],protocol_sha256=digest((output/'protocol.json').read_bytes()),counts={s:sum(c['split']==s for c in cases) for s in sorted(SPLITS)},boundary_count=len(boundaries),formal_prompt_count=len(prompts),noise_layout='CHW',noise_shape='[128,H/16,W/16]',noise_dtype='<f4',shape_order='WH',formal_results_revealed=False,prompts=[dict(name=p['name'],source=p['prompt_source'],sha256=digest(p['prompt'].encode()),token_count=p['tokenizer']['token_count'],cohort=p['cohort']) for p in prompts],images=[dict(id=x[0],source=x[2],sha256=digest(x[1].read_bytes())) for x in images],case_identities=[{k:c[k] for k in ('id','split','prompt_sha256','noise_sha256','shape')} for c in m['cases']],file_count=len(m['files']),total_file_bytes=sum(x['size_bytes'] for x in m['files']))
        summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        return summary
