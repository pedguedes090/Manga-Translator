import json, os, shutil
from PIL import Image, ImageDraw

BASE = os.path.join('temp_sessions')
SID_PRE = 'a0000000-0000-0000-0000-000000000001'
SID_POST = 'a0000000-0000-0000-0000-000000000002'

def make_manga_image(path, w=800, h=1100, variant=0):
    img = Image.new('RGB', (w, h), (245, 242, 238))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, w-20, h//2-10], outline=(30,30,30), width=4)
    d.rectangle([20, h//2, w-20, h-20], outline=(30,30,30), width=4)
    bubbles = [
        (90, 90, 340, 200),
        (430, 90, 700, 210),
        (90, h//2+70, 360, h//2+220),
        (450, h//2+60, 720, h//2+230)
    ]
    for i, (x1,y1,x2,y2) in enumerate(bubbles):
        d.ellipse([x1, y1, x2, y2], fill=(255,255,255), outline=(20,20,20), width=3)
        for r in range(5 + (i%3)):
            yy = y1 + 26 + r*16
            d.line([x1+24, yy, x2-24 - (i*7 % 60), yy], fill=(40,40,40), width=6)
        d.polygon([(x1+60, y2-6), (x1+60, y2+24), (x1+110, y2-2)], fill=(255,255,255), outline=(20,20,20))
    if variant == 1:
        d.rectangle([0,0,w,h], fill=(232, 228, 236))
    img.save(path, 'JPEG', quality=92)
    return img

# PREVIEW
for s in (SID_PRE, SID_POST):
    p = os.path.join(BASE, s)
    if os.path.isdir(p):
        shutil.rmtree(p)
    os.makedirs(p)

make_manga_image(os.path.join(BASE, SID_PRE, 'page_0.jpg'), variant=0)
make_manga_image(os.path.join(BASE, SID_PRE, 'page_1.jpg'), variant=1)
preview_data = {
    'all_ocr_results': [
        {'name': 'page1', 'blocks': [
            {'text': 'konnichiha sekai', 'bbox': [90, 90, 340, 200]},
            {'text': 'ohayou', 'bbox': [430, 90, 700, 210]},
            {'text': 'sayounara', 'bbox': [90, 620, 360, 780]},
        ]},
        {'name': 'page2', 'blocks': [
            {'text': 'arigatou', 'bbox': [450, 610, 720, 790]},
        ]},
    ],
    'selected_font': 'animeace_',
    'source_lang': 'ja',
}
with open(os.path.join(BASE, SID_PRE, 'session.json'), 'w', encoding='utf-8') as f:
    json.dump(preview_data, f, ensure_ascii=False, indent=2)

# POSTRENDER
orig = make_manga_image(os.path.join(BASE, SID_POST, 'page_0.jpg'), variant=0)
rend = orig.copy()
dr = ImageDraw.Draw(rend)
texts = ['Xin chao the gioi', 'Chao buoi sang', 'Tam biet nhe', 'Cam on ban']
boxes = [(90, 90, 340, 200), (430, 90, 700, 210), (90, 620, 360, 780), (450, 610, 720, 790)]
for (x1,y1,x2,y2), t in zip(boxes, texts):
    dr.rounded_rectangle([x1+8, y1+8, x2-8, y2-8], radius=6, fill=(255,255,255), outline=(94,22,117), width=2)
    dr.text((x1+20, y1 + (y2-y1)//2 - 8), t, fill=(20,20,20))
rend.save(os.path.join(BASE, SID_POST, 'page_0_rendered.jpg'), 'JPEG', quality=92)

post_data = {
    'all_ocr_results': [
        {'name': 'page1', 'blocks': [
            {'text': 'konnichiha sekai', 'bbox': [90, 90, 340, 200]},
            {'text': 'ohayou', 'bbox': [430, 90, 700, 210]},
            {'text': 'sayounara', 'bbox': [90, 620, 360, 780]},
            {'text': 'arigatou', 'bbox': [450, 610, 720, 790]},
        ]},
    ],
    'render_plan': [
        {'name': 'page1',
         'erase_regions': [[90, 90, 340, 200], [430, 90, 700, 210], [90, 620, 360, 780], [450, 610, 720, 790]],
         'blocks': [
            {'text': 'konnichiha sekai', 'translated': 'Xin chao the gioi', 'bbox': [90, 90, 340, 200]},
            {'text': 'ohayou', 'translated': 'Chao buoi sang', 'bbox': [430, 90, 700, 210]},
            {'text': 'sayounara', 'translated': 'Tam biet nhe', 'bbox': [90, 620, 360, 780]},
            {'text': 'arigatou', 'translated': 'Cam on ban', 'bbox': [450, 610, 720, 790]},
         ]},
    ],
    'selected_font': 'animeace_',
    'source_lang': 'ja',
}
with open(os.path.join(BASE, SID_POST, 'session.json'), 'w', encoding='utf-8') as f:
    json.dump(post_data, f, ensure_ascii=False, indent=2)
print('PRE:', SID_PRE)
print('POST:', SID_POST)
