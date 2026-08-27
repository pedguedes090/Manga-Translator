from PIL import Image
import os

def sample(path, pts):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    out = {'size': (w, h)}
    for name, (x, y) in pts.items():
        out[name] = img.getpixel((x, y))
    return out

# Desktop preview: topbar purple at y=12, toolbar white at y=60, canvas area mid
d = sample('debug_outputs/qa_preview_desktop.png', {
    'topbar': (720, 12),
    'toolbar': (720, 60),
    'canvas_region': (720, 500),
    'sidebar': (90, 400),
    'editor_panel': (1370, 400),
})
print('PREVIEW DESKTOP:', d)

d2 = sample('debug_outputs/qa_post_desktop.png', {
    'topbar': (720, 12),
    'toolbar': (720, 60),
    'canvas_region': (720, 500),
})
print('POST DESKTOP:', d2)

d3 = sample('debug_outputs/qa_preview_mobile.png', {
    'topbar': (195, 12),
    'toolbar': (195, 60),
    'canvas_region': (195, 500),
    'sidebar_strip': (50, 180),
})
print('PREVIEW MOBILE:', d3)

d4 = sample('debug_outputs/qa_post_mobile.png', {
    'topbar': (195, 12),
    'toolbar': (195, 60),
    'canvas_region': (195, 500),
})
print('POST MOBILE:', d4)
