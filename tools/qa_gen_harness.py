import base64
b64 = open('tools/qa_img_b64.txt').read().strip()
src = open('static/qa/harness.html', encoding='utf-8').read()
for mode in ('preview', 'postrender'):
    out = src.replace('__IMG_B64__', b64).replace('__MODE__', mode)
    open('static/qa/harness_' + mode + '.html', 'w', encoding='utf-8').write(out)
print('regenerated')
