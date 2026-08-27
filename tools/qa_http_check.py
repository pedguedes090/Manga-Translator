import requests, json
BASE = 'http://127.0.0.1:5055'
SID_PRE = 'a0000000-0000-0000-0000-000000000001'
SID_POST = 'a0000000-0000-0000-0000-000000000002'

r = requests.get(BASE + '/correction/' + SID_PRE, timeout=20)
html = r.text
checks = {
  'status': r.status_code,
  'toolbar': 'tool-select' in html and 'tool-add' in html and 'tool-reset' in html,
  'hintbar': 'corr-hints' in html,
  'toggle': 'btn-toggle-editor' in html,
  'badge': 'thumb-badge' in html,
  'mode_json': '"mode": "preview"' in html,
  'canvas_aria': 'role="img"' in html,
  'continue': 'btn-continue' in html,
  'no_postrender_footer': 'btn-rerender-one' not in html,
  'editor_close': 'btn-close-editor' in html,
  'shortcuts_id': 'shortcuts-hint' in html,
  'toast_role': 'role="status"' in html,
}
print('PREVIEW:', json.dumps(checks, ensure_ascii=False))

r2 = requests.get(BASE + '/postrender/' + SID_POST + '?img=0', timeout=20)
html2 = r2.text
checks2 = {
  'status': r2.status_code,
  'rerender_footer': 'btn-rerender-one' in html2 and 'btn-save-all' in html2 and 'btn-cancel' in html2,
  'hidden_add': 'tool-add' not in html2,
  'hidden_reset': 'tool-reset' not in html2,
  'title': 'Chỉnh sửa sau dịch' in html2,
  'mode_json': '"mode": "postrender"' in html2,
  'idx_json': '"postrenderImageIdx": 0' in html2,
  'translated_present': '"translated": "Xin chao the gioi"' in html2,
  'no_continue': 'btn-continue' not in html2,
}
print('POSTRENDER:', json.dumps(checks2, ensure_ascii=False))

r3 = requests.get(BASE + '/translate-result/' + SID_POST, timeout=20)
print('TRANSLATE_RESULT:', r3.status_code, 'edit-btn:', 'edit-btn' in r3.text, 'edit-note:', 'edit-note' in r3.text)

blocks = [
  {'text': 'konnichiha sekai', 'translated': 'Xin chao MOI', 'bbox': [100, 100, 350, 210]},
  {'text': 'ohayou', 'translated': 'Chao buoi sang', 'bbox': [430, 90, 700, 210]},
  {'text': 'sayounara', 'translated': 'Tam biet nhe', 'bbox': [90, 620, 360, 780]},
]
rr = requests.post(BASE + '/re-render-image', data={
  'session_id': SID_POST,
  'image_idx': '0',
  'blocks_json': json.dumps(blocks),
  'deleted_regions_json': json.dumps([[450, 610, 720, 790]]),
}, timeout=180)
print('RE-RENDER:', rr.status_code)
if rr.status_code == 200:
    d = rr.json()
    print('  keys:', sorted(d.keys()), '| blocks:', len(d.get('blocks', [])), '| data b64:', len(d.get('data', '')))
    print('  block0:', d.get('blocks', [{}])[0])
    print('  block1:', d.get('blocks', [{}])[1])
else:
    print('  body:', rr.text[:300])

rr2 = requests.post(BASE + '/re-render-image', data={
  'session_id': SID_POST,
  'image_idx': '0',
  'blocks_json': json.dumps([{'text': 'x', 'translated': 'y', 'bbox': [500, 500, 100, 100]}]),
  'deleted_regions_json': '[]',
}, timeout=60)
print('RE-RENDER invalid bbox:', rr2.status_code, rr2.text[:120])

rra = requests.post(BASE + '/re-render-all', data={
  'session_id': SID_POST,
  'dirty_indices_json': '[0]',
}, timeout=180)
print('RE-RENDER-ALL:', rra.status_code, 'edit-btn:', 'edit-btn' in rra.text)
