import requests
r = requests.get('http://127.0.0.1:5055/translate-result/a0000000-0000-0000-0000-000000000002', timeout=20)
print('translate-result:', r.status_code, '| viewport:', 'name="viewport"' in r.text, '| edit-btn:', 'edit-btn' in r.text)
r2 = requests.get('http://127.0.0.1:5055/correction/a0000000-0000-0000-0000-000000000001', timeout=20)
print('correction:', r2.status_code, '| thumb role:', 'role="button"' in r2.text, '| thumb tabindex:', 'tabindex="0"' in r2.text)
