import json

with open('CreditCard.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Print cells 60-78 fully (model training and evaluation section)
for i in range(55, len(cells)):
    c = cells[i]
    src = ''.join(c['source'])
    print(f"\n{'='*60}")
    print(f"Cell {i} ({c['cell_type']})")
    print(f"{'='*60}")
    print(src[:800])
    if c['cell_type'] == 'code' and c.get('outputs'):
        for out in c['outputs']:
            if out.get('text'):
                text = ''.join(out['text'])[:500]
                print(f"\n  >>> OUTPUT:\n{text}")
            elif out.get('data') and out['data'].get('text/plain'):
                text = ''.join(out['data']['text/plain'])[:500]
                print(f"\n  >>> OUTPUT:\n{text}")
