import base64, io
from PIL import Image, ImageDraw
img = Image.new('RGB', (640, 480), (240, 236, 244))
d = ImageDraw.Draw(img)
d.rectangle([10, 10, 630, 230], outline=(40,40,40), width=3)
d.rectangle([10, 240, 630, 470], outline=(40,40,40), width=3)
for (x1,y1,x2,y2) in [(60,60,300,160), (360,60,600,170), (60,280,320,400)]:
    d.ellipse([x1,y1,x2,y2], fill=(255,255,255), outline=(30,30,30), width=2)
    for r in range(4):
        d.line([x1+18, y1+22+r*14, x2-18, y1+22+r*14], fill=(50,50,50), width=5)
buf = io.BytesIO()
img.save(buf, 'JPEG', quality=92)
print(base64.b64encode(buf.getvalue()).decode())
