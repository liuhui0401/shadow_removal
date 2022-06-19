from PIL import Image
import glob

i = 0
for dir in glob.glob('./dataset/test_ori/*'):
    i += 1
    img = Image.open(dir)
    m,n = img.size
    region = img.crop((m/2-320,n/2-240,m/2+320,n/2+240))
    region.save('./dataset/test/test_A/'+dir.split('/')[-1][:-3]+'png')
print(i)