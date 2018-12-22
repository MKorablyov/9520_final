import os,sys,time
from PIL import Image


images = ["ackley1.png","ackley2.png","ackley3.png","schwefel1.png","schwefel2.png","schwefel3.png"]
db_root = "/home/maksym/Desktop/9520_final/plots"


images = [Image.open(os.path.join(db_root,img)) for img in images]

h = images[0].size[0]
l = images[0].size[1]

new_im = Image.new('RGB', (3*h,2*l))
print new_im.size

x_offset = 0
for im in images[:3]:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
x_offset = 0
for im in images[3:6]:
    new_im.paste(im, (x_offset,l))
    x_offset += im.size[0]

new_im.save(os.path.join(db_root,'test.png'))