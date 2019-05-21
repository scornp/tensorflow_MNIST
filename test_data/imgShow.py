import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from PIL import Image

fig = plt.figure()
a=fig.add_subplot(1,2,1)

x = range( 0, 10 )
y = range( 0, 10 )

width = 0.35

plt.bar(x, y, width)
plt.xlabel('digit')
plt.ylabel('prediction')

#img = mpimg.imread('3a1.png')
a=fig.add_subplot(1,2,2)

img = Image.open('3a1.png')
#print(img)
img.thumbnail((28, 28), Image.ANTIALIAS)  # resizes image in-place

imgplot = plt.imshow(img)

plt.show()

print('end')
