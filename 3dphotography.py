# from DepthEstimation.inference import main
# main()
# print('Depth Estimation Done!')

import os
import sys
from Inpainting import main
print('impainting started')
for i in os.listdir(sys.argv[1]):
  print(f'processing {i}')
  main.inpaint(i)
