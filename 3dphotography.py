import os
import sys
from Inpainting import main

print("impainting started")

try:
    for i in os.listdir(sys.argv[1]):
        print(f"processing {i}")
        main.inpaint(i)
except Exception as e:
    print(f'{e}')
    print(f"processing {sys.argv[1]}")
    main.inpaint(sys.argv[1])
