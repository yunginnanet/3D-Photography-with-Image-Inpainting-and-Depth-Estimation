#!/usr/bin/env bash

set -e

mkdir -p Inpainting/checkpoints

wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
mv color-mode.pth Inpainting/checkpoints/
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
mv depth-model.pth Inpainting/checkpoints/
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth
mv edge-model.pth Inpainting/checkpoints/
wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/model.pt
mv model.pt Inpainting/MiDaS/

echo "congrats it might even work now maybe :^)"
