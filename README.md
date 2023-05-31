# 3D-Photography: Image Inpainting | Depth-Estimation
<p>Sensing Depth from 2D Images and Inpainting Background behind the Foreground objects to create 3D Photos with Parallax Animation.</p>


### it may even work now
you can get real cheeky and try to get this to work in 2023, i had success with 3.10 venv after my modifications, check `getmodels.sh` as well
#### example
`python3 3dphotography.py /path/to/images`

##### you're welcome, from yung innanet

---

## Depth Estimation by Tiefenrausch	and PyDNet respectively.

<p align="center">
  <img src="assets/depth.png" width="750">
</p>

# Image Inpainting with Parallax Animation

<p align="center">
    <img src="assets/swing.gif" height="350">
    <img src="assets/dolly-zoom.gif" height="350">
</p>


---
## Reference

    @article{Kopf-OneShot-2020,
    author    = {Johannes Kopf and Kevin Matzen and Suhib Alsisan and Ocean Quigley and Francis Ge and Yangming Chong and Josh Patterson and Jan-Michael Frahm and Shu Wu and Matthew Yu and Peizhao Zhang and Zijian He and Peter Vajda and Ayush Saraf and Michael Cohen},
    title     = {One Shot 3D Photography},
    booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
    publisher = {ACM},
    volume = {39},
    number = {4},
    year = {2020}
    }
