> Under Construction

# Transfer learning on PINNs for tracking hemodynamics

The source code and data for the article [M. Daneker, S. Cai, Y. Qian, E. Myzelev, A. Khumbhat & L. Lu. Transfer Learning on Physics-Informed Neural Networks for
Tracking the Hemodynamics in the Evolving False Lumen of Dissected Aorta. *Nexus*, 1(2), 2024](https://doi.org/10.1016/j.ynexs.2024.100016).

## Data

The data in this repository consits of computational fluid dynamics results for 3 aneuerysms, as well as the simulated MRI slice data for each aneurysm. 

Due to the size of the data it was uploaded to [OneDrive](https://yaleedu-my.sharepoint.com/:f:/g/personal/lu_lu_yale_edu/EiXTCD4kdgxIu1MII7nqsiEBNKcd-Yt6wSbEjGynnTOgQQ).

## Code

The [Navier-Stokes flow nets base code](code/NSFnets3D.py) is required for all other .py files to run, make sure it is included in the same directory. 

- [Data only](code/data_only.py)
- [Warm-start PINN](code/WS_PINN.py)
- [Transfer learning warm-start PINN](code/TL_WS_PINN.py)

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{Daneker2024,
  author    = {Daneker, Mitchell and Cai, Shengze and Qian, Ying and Myzelev, Eric and Khumbat, Arsh and Li, He and Lu, Lu},
  title     = {Transfer Learning on Physics-Informed Neural Networks for Tracking the Hemodynamics in the Evolving False Lumen of Dissected Aorta},
  journal   = {Nexus}
  volume    = {1},
  issue     = {2},
  year      = {2024},
  doi       = {https://doi.org/10.1016/j.ynexs.2024.100016},
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
