========
BH-ptychography
========

Simple demonstration of ptychography reconstruction using the Bilinear Hessian (BH) method. Implementation on GPU using cuPy library

  
================
Jupyter Notebooks
================

demo_object_probe.ipynb - ptychography reconstruction of the object and probe with the BH-CG (Conjugate gradient) method 

demo_object_probe_positions.ipynb - ptychography reconstruction of the object, probe, and positions with the BH-CG method 

demo_object_probeQN.ipynb - ptychography reconstruction of the object and probe with the BH-QN (Quasi-Newton) method and convergence comparison with the BH-CG

============
Google Colab
============

The code can be quickly tested in Google Colab, without installing any packages. See video guidance https://anl.box.com/s/gk9k8fjb7zsmy1qkshvnetdytiotpja0

===================================
Installation on a machine with GPU
===================================

conda create -n BH-ptychography -c conda-forge cupy notebook pandas

conda activate BH-ptychography

  
