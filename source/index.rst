Conditional Diffusion for Grayscale Image Colorization
======================================================

A U-Net based diffusion model that colorizes grayscale face images.
The grayscale image is used as a conditioning signal by concatenating it
as an additional channel to the noisy RGB image during the forward and reverse processes.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   train
   unet
   noise

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
