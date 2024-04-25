# PHYS416_S24
_final project for PHYS416 - Computational Physics_

* This project focuses on the application of the Forward in Time, Centered in Space (FTCS) method towards solving a collection of diffusion equations used to model the behavior of neutrons in fission reactors.

* A spherical system with radial symmetry is considered, and a cylindrical system with symmetry in the radial and longtudinal directions is also analyzed.

* Phase plots of neutron density as a function of time and location on a grid mesh are generated to understand the behavior under different boundary conditions.

* The average neutron density as a function of time is plotted to consider the approximate criticality of the system, under unique boundary conditions. 

* This project utilizes python, numpy, matplotlib, and ffmpeg. 

* Animations exist as .mp4 files for the spherical and cylindrical systems. I used the ffmpeg library with the FuncAnimation matplotlib class, in order to produce an .mp4 file instead of a .gif file. In order to run both programs in this repository, ffmpeg may need to be installed at the following path: '/usr/local/bin/ffmpeg'

References for getting a functioning animation in python using ffmpeg:
https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
https://www.ffmpeg.org/download.html
https://docs.dkrz.de/doc/visualization/sw/python/source_code/python-matplotlib-animation-with-ffmpeg-normal-and-faster.html
https://holypython.com/how-to-save-matplotlib-animations-the-ultimate-guide/
https://spatialthoughts.com/2022/01/14/animated-plots-with-matplotlib/
