# Hydrodynamix Particle Engine

This is a clean and lightweight implementation of Lagrangian Fluid Dynamics (SPH) which utilizes the power of modern GPUs to do computation and rendering.
The software is written in C++ and uses Nvidias CUDA-C for the GPGPU part.

#### Warning: This is incomplete! Will be updated over the next few weeks!

### Features

* Realtime simulation and rendering
* Change physical properties in realtime through a modern CSS based UI
* Import and place meshes (supports all common formats; .obj, .fbx, .3ds, .x, ...)
* Interact with the fluid and create waves
* Create waterjets and water droplets by pressing the R/F-Keys
* Highly efficient surface rendering using a screen space curvature flow method

##### [Click](https://www.youtube.com/watch?v=vFrt6ObnsEM) to watch on YouTube:

[![Preview](https://github.com/Hydrodynamix/ParticleEngine/blob/master/Preview.png)](https://www.youtube.com/watch?v=vFrt6ObnsEM)

### Getting started

For instructions on how to get started please check out the [INSTALL](https://github.com/Hydrodynamix/ParticleEngine/blob/master/INSTALL.md) file.

### Libraries used

* Qt 5.4
* OpenGL 4.3 (using Qt)
* CUDA Toolkit 7.0
* Assimp 3.1.1
* Boost 1.57.0

### Compilation

We were using Visual Studio 2013 Update 4 to build our software.
There's no gurantee that you will be able to build our application on other compilers/platforms without modifications.

You will need to install all the dependencies on your own.
Please make sure that your environment variables are set up correctly before trying to build the application.

- We use `$(QTDIR), $(CUDA_LIB_PATH), $(BOOST_ROOT)` and `$(ASSIMP_ROOT)` as environment variables for our dependencies.
- The batch script that we used to build the boost libraries can be found in `ThirdParty/Install/BuildBoost.bat`
- We do not make use of Windows specific code so the application should also compile on other platforms.

For more detailed instructions, please refer to the [INSTALL](https://github.com/Hydrodynamix/ParticleEngine/blob/master/INSTALL.md) file.

### Implementation details

* Screen Space Fluid Rendering with Curvature Flow (van der Laan et. al.)
* TODO: others

### License

Hydrodynamix is licensed under the MIT license. Dependencies are under their respective licenses.

### Contributing

We accept contributions through pull requests on github. When contributing, please make sure that your code conforms to our coding style.

- Our coding style guidelines document will be uploaded at a later time
