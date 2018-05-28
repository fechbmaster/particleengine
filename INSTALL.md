# Getting started

## Binary release

A precompiled binary of Hydrodynamix can be found under:

https://github.com/Hydrodynamix/ParticleEngine/releases

Please make sure your graphics driver is up-to-date.
Because we're using CUDA-C, the software will only run on Nvidia GPUs.

Your GPU needs to support compute capability 3.0 and OpenGL 4.3 (GTX 6xx series or newer).

## Building from source

### Step 1: Install dependencies

* [Visual Studio 2013 Update 4](https://www.visualstudio.com/)
* [Qt 5.4 with OpenGL](https://www.qt.io/download-open-source/#section-5) (VS addin recommended)
* [CUDA Toolkit 7.0](https://developer.nvidia.com/cuda-toolkit)
* [Assimp 3.1.1](http://assimp.sourceforge.net/)
* [Boost 1.57.0](http://www.boost.org/) (an install script for boost can be found in `ThirdParty/Install/BuildBoost.bat`)

##### Optional: Using the boost install script

* Download the newest boost version from [boost.org](http://www.boost.org/)
* Extract the archive to the place where you want to install boost
* Copy ``BuildBoost.bat`` from the repository to your boost folder
* Run ``BuildBoost.bat`` as *administrator*.

### Step 2: Environment variables

Make sure the following environment variables are set:

* `QTDIR` (e.g. `C:\Program Files\Qt\5.4\msvc2013_64_opengl`)
* `BOOST_ROOT` (e.g. `C:\Program Files\boost_1_57_0`) - Note: If you use our install script it should automatically set this
* `CUDA_PATH` (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0`)
* `CUDA_PATH_V7_0` (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0`)
* `ASSIMP_ROOT` (e.g. `C:\Program Files\assimp-3.1.1`)

### Step 3: Building the solution

Open `Hydrodynamix.sln` and select your Qt version under:

> Qt5 -> Qt Options -> Qt Versions

It should say something like `msvc2013_64_opengl` and point to your `QTDIR` path.

Change the build configuration to Release/x64 if you want to build under 64-bit and click on build.
