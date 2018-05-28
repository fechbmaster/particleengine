:: Make sure to run this script as administrator!
@setlocal enableextensions
@cd /d "%~dp0"
@echo Setting environment variable BOOST_ROOT to %CD%
@setx BOOST_ROOT %CD% -m
@call bootstrap.bat
@echo Building 32 bit libs
@b2 --build-dir=build\x86 --toolset=msvc-12.0 address-model=32 --build-type=complete --stagedir=stage\x86 stage
@echo Building 64 bit libs
@b2 --build-dir=build\x64 --toolset=msvc-12.0 address-model=64 --build-type=complete --stagedir=stage\x64 stage
@echo Finished! Press any key to continue!
@pause>nul