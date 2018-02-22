:: Installation scripts for appveyor.

@echo on

:: Miniconda path for appveyor
set PATH=C:\Miniconda-x64;C:\Miniconda-x64\Scripts;%PATH%
:: Install numpy
conda install -y numpy
