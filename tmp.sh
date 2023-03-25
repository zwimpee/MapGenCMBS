#!/bin/bash

# Create project directories
mkdir data
mkdir models
mkdir src
mkdir src/data_collection
mkdir src/deployment
mkdir src/models
mkdir src/training
mkdir src/utils

# Create necessary files
touch config.py
touch init.sh
touch LICENSE
touch README.md
touch requirements.txt

# Populate README.md
echo "# Project Name" >> README.md
echo "" >> README.md
echo "Authors: Zach Wimpee and Zack Milam" >> README.md
echo "" >> README.md
echo "## Overview" >> README.md
echo "" >> README.md
echo "This project aims to build a model that takes in satellite image and topographic map data, and generates a corresponding output file which recreates the input location as a map in Combat Mission: Black Sea. The specific goal is to recreate Bahkmut in-game." >> README.md
echo "" >> README.md
echo "## Project Structure" >> README.md
echo "" >> README.md
echo "```
data/
models/
src/
    data_collection/
    deployment/
    models/
    training/
    utils/
config.py
init.sh
LICENSE
README.md
requirements.txt
```" >> README.md

# Populate config.py
echo "from dataclasses import dataclass

@dataclass
class Config:
    # TODO: Fill in configuration variables
    pass" >> config.py

# Populate init.sh
echo "#!/bin/bash" >> init.sh
echo "" >> init.sh
echo "# TODO: Initialize environment" >> init.sh

# Populate LICENSE
echo "MIT License" >> LICENSE
echo "" >> LICENSE
echo "Copyright (c) 2023 Zach Wimpee and Zack Milam" >> LICENSE

# Populate requirements.txt
echo "# TODO: Add necessary packages and their versions" >> requirements.txt
