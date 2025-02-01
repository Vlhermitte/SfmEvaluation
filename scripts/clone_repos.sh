#!/bin/bash

# Make sure we are in the root directory of the project
cd "$(dirname "$0")/.." || exit

# Clone the SFM approaches (Glomap, VGGSfm, Flowmap, AceZero)

# Clone Glomap
git clone https://github.com/Vlhermitte/glomap.git

# Clone VGGSfm
git clone https://github.com/Vlhermitte/vggsfm.git

# Clone Flowmap
git clone https://github.com/Vlhermitte/flowmap.git

# Clone AceZero
git clone https://github.com/Vlhermitte/acezero.git

# Clone Tank and Temples (Dataset)
git clone https://github.com/Vlhermitte/TanksAndTemples.git