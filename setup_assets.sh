#!/bin/bash

# Ensure working dir is at the location of this script
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null

# Activate the environment (only necessary for the "gdown" command)
eval "$(conda shell.bash hook)"
conda activate gtavatar

TMP=/tmp/gtavatar
mkdir $TMP

echo -e "\n############################## FLAME Model ##############################"
if [ -f "FLAME2020.zip" ]; then
    echo "Found FLAME2020.zip, unzipping..." &&
    unzip FLAME2020.zip -d $TMP &&
    mv $TMP/FLAME2020/generic_model.pkl ./assets/flame/flame2020.pkl
else
    echo "FLAME2020.zip not found. Download the FLAME model from https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=FLAME2020.zip"
fi

echo -e "\n############################## FLAME Texture Space ##############################"
if [ -f "TextureSpace.zip" ]; then
    echo "Found TextureSpace.zip, unzipping..." &&
    unzip TextureSpace.zip -d $TMP &&
    mv $TMP/FLAME_texture.npz ./assets/flame/
else
    echo "TextureSpace.zip not found. Download the FLAME texture space from https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=TextureSpace.zip"
fi

echo -e "\n############################## FLAME Masks ##############################"
echo "Downloading FLAME masks" &&
wget https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip -O $TMP/FLAME_masks.zip &&
unzip $TMP/FLAME_masks.zip -d $TMP &&
mv $TMP/FLAME_masks.pkl ./assets/flame/

echo -e "\n############################## Pretrained SMIRK model (tracking) ##############################"
gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O ./assets/SMIRK_em1.pt

echo -e "\n############################## MediaPipe face landmarker ##############################"
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task --directory-prefix assets

echo -e "\nDone: cleaning-up" &&
rm -rf $TMP
