#!/bin/bash 
OUTDIR="./dataset"
mkdir -p "$OUTDIR"
wget -O "$OUTDIR/3dshapes.h5" https://storage.googleapis.com/3d-shapes/3dshapes.h5