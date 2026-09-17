#!/usr/bin/env bash
# One-time push to https://github.com/Ayudyog-10
# 1) Create an EMPTY private repo named cinchona-nir-quinine at https://github.com/new (owner: Ayudyog-10, no README).
# 2) Run this script from inside the unzipped folder.
set -e
REPO=${1:-cinchona-nir-quinine}
git init -b main
git add .
git commit -m "Cinchona NIR quinine screening: analysis, model v1.0.0, Streamlit app"
git remote add origin "https://github.com/Ayudyog-10/${REPO}.git"
git push -u origin main
