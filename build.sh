#!/bin/bash

mkdir -p build
cmake -S src -B build
cd build
make