#!/bin/bash
OUTPUT="$(aubio tempo $1)"
./hack "$1 $OUTPUT"
