#!/bin/bash
OUTPUT="$(aubio tempo $1)"
BEAT="$(aubio beat $1 | head -n 1)"
echo $OUTPUT
echo $BEAT
./hack $1 $OUTPUT $BEAT
