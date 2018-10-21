#!/bin/bash
OUTPUT="$(aubio beat $1)"
./hack $1 $OUTPUT
