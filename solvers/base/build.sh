#!/bin/bash
./configure --quiet --test  && make 2>&1
cp build/kissat ./