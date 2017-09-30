#!/bin/bash

R -d "valgrind --tool=callgrind" -f perf.R
