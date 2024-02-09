#!/bin/bash

mkdir -p results_hpc

rsync -auv sesia@discovery.usc.edu:/home1/sesia/Workspace/tutorial_cluster/code/results/ results_hpc/
