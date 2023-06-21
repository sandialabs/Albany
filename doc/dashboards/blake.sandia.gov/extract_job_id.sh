#!/bin/bash
awk '{print $4}' log-gcc.out >& jobid.out 
