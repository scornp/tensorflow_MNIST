#!/bin/bash
#https://movidius.github.io/ncsdk/tf_compile_guidance.html 


if [ ! -d Result_03_movidiusGraph ];then
   mkdir Result_03_movidiusGraph
fi
#mkdir Result_03_movidiusGraph

cd Result_03_movidiusGraph 

cp ../Result_02_IOAdded/* .

mvNCCompile IOAdded.meta -s 12 -in input -on output -o IOAdded.graph


