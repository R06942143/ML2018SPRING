#!/bin/bash 
SUCCESS=0
python hw2_best.py $1 $2 $3 $4 $5 $6
if [ "$?" -ne $SUCCESS ]
then
        echo "================TA!!!!wait for a moment I have written exception============="
        python hw2_svm.py $1 $2 $3 $4 $5 $6
        exit 1
fi