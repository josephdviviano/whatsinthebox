#!/bin/bash

while read line; do
    wget -c ${line};
done < aws-sample-wet.paths
