#!/bin/sh -e

cd $( dirname -- "$0"; )
docker build -t ccldocgen .
docker run -v $(pwd)/rst:/app/rst -v $(pwd)/../include:/include ccldocgen