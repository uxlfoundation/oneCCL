#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOXYFILE=$SCRIPT_DIR/Doxyfile
DOXYGEN_OUTPUT_DIR=$SCRIPT_DIR/doxygen
DOXYGEN_XML_OUTPUT=$DOXYGEN_OUTPUT_DIR/xml
RST_DIR=$SCRIPT_DIR/rst
RST_SOURCE_DIR=$RST_DIR/source
RST_BUILD_DIR=$RST_DIR/build
RST_HTML_BUILD_DIR=$RST_BUILD_DIR/html

if ! [ -x "$(command -v doxygen)" ]
then
  echo 'Error: Doxygen not found.'
  exit 1
fi

if ! [ -x "$(command -v sphinx-build)" ]
then
  echo 'Error: Sphinx not found.'
  exit 1
fi

if [ -d "$DOXYGEN_XML_OUTPUT" ]
then
  echo "Removing $DOXYGEN_XML_OUTPUT"
  rm -r $DOXYGEN_XML_OUTPUT
fi

if [ -d "$RST_HTML_BUILD_DIR" ]
then
  echo "Removing $RST_HTML_BUILD_DIR"
  rm -r $RST_HTML_BUILD_DIR
fi

echo "Generating API using Doxygen..."
pushd $SCRIPT_DIR
  doxygen $DOXYFILE
popd

echo "Generating HTML using Sphinx..."
sphinx-build -b html $RST_SOURCE_DIR $RST_HTML_BUILD_DIR

echo "Removing files from build/html"

rm -r $RST_HTML_BUILD_DIR/.doctrees
rm -r $RST_HTML_BUILD_DIR/.buildinfo