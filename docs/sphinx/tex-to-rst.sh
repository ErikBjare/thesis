#!/bin/bash

set -e

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

src=$SCRIPTPATH/../tex/content
dest=$SCRIPTPATH/src/content

mkdir -p $dest
for f in $src/*.tex; do
    out=$dest/$(basename $f | sed 's/\.tex/\.rst/')
    echo "$f to $out"
    pandoc -o $out  $f
    sed -i -r -e 's/(:raw-latex:)`[\]cite[{]([^\}]+)[}]`/:cite:p:`\2`/g' $out

    #echo -e "\n.. bibliography::" >> $out
done

cp -r $src/../img $dest/
