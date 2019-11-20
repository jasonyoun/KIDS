#!/bin/bash

set -e

module load java/8u131

tar xzf pra.tar.gz
rm -f pra.tar.gz

# config file from arguments
cp $1 pra/conf

# move the directory to a unique name
mv pra pra-$1

cd pra-$1
java -Xms1G -Xmx1G -cp pra.jar edu.cmu.lti.util.run.TunnerSweep conf





