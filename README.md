# MODULE2
/home/manoja450/MODULE2H2Oanalysis
g++ -o Micheltest BestMichelCodeMODULE2withareadirectuse1.cpp `root-config --cflags` `root-config --libs | sed 's/-lROOTNTuple[^ ]*//g'` -std=c++17
sbatch Micheltest.slurm

**COMPILE IN COMPUTE NODE:******

srun --partition=longjobs --time=10:30:00 --mem=10G --cpus-per-task=1 --pty bash

# First, set up ROOT environment
export LD_LIBRARY_PATH=/usr/lib64/root:$LD_LIBRARY_PATH
export ROOTSYS=/usr

# Get the correct flags from root-config
ROOT_CFLAGS=$(root-config --cflags)
ROOT_LIBS=$(root-config --libs)

# Now compile
g++ -O2 -o Micheltest BestMichelCodeMODULE2withareadirectuse1.cpp \
    ${ROOT_CFLAGS} \
    ${ROOT_LIBS} \
    -Wl,-rpath,/usr/lib64/root \
    -std=c++17
