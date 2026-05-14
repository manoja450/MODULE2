# MODULE2
g++ -o Micheltest BestMichelCodeMODULE2withareadirectuse1.cpp `root-config --cflags` `root-config --libs | sed 's/-lROOTNTuple[^ ]*//g'` -std=c++17
sbatch Micheltest.slurm
