# MODULE2
/home/manoja450/MODULE2H2Oanalysis
g++ -o Micheltest BestMichelCodeMODULE2withareadirectuse1.cpp `root-config --cflags` `root-config --libs | sed 's/-lROOTNTuple[^ ]*//g'` -std=c++17
sbatch Micheltest.slurm

**COMPILE IN COMPUTE NODE:******

srun --partition=longjobs --time=1:30:00 --mem=10G --cpus-per-task=1 --pty bash

# First set up ROOT environment
export LD_LIBRARY_PATH=/usr/lib64/root:$LD_LIBRARY_PATH
export ROOTSYS=/usr
export PATH=/usr/bin:$PATH
ROOT_CFLAGS=$(root-config --cflags)
ROOT_LIBS=$(root-config --libs)

# Compile - ALL ON ONE LINE!
g++ -O2 -o Micheltest BestMichelCodeMODULE2withareadirectuse1.cpp ${ROOT_CFLAGS} ${ROOT_LIBS} -Wl,-rpath,/usr/lib64/root -std=c++17


**ON ROOT:**
TCanvas *c1 = new TCanvas("c1","Michel Electron Energy",800,600);

 michel_energy->Draw();
 
c1->SaveAs("michel_energy.png");
