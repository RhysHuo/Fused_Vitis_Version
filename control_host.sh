git clone https://github.com/RhysHuo/Fused_Vitis_Version.git
cd Fused_Vitis_Version
cp host.cpp ..
cp host.h ..
cd ..
rm -rf Fused_Vitis_Version
g++ -g -std=c++14 -I$XILINX_XRT/include -L${XILINX_XRT}/lib/ -I/mnt/scratch/rhyhuo/HLS_arbitrary_Precision_Types/include -o host.exe host.cpp host.h -lOpenCL -pthread -lrt -lstdc++

