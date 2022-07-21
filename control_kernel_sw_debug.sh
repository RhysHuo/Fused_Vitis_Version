git clone https://github.com/RhysHuo/Fused_Vitis_Version.git
cd Fused_Vitis_Version
cp kernelMatrixmult_sw.cpp ..
cp kernelMatrixmult.h ..
cd ..
rm -rf Fused_Vitis_Version
v++ -t sw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -c -k kernelmult1_sw -o'kernelmult1.sw_emu.xo' kernelMatrixmult_sw.cpp kernelMatrixmult.h xcl2.hpp
v++ -t sw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 --link kernelmult1.sw_emu.xo -o'kernelmult1.sw_emu.xclbin'
