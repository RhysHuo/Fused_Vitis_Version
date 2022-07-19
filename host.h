#ifndef HOST_H_
#define HOST_H_

#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <fstream>
#include "ap_int.h"

#define DTYPE_LENGTH 32
typedef ap_int<DTYPE_LENGTH> DTYPE;
typedef ap_int<32> DTYPE_OUT;

typedef unsigned int u32;

//***********************************************************************************************************************
#define DTYPE_LENGTH 32 //8//32 //8 //32//32 //8//32//8
#define MAX_N    2048 //2048 //64
#define MAX_M    512 //2048 //384 //1536 //384 //1536 // 384 //48 //768 //96 //384 //96 //384 //96 //384// 96
#define MAX_P    32768 //2048 //512 //64//1//64//1

#define A_HEIGHT   MAX_N
#define A_WIDTH    MAX_M

#define B_HEIGHT   MAX_M
#define B_WIDTH    MAX_P

#define C_HEIGHT   MAX_N
#define C_WIDTH    MAX_P

#define A_HEIGHT_BLOCK  1// 4096 //(512/4)
#define B_WIDTH_BLOCK 128 //16 //32 //64 //64 //128 // 64 //64 //64 //8//8// //16//32//1//32//1//32//1// 1//32//(128/4)
#define B_BLOCK_PARALLEL 1
#define C_HEIGHT_BLOCK  A_HEIGHT_BLOCK 
#define C_WIDTH_BLOCK   B_WIDTH_BLOCK

#define ENABLE_GEMM
#define ENABLE_SPMM
//#define ENABLE_SCALING
//#define ENABLE_TRANSPOSE


//typedef ap_int<DTYPE_LENGTH> DTYPE;
//typedef ap_int<32> DTYPE_OUT;

const static int II = 1;

const static int ROW_SIZE_MAX        = (1024);
const static int ROW_SIZE_THREAD_MAX = (1024);
const static int COL_SIZE_MAX        = (1024);


#endif
