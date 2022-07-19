#include <iostream>
#include <sstream> // std::stringstream
#include <algorithm>

#include "xcl2.hpp"
#include <CL/cl.h>
#include <CL/cl2.hpp>
#include "host.h"

int SN, SM, SP;

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

const int BLOCK=B_WIDTH_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK = 128
const int PARALLEL_ROW = B_BLOCK_PARALLEL;
const int A_WIDTH_FIFO =  A_WIDTH;

void dsp_kernel_sw(
	DTYPE a_value,
	DTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],
	ap_int<32> b_row,
	ap_int<8> zero_point_lhs,
	ap_int<8> zero_point_rhs,
	DTYPE_OUT acc[B_WIDTH_BLOCK]
)

{
	//#pragma HLS ALLOCATION instances=mul limit=64 operation
	//#pragma HLS INLINE

	//DTYPE_OUT acc[B_WIDTH_BLOCK];
	//#pragma HLS ARRAY_PARTITION variable=acc complete

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

		#pragma HLS UNROLL
		acc[j] = 0;
    }

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
        //#pragma HLS UNROLL
	    #pragma HLS PIPELINE
		for(int z = 0; z < DTYPE_LENGTH; z+=8) {
 	  		ap_int<8> A_val = a_value.range(z+7,z);
	  		ap_int<8> B_val = b_block[b_row][j].range(z+7,z);
			//acc[j] += (A_val-zero_point_lhs)*(B_val-zero_point_rhs);
			acc[j] += A_val*(B_val-zero_point_rhs);
		}
	} // j loop
}

void compute_sw(ap_uint<2> mode, 
		ap_int<8> zero_point_lhs,  
		ap_int<8> zero_point_rhs, 
		int N, 
		int M, 
		int P, 
		DTYPE* A, 
		DTYPE* B,
		//hls::stream<DTYPE_OUT> C_fifo[B_WIDTH_BLOCK],
		DTYPE C_fifo[B_WIDTH_BLOCK],
		int B_index, 
		int B_index_loop, 
		int tail,
		int *rowPtr,
		int *columnIndex,
		DTYPE *values
)
//./host.exe kernelmult1.sw_emu.xclbin 1 0 gemm_weights_byte_099.csv 128 128 128
{


       //#pragma HLS allocation function instances=dsp_kernel limit=1
	DTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK];
    	#pragma HLS array_partition variable=B_accel block factor= BLOCK/2 dim=2

        //hls::stream<DTYPE>       A_accel;
        //#pragma HLS STREAM variable=A_accel depth=A_WIDTH_FIFO dim=1

	DTYPE A_accel[A_WIDTH];
        //#pragma HLS array_partition variable=A_accel cyclic factor=


	DTYPE_OUT acc[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc complete


	DTYPE_OUT acc2[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc2 complete


	//hls::stream<int>             col_indices_fifo;
	//#pragma HLS STREAM variable=col_indices_fifo depth=1024 dim=1

	int col_indices[A_WIDTH];

    	#pragma HLS DATAFLOW	

	int B_WIDTH_INT,rnnz;

	if (B_index < (B_index_loop-1))
		B_WIDTH_INT = B_WIDTH_BLOCK;
	else
		B_WIDTH_INT = tail;


	for (int j = 0; j < B_WIDTH_INT; j++) {
		LOOP_BLOCK1 : 
			for (int i = 0; i < M; i++) {
				#pragma HLS loop_tripcount min=84 max=84 avg=84
				#pragma HLS PIPELINE
				#pragma HLS loop_tripcount min=16 max=16 avg=16
				B_accel[i][j] = B[i+j*M+B_index*B_WIDTH_BLOCK*M];
				//printf("B_accel[%d][%d] = %d \n", i, j, B_accel[i][j]);
				//std::cout << i << " " << j << std::endl;
			}
	}
    
	for (int A_index = 0; A_index < N; A_index++) {
		#pragma HLS loop_tripcount min=6 max=6 avg=6

		//std::cout << "A_index is " << A_index << " out of " << N/A_HEIGHT_BLOCK << std::endl;

		//load A row

		if (mode == 0) //gemm load A row
		{

			#ifdef ENABLE_GEMM
			//printf("gemm : loading A row \n");
			LOOP_A_ROW_GEMM : 
				for (int j = 0; j < M; j++) {
					#pragma HLS PIPELINE
					//A_accel <<  A[A_index*M*A_HEIGHT_BLOCK+j];
					A_accel[j] =  A[A_index*M*A_HEIGHT_BLOCK+j];
					//if((A_index*M*A_HEIGHT_BLOCK+j) < 100)
						//printf("A[%d] = %d \n", A_index*M*A_HEIGHT_BLOCK+j, A[A_index*M*A_HEIGHT_BLOCK+j]);
					//std::cout << "A is " << A_accel[i][j] << std::endl;
				}
			#endif

		}
		else //spmm load A row
		{

			#ifdef ENABLE_SPMM
			printf("spmm : loading A row \n");
			int current_index= rowPtr[A_index];
			int next_index=rowPtr[A_index+1];
			rnnz = next_index-current_index;
			//LOOP_A_ROW_SPMM : for (int j = current_index; j < next_index; j++) {
			LOOP_A_ROW_SPMM : 
				for (int j = 0; j < rnnz; j++) {
					#pragma HLS PIPELINE
					//A_accel <<  values[j];
					//col_indices_fifo << columnIndex[j];
					A_accel[j] =  values[current_index+j];
					col_indices[j] = columnIndex[current_index+j];
					//A_accel[z] =  current_index+j;
					//col_indices[z] = current_index+j;

				}
			#endif

		}
		
		//computing
	
		for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			#pragma HLS UNROLL
				acc2[j] = 0;
		}

		if (mode == 0) //gemm
		{
			#ifdef ENABLE_GEMM
			//printf("gemm : computing \n");

	   		DSP_LOOP_GEMM: 
				for(int k = 0; k < M; k+=1) {
					#pragma HLS loop_tripcount min=84 max=84 avg=84
	        			#pragma HLS PIPELINE
	        			#pragma HLS UNROLL factor=PARALLEL_ROW

					//how many rows of B block are computed in parallel in multiplication loop
					//for example a couple of B block rows are multiplied for A 1 row in each loop iteration
					//it basically reduces how the loop iterations by 2 if it is 2.

					//DTYPE v = A_accel.read();
					DTYPE v = A_accel[k];
					
					dsp_kernel_sw(v,B_accel,k,zero_point_lhs,zero_point_rhs,acc);

					for (int j = 0; j < B_WIDTH_BLOCK; j++) {
						#pragma HLS UNROLL
						acc2[j] += acc[j];
					}
				} // k loop
     			for (int j = 0; j < B_WIDTH_BLOCK; j++) {
				//#pragma HLS loop_tripcount min=16 max=16 avg=16
				#pragma HLS UNROLL
				if (j < B_WIDTH_INT)
				{
					C_fifo[j] = acc2[j];
					//if(A_index == 0)
					//printf("acc2[%d] = %d \n", j, acc2[j]);
				}
			}

			#endif
		} //mode spmm
		else
		{
			#ifdef ENABLE_SPMM
			printf("spmm : computing \n");
			
			DSP_LOOP_SPMM: 
				for (int i = 0; i < rnnz; i+=1) {
					#pragma HLS PIPELINE
					#pragma HLS UNROLL factor=PARALLEL_ROW
					//DTYPE v = A_accel.read();
					//int   ci = col_indices_fifo.read();
					DTYPE v = A_accel[i];
					int   ci = col_indices[i];

					dsp_kernel_sw(v,B_accel,ci,zero_point_lhs,zero_point_rhs,acc);

					for (int j = 0; j < B_WIDTH_BLOCK; j++) {
						#pragma HLS UNROLL			
						acc2[j] += acc[j];
					}
				} //i loop

				for (int j = 0; j < B_WIDTH_BLOCK; j++) {
					#pragma HLS UNROLL
					if (j < B_WIDTH_INT)
					{
						C_fifo[j] = acc2[j];
					}
				}

			#endif
		} //else
    } // A_index loop
}

void writec_sw(
	       int N,
	       int P, 
	       //hls::stream<DTYPE_OUT> write_fifo[C_WIDTH_BLOCK], 
	       DTYPE write_fifo[C_WIDTH_BLOCK],
	       DTYPE* C,
	       int array_c_adjust,
	       int B_index, 
	       int B_index_loop,
	       int tail
)

{
	int B_WIDTH_INT;
	if (B_index < (B_index_loop-1))
		B_WIDTH_INT = B_WIDTH_BLOCK;
	else
		B_WIDTH_INT = tail;

	LOOP_WRITE1:    
		for (int i = 0; i < (N>>2); i++) {
			DTYPE C_out;
			LOOP_WRITE2: 
				for (int j = 0; j < B_WIDTH_INT; j++) {
					#pragma HLS PIPELINE
					#pragma HLS loop_tripcount min=1 max=1 avg=1
					//C_out =  write_fifo[j].read();
					C_out =  write_fifo[j];
					#ifdef ENABLE_TRANSPOSE
						//C[i+(j+B_index*B_WIDTH_BLOCK)*(array_c_adjust>>2)] = C_out;
						//printf("Wrote address %x\n", (int)(i+(j+B_index*B_WIDTH_BLOCK)*(array_c_adjust>>2))); 
					#else
						C[i*P+j+B_index*B_WIDTH_BLOCK] = C_out;
					#endif
				}	
		}				
}

//gemm
static void init_arrays_gemm(DTYPE *B, DTYPE *C_sw, DTYPE *C)
{
    for (int i = 0; i < SM; i++) {    
        for (int j = 0; j < SP; j++) {
        	B[i * SP + j] =  0x01;
        }
    }
    for (int i = 0; i < SN; i++) {
        for (int j = 0; j < SP; j++) {
			C_sw[i * SP + j] = 0;
			C[i * SP + j] = 0;
		}
	}
}

static void load_arrays_byte_gemm(DTYPE *A, std::ifstream& myFile)
{
	// Make sure the file is open
	if(!myFile.is_open()) throw std::runtime_error("Could not open byte file");

	// Helper vars
	std::string line;
	int val;
	int val_count=0;
	DTYPE array_val;

    for (int i = 0; i < SN; i++) {
    	// Read data, line by line
    	std::getline(myFile, line);

	    // Create a stringstream of the current line
	    std::stringstream ss(line);

        for (int j = 0; j < SM; j++) {

	        //fill one array val
        	array_val = 0;
	        for(int z =0; z< DTYPE_LENGTH/8; z++)
	        {
	        	// Extract each integer
	        	ss >> val;
	        	array_val = (array_val << 8) + val;

	            // If the next token is a comma, ignore it and move on
	            if(ss.peek() == ',') ss.ignore();
	        }
	        A[i * SM + j] = array_val;
	        val_count++;
	    }
    }
    std::cout << "(BYTE) Val count " << val_count << std::endl;
}

void mmult_golden_byte(DTYPE *A, DTYPE *B, DTYPE *C)
{
	for (int row = 0; row < SN; row++) {
		for (int col = 0; col < SP; col++) {
			DTYPE result = 0;
			for (int k = 0; k < SM; k++) {
				for(int z = 0; z < DTYPE_LENGTH; z+=8) {
					DTYPE A_temp1 = A[row*SM+k];
					ap_int<8> A_val = A_temp1.range(z+7,z);
					result+=A_val*B[k*SP+col];
				}
			}
			C[row*SP+col] = result;
		}
	}
}

//spmm
void golden_spmm_byte(DTYPE *values, int *row_ptr, int *col_indices, DTYPE *x, int no_vectors, DTYPE *y, int row_size, int col_size) {

	int nvc = 0, i = 0, j = 0, rowStart = 0, rowEnd = row_size;

	DTYPE y0 = 0;
	int last_j = 0;
	for (nvc = 0; nvc < no_vectors; nvc++) {
		for (i = rowStart; i < rowEnd; ++i) {
			y0 = 0;
			for (j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
				for(int z = 0; z < DTYPE_LENGTH; z+=8) {
					            DTYPE values_val1 = values[j];
								ap_int<8> values_val = values_val1.range(z+7,z);
								int x_value = nvc*col_size+col_indices[j];
								int x_up = x_value >> 2;
								int x_down = (x_value & 0x3);
						       	y0 += values_val * x[x_up].range(x_down*8+7,x_down*8);
				}
			}
			y[nvc*row_size+i] = y0;
		}
	}
}

void init_arrays_spmm(DTYPE *x, int row, int col)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < (col>>2); j++) {
            x[i*(col>>2)+j] = 0x01010101;
        }
    }
}

//both
static int result_check(DTYPE *y, DTYPE *y_golden, int row, int col)
{
	for (int i = 0; i < row * col; i++) {
		if (y_golden[i] != y[i]) {
			std::cout 	<< "Mismatch: data index= " << i << " golden = " << y_golden[i]
						<< ", kernel = " << y[i] << std::endl;
			return 1;
		}
	}
    std::cout 	<< "TEST PASSED !" <<  std::endl;
	return 0;
}

//main
int main(int argc, char** argv) {

    if (argc != 8) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << " cores" << " mode: 0 for gemm / 1 for spmm" << " file" << " N" << " M" << " P" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];
    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        program = cl::Program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl = cl::Kernel(program, "kernelmult1_sw", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    int S_cores = atoi(argv[2]);
    ap_uint<2> spmm = atoi(argv[3]);
	FILE *fp_input;
        fp_input = fopen(argv[4], "r");
	
    ap_int<32> bias_count = 0;
	ap_int<8> zero_point_lhs = 0;
	ap_int<8> zero_point_rhs = -128;
	ap_int<8> zero_point_dst = -128;
	ap_int<8> clamp_max = 127;
	ap_int<8> clamp_min = -128;
    int nnz = 512;
	int row_size = 0;
        int col_size = 0;

    if(spmm) {
	    
        if (fp_input != NULL) {
            char line_1[1000];
            if(fgets(line_1, sizeof(line_1), fp_input) != NULL){
                sscanf(line_1, "%d %d %d", &row_size, &col_size, &nnz);
            }
        }
        else {
            std::cout << "Error with input file name" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    int no_vectors = 512;

    if(spmm){
        SN = row_size;
        SM = col_size;
        SP = no_vectors;
    }
    else{
        SN = atoi(argv[5]);
        SM = atoi(argv[6]);
        SP = atoi(argv[7]);
    }

    // Map our user-allocated buffers as OpenCL buffers using a shared host pointer
    OCL_CHECK(err, cl::Buffer buffer_array_a(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , SN * SM * sizeof(DTYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_array_b(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , SM * SP * sizeof(DTYPE), NULL, &err));    
    OCL_CHECK(err, cl::Buffer buffer_array_values(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , SN * SM * sizeof(DTYPE), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_array_c(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR , SN * SP * sizeof(DTYPE), NULL, &err));
	
	OCL_CHECK(err, cl::Buffer buffer_quantized_multiplier(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , SN * sizeof(DTYPE_OUT), NULL, &err));
	OCL_CHECK(err, cl::Buffer buffer_shift(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , SN * sizeof(DTYPE_OUT), NULL, &err));
	OCL_CHECK(err, cl::Buffer buffer_bias(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , SN * sizeof(DTYPE_OUT), NULL, &err));
	OCL_CHECK(err, cl::Buffer buffer_array_colIndices(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , nnz * sizeof(int), NULL, &err));
	OCL_CHECK(err, cl::Buffer buffer_array_rowPtr(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR , nnz * sizeof(int), NULL, &err));

    // Set the kernal argument
    /*
    int narg = 0;
    OCL_CHECK(err, err = krnl.setArg(narg++, S_cores));
    OCL_CHECK(err, err = krnl.setArg(narg++, spmm));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_quantized_multiplier));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_shift));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_bias));
    OCL_CHECK(err, err = krnl.setArg(narg++, bias_count));
    OCL_CHECK(err, err = krnl.setArg(narg++, zero_point_lhs));
    OCL_CHECK(err, err = krnl.setArg(narg++, zero_point_rhs));
    OCL_CHECK(err, err = krnl.setArg(narg++, zero_point_dst));
    OCL_CHECK(err, err = krnl.setArg(narg++, clamp_max));
    OCL_CHECK(err, err = krnl.setArg(narg++, clamp_min));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_a));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_b));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_c));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_values));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_colIndices));
    OCL_CHECK(err, err = krnl.setArg(narg++, buffer_array_rowPtr));
    OCL_CHECK(err, err = krnl.setArg(narg++, nnz));
    OCL_CHECK(err, err = krnl.setArg(narg++, SN));
    OCL_CHECK(err, err = krnl.setArg(narg++, SM));
    OCL_CHECK(err, err = krnl.setArg(narg++, SP));
    */

    DTYPE *array_a;
    DTYPE *array_b;
    DTYPE *array_values;
    DTYPE *array_c;
    DTYPE *array_c_golden = new DTYPE[SN * SP];
	
    DTYPE_OUT *quantized_multiplier;
    DTYPE_OUT *shift;
    DTYPE_OUT *bias;
    int *array_colIndices;
    int *array_rowPtr;
    
    //Map buffers to userspace pointers
    OCL_CHECK(err, array_a = (DTYPE*)q.enqueueMapBuffer(buffer_array_a, CL_TRUE, CL_MAP_WRITE, 0, SN * SM * sizeof(DTYPE), nullptr, nullptr, &err));
    OCL_CHECK(err, array_b = (DTYPE*)q.enqueueMapBuffer(buffer_array_b, CL_TRUE, CL_MAP_WRITE, 0, SM * SP * sizeof(DTYPE), nullptr, nullptr, &err));
    OCL_CHECK(err, array_values = (DTYPE*)q.enqueueMapBuffer(buffer_array_values, CL_TRUE, CL_MAP_WRITE, 0, SN * SM * sizeof(DTYPE), nullptr, nullptr, &err));
	OCL_CHECK(err, array_c = (DTYPE*)q.enqueueMapBuffer(buffer_array_c, CL_TRUE, CL_MAP_READ, 0, SN * SP * sizeof(DTYPE), nullptr, nullptr, &err));
	
	OCL_CHECK(err, quantized_multiplier = (DTYPE_OUT*)q.enqueueMapBuffer(buffer_quantized_multiplier, CL_TRUE, CL_MAP_WRITE, 0, SN * sizeof(DTYPE_OUT), nullptr, nullptr, &err));
	OCL_CHECK(err, shift = (DTYPE_OUT*)q.enqueueMapBuffer(buffer_shift, CL_TRUE, CL_MAP_WRITE, 0, SN * sizeof(DTYPE_OUT), nullptr, nullptr, &err));
	OCL_CHECK(err, bias = (DTYPE_OUT*)q.enqueueMapBuffer(buffer_bias, CL_TRUE, CL_MAP_WRITE, 0, SN * sizeof(DTYPE_OUT), nullptr, nullptr, &err));
	OCL_CHECK(err, array_colIndices = (int*)q.enqueueMapBuffer(buffer_array_colIndices, CL_TRUE, CL_MAP_WRITE, 0, nnz * sizeof(int), nullptr, nullptr, &err));
	OCL_CHECK(err, array_rowPtr = (int*)q.enqueueMapBuffer(buffer_array_rowPtr, CL_TRUE, CL_MAP_WRITE, 0, nnz * sizeof(int), nullptr, nullptr, &err));
	
    //Initialization
    if(spmm)
        init_arrays_spmm(array_b, SM, SP);
    else
        init_arrays_gemm(array_b, array_c_golden, array_c);
	
	for(int i = 0; i < SN; i++)
	{
		quantized_multiplier[i] = 1;
		shift[i] = 0;
		bias[i] = 0;
	}
	
	std::cout << "Init_arrays completed." << std::endl;

	// load arrays
    if(spmm){
        int r;
        int c;
        DTYPE v;

        if (fp_input != NULL) {
            char line_2[1000];
            int line_number = 0;
                while (fgets(line_2, sizeof(line_2), fp_input) != NULL) {
                if (line_number < nnz) {
                    sscanf(line_2, "%d %d", &c, &v);
                    array_colIndices[line_number] = c;
                    std::cout << "array_colIndices = " << array_colIndices[line_number] << std::endl;
                    array_values[line_number] = v;
                    std::cout << "array_values = " << array_values[line_number] << std::endl;
                }
                else {
                    sscanf(line_2, "%d", &r);
                    array_rowPtr[line_number - nnz] = r;
                    std::cout << "array_rowPtr = " << array_rowPtr[line_number - nnz] << std::endl;
                }
                line_number++;
            }
        }
    }
    else {
		std::ifstream myFile(argv[4]);
		load_arrays_byte_gemm(array_a, myFile);
    }
        
	std::cout << "Load data completed." << std::endl;
	
	
	//kernel execution
	
	ap_int<32> tail = SP % B_WIDTH_BLOCK;
	ap_int<32> B_index_loop = SP / B_WIDTH_BLOCK + 1;
	DTYPE C_fifo[B_WIDTH_BLOCK];
	
	compute_sw(
		spmm, 
		zero_point_lhs,  
		zero_point_rhs, 
		SN, 
		SM, 
		SP, 
		array_a, 
		array_b,
		//hls::stream<DTYPE_OUT> C_fifo[B_WIDTH_BLOCK],
		C_fifo[B_WIDTH_BLOCK],
		0, //B_index
		B_index_loop, 
		tail,
		array_rowPtr,
		array_colIndices,
		array_values
	);
	
	writec_sw(
	       SN,
	       SP, 
	       //hls::stream<DTYPE_OUT> write_fifo[C_WIDTH_BLOCK], 
	       C_fifo[B_WIDTH_BLOCK],
	       array_c,
	       SN, //array_c_adjust
	       0, //B_index
	       B_index_loop,
	       tail
	);
	
	
	//double start_time, end_time, execution_time;
    
    // Date will be migrate to the kernal space
	/*
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_array_a, buffer_array_b, buffer_array_values, buffer_quantized_multiplier, buffer_shift, buffer_bias, buffer_array_colIndices, buffer_array_rowPtr}, 0));
	std::cout << "enqueueMigrateMemObjects_0 completed." << std::endl;

    // Lauch the kernal
    OCL_CHECK(err, err = q.enqueueTask(krnl));
	std::cout << "enqueueTask completed." << std::endl;
    
    // To view the results, this call will transfer the data from FPGA to the host

	// Rather than manually enqueueing a migration, we can instead just map the buffer. 
	// The OpenCL runtime will recognize that the buffer contents are currently resident in 
	// the Alveo Data Center accelerator card global memory and will take care of 
	// migrating the buffer back to the host for us. This is a coding style choice you must make.

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_array_c}, CL_MIGRATE_MEM_OBJECT_HOST));
	std::cout << "enqueueMigrateMemObjects_CL_MIGRATE_MEM_OBJECT_HOST completed." << std::endl;
    
    q.finish();
	std::cout << "q.finish() completed." << std::endl;
	*/

    std::cout << "Start to mmult_golden " << std::endl;
	if(spmm)
        golden_spmm_byte(
            array_values,
            array_rowPtr,
            array_colIndices,
            array_b,
            SP,
            array_c_golden,
            SN,
            SM
        );
    else
        mmult_golden_byte(array_a, array_b, array_c_golden);

    /*
	end_time = getTimestamp();
	execution_time = (end_time - start_time) / (1000);
	std::cout << "CPU " << " Total execution time = " << execution_time << " msec" << std::endl;
    */

    // Compare the results of the Device to the simulation
    std::cout << "Start to result_check " << std::endl;

    if(result_check(array_c, array_c_golden, SN, SP))
        return 1;

	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_a, array_a));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_b, array_b));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_values, array_values));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_c, array_c));
	
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_quantized_multiplier, quantized_multiplier));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_shift, shift));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_bias, bias));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_colIndices, array_colIndices));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_array_rowPtr, array_rowPtr));
	q.finish();

}
