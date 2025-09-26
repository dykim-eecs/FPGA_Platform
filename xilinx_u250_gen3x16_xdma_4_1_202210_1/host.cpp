#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>
#include <xcl2.hpp>
#include <ap_int.h>
#include <CL/cl_ext_xilinx.h> // For cl_mem_ext_ptr_t and XCL_MEM_TOPOLOGY
#include <thread>
#include <atomic>
#include <utility>
#include <algorithm> // For std::transform, etc.
#include <fstream>

std::string to_sha512_hex(ap_uint<512> hash) {
    std::ostringstream oss;
    for (int i = 0; i < 64; ++i) {
        unsigned char byte = hash.range((511 - i * 8), (511 - i * 8 - 7));
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
    }
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <krnl_sha512.xclbin> <n_msg> <iterations>" << std::endl;
        return 1;
    }
    int n_msg = std::stoi(argv[2]);
    int iterations = std::stoi(argv[3]);
    if (n_msg <= 0 || iterations <= 0) {
        std::cerr << "n_msg and iterations must be > 0" << std::endl;
        return 1;
    }
    std::string binaryFile = argv[1];
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    cl::Context context(device, nullptr, nullptr, nullptr, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    cl::Program program(context, {device}, bins);
    int num_cus = 3;
    std::vector<std::string> cu_names = {"krnl_sha512_1_1", "krnl_sha512_2_1", "krnl_sha512_3_1"};
    std::vector<unsigned int> bank_ids = {0, 2, 3};
    std::vector<cl::Kernel> kernels(num_cus);
    for (int i = 0; i < num_cus; ++i) {
        std::string kname = "krnl_sha512:{" + cu_names[i] + "}";
        kernels[i] = cl::Kernel(program, kname.c_str(), &err);
    }
    size_t total_tasks = n_msg / 4;
    std::vector<size_t> tasks_per_cu(num_cus, total_tasks / num_cus);
    size_t rem = total_tasks % num_cus;
    for (size_t i = 0; i < rem; ++i) {
        tasks_per_cu[i]++;
    }
    std::vector<size_t> input_sizes(num_cus);
    for (int i = 0; i < num_cus; ++i) {
        input_sizes[i] = sizeof(ap_uint<1024>) * tasks_per_cu[i];
    }
    size_t output_size = sizeof(ap_uint<1024>) * 2;

    // Load pre-generated input data from file
    std::string input_file = "input_" + std::to_string(n_msg) + ".dat";
    std::ifstream infile(input_file, std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "Failed to open input file: " << input_file << std::endl;
        return 1;
    }
    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> raw_data(file_size);
    infile.read(raw_data.data(), file_size);
    infile.close();
    if (file_size != total_tasks * sizeof(ap_uint<1024>)) {
        std::cerr << "Input file size mismatch!" << std::endl;
        return 1;
    }

    double total_alloc_time_us = 0.0;
    double total_h2d_us = 0.0;
    double total_kt_us = 0.0;
    double total_hash_rate = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        // Measure allocation time
        auto alloc_start = std::chrono::high_resolution_clock::now();
        std::vector<cl::Buffer> inputBufs(num_cus);
        std::vector<cl::Buffer> outputBufs(num_cus);
        for (int i = 0; i < num_cus; ++i) {
            cl_mem_ext_ptr_t in_ext;
            in_ext.flags = bank_ids[i] | XCL_MEM_TOPOLOGY;
            in_ext.obj = nullptr;
            in_ext.param = 0;
            inputBufs[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_EXT_PTR_XILINX, input_sizes[i], &in_ext, &err);
            cl_mem_ext_ptr_t out_ext;
            out_ext.flags = bank_ids[i] | XCL_MEM_TOPOLOGY;
            out_ext.obj = nullptr;
            out_ext.param = 0;
            outputBufs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_EXT_PTR_XILINX, output_size, &out_ext, &err);
        }
        auto alloc_end = std::chrono::high_resolution_clock::now();
        double alloc_time_us = std::chrono::duration<double, std::micro>(alloc_end - alloc_start).count();
        total_alloc_time_us += alloc_time_us;

        // Distribute data to per-CU
        size_t offset = 0;
        std::vector<ap_uint<1024>*> inputDatas(num_cus);
        std::vector<ap_uint<1024>*> outputDatas(num_cus);
        for (int i = 0; i < num_cus; ++i) {
            inputDatas[i] = (ap_uint<1024>*)q.enqueueMapBuffer(inputBufs[i], CL_TRUE, CL_MAP_WRITE, 0, input_sizes[i]);
            std::memcpy(inputDatas[i], raw_data.data() + offset, input_sizes[i]);
            offset += input_sizes[i];
            outputDatas[i] = (ap_uint<1024>*)q.enqueueMapBuffer(outputBufs[i], CL_TRUE, CL_MAP_READ, 0, output_size);
        }

        std::vector<cl::Event> migrate_in_events(num_cus);
        std::vector<cl::Event> kernel_events(num_cus);
        std::vector<cl::Event> migrate_out_events(num_cus);

        for (int i = 0; i < num_cus; ++i) {
            kernels[i].setArg(0, inputBufs[i]);
            kernels[i].setArg(1, outputBufs[i]);
            kernels[i].setArg(2, (unsigned int)tasks_per_cu[i]);
            q.enqueueMigrateMemObjects({inputBufs[i]}, 0, nullptr, &migrate_in_events[i]);
        }

        // Enqueue kernels after H2D
        for (int i = 0; i < num_cus; ++i) {
            std::vector<cl::Event> dep_in(1, migrate_in_events[i]);
            q.enqueueTask(kernels[i], &dep_in, &kernel_events[i]);
        }

        // Wait for kernels to finish
        for (int i = 0; i < num_cus; ++i) {
            kernel_events[i].wait();
        }

        // Enqueue D2H after kernels
        for (int i = 0; i < num_cus; ++i) {
            std::vector<cl::Event> dep_k(1, kernel_events[i]);
            q.enqueueMigrateMemObjects({outputBufs[i]}, CL_MIGRATE_MEM_OBJECT_HOST, &dep_k, &migrate_out_events[i]);
        }

        q.finish();

        // Calculate H2D time
        double sum_h2d_us = 0.0;
        for (int i = 0; i < num_cus; ++i) {
            cl_ulong start_time, end_time;
            migrate_in_events[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
            migrate_in_events[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
            sum_h2d_us += (end_time - start_time) / 1000.0;
        }
        double avg_h2d_us = sum_h2d_us / num_cus;
        total_h2d_us += avg_h2d_us;

        // Calculate kernel time and hash rate
        double sum_kt_us = 0.0;
        double iter_hash_rate = 0.0;
        for (int i = 0; i < num_cus; ++i) {
            cl_ulong start_time, end_time;
            kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
            kernel_events[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
            double kt_us = (end_time - start_time) / 1000.0;
            sum_kt_us += kt_us;
            double cu_hashes = tasks_per_cu[i] * 4.0;
            double cu_rate = cu_hashes / (kt_us / 1e6);
            iter_hash_rate += cu_rate;
        }
        double avg_kt_us = sum_kt_us / num_cus;
        iter_hash_rate /= 1e6; // to MH/s
        total_kt_us += avg_kt_us;
        total_hash_rate += iter_hash_rate;

        for (int i = 0; i < num_cus; ++i) {
            q.enqueueUnmapMemObject(inputBufs[i], inputDatas[i]);
            q.enqueueUnmapMemObject(outputBufs[i], outputDatas[i]);
        }
        q.finish();

        if (iter == iterations - 1) {
            ap_uint<1024>* outputData = outputDatas[0]; // Using CU 0 for final digests
            ap_uint<512> digest1 = outputData[0].range(1023, 512);
            ap_uint<512> digest2 = outputData[0].range(511, 0);
            ap_uint<512> digest3 = outputData[1].range(1023, 512);
            ap_uint<512> digest4 = outputData[1].range(511, 0);
            std::cout << "Final 4 digests from last iteration:" << std::endl;
            std::cout << " [1] " << to_sha512_hex(digest1) << std::endl;
            std::cout << " [2] " << to_sha512_hex(digest2) << std::endl;
            std::cout << " [3] " << to_sha512_hex(digest3) << std::endl;
            std::cout << " [4] " << to_sha512_hex(digest4) << std::endl;
        }
    }

    double avg_alloc_time_us = total_alloc_time_us / iterations;
    double avg_h2d_us = total_h2d_us / iterations;
    double avg_kt_us = total_kt_us / iterations;
    double avg_hash_rate = total_hash_rate / iterations;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total messages per iteration: " << n_msg << std::endl;
    std::cout << "Average Kernel time (us): " << avg_kt_us << " | Average Hash rate (MH/s): " << avg_hash_rate << std::endl;
    std::cout << "Average Buffer allocation time (us): " << avg_alloc_time_us << std::endl;
    std::cout << "Average H2D memcpy time (us): " << avg_h2d_us << std::endl;
    return 0;
}