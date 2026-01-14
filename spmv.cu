#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <stdio.h>

const int PADDING_VALUE = -1;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t err = (call); \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while (0)

struct Vector {
    size_t length;
    double* vals;
};

struct Csr {
    int64_t nrows;
    int64_t ncols;
    int64_t nnz;
    int* col_ind;
    int* row_ptr;
    double* nz;
};

struct Ellpack8 {
    size_t nrows;
    size_t ncols;
    size_t max_row_nnz;
    int* col_ind;
    double* nz;
};

struct Ellpack7 {
    size_t nrows;
    size_t ncols;
    size_t nrows_padded;
    int* col_ind[7];
    double* nz[7];
};

struct SlicedEllpack {
    size_t nrows;
    size_t ncols;
    size_t slice_size;
    int* slice_ptr;
    int* col_ind;
    double* nz;
};

void banded_matrix_fill(size_t nrows, size_t ncols, std::vector<size_t>& row, std::vector<size_t>& col, std::vector<double>& nz) {
    row.reserve(8 * nrows);
    col.reserve(8 * nrows);
    nz.reserve(8 * nrows);

    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = i; j < i + 7 && j < ncols; ++j) {
            row.push_back(i);
            col.push_back(j);
            nz.push_back(static_cast<double>(rand()) / RAND_MAX);
        }
    }
}

void csr_host_fill(const std::vector<size_t>& row_in, const std::vector<size_t>& col_in, const std::vector<double>& nz_in, int64_t nrows, int64_t ncols, Csr& host_csr) {
    host_csr.nrows = nrows;
    host_csr.ncols = ncols;
    host_csr.nnz = nz_in.size();
    
    host_csr.col_ind = new int[host_csr.nnz];
    host_csr.nz = new double[host_csr.nnz];
    host_csr.row_ptr = new int[host_csr.nrows + 1];

    for (size_t i = 0; i < host_csr.nnz; ++i) {
        host_csr.nz[i] = nz_in[i];
        host_csr.col_ind[i] = static_cast<int>(col_in[i]);
    }

    std::fill(host_csr.row_ptr, host_csr.row_ptr + host_csr.nrows + 1, 0);
    for (size_t i = 0; i < host_csr.nnz; ++i) {
        host_csr.row_ptr[row_in[i] + 1]++;
    }
    for (int64_t i = 0; i < host_csr.nrows; ++i) {
        host_csr.row_ptr[i + 1] += host_csr.row_ptr[i];
    }
}

void ellpack8_host_fill(const std::vector<size_t>& row_in, const std::vector<size_t>& col_in, const std::vector<double>& nz_in, size_t nrows, size_t ncols, Ellpack8& host_ell) {
    host_ell.nrows = nrows;
    host_ell.ncols = ncols;
    host_ell.max_row_nnz = 8;

    size_t total_ell_elements = host_ell.nrows * host_ell.max_row_nnz;
    host_ell.nz = new double[total_ell_elements];
    host_ell.col_ind = new int[total_ell_elements];
    
    std::fill(host_ell.col_ind, host_ell.col_ind + total_ell_elements, PADDING_VALUE);
    std::fill(host_ell.nz, host_ell.nz + total_ell_elements, 0.0);

    std::vector<int> row_counts(nrows, 0);
    for (size_t i = 0; i < nz_in.size(); ++i) {
        size_t r = row_in[i];
        size_t c_idx = row_counts[r]++;
        size_t ell_idx = r * host_ell.max_row_nnz + c_idx;
        host_ell.nz[ell_idx] = nz_in[i];
        host_ell.col_ind[ell_idx] = col_in[i];
    }
}

void ellpack7_host_fill(const std::vector<size_t>& row_in, const std::vector<size_t>& col_in, const std::vector<double>& nz_in, size_t nrows, size_t ncols, Ellpack7& host_ell) {
    host_ell.nrows = nrows;
    host_ell.ncols = ncols;
    host_ell.nrows_padded = (nrows + 7) / 8 * 8;

    for (int i = 0; i < 7; ++i) {
        host_ell.nz[i] = new double[host_ell.nrows_padded];
        host_ell.col_ind[i] = new int[host_ell.nrows_padded];
        std::fill(host_ell.nz[i], host_ell.nz[i] + host_ell.nrows_padded, 0.0);
        std::fill(host_ell.col_ind[i], host_ell.col_ind[i] + host_ell.nrows_padded, PADDING_VALUE);
    }
    
    std::vector<int> row_counts(nrows, 0);
    for (size_t i = 0; i < nz_in.size(); ++i) {
        size_t r = row_in[i];
        size_t c_idx = row_counts[r]++;
        if (c_idx < 7) {
            host_ell.nz[c_idx][r] = nz_in[i];
            host_ell.col_ind[c_idx][r] = col_in[i];
        }
    }
}

void sliced_ellpack_host_fill(const std::vector<size_t>& row_in, const std::vector<size_t>& col_in, const std::vector<double>& nz_in, size_t nrows, size_t ncols, SlicedEllpack& host_sell, const size_t slice_size, std::vector<size_t>& sorted_to_original_map) {
    host_sell.nrows = nrows;
    host_sell.ncols = ncols;
    host_sell.slice_size = slice_size;
    
    std::vector<std::vector<std::pair<int, double>>> rows_data(nrows);
    std::vector<std::pair<size_t, size_t>> row_metadata(nrows);
    
    for (size_t i = 0; i < nz_in.size(); ++i) {
        size_t r = row_in[i];
        rows_data[r].push_back({static_cast<int>(col_in[i]), nz_in[i]});
    }

    for (size_t i = 0; i < nrows; ++i) {
        row_metadata[i] = {i, rows_data[i].size()};
    }

    std::sort(row_metadata.begin(), row_metadata.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    
    for (size_t i = 0; i < nrows; ++i) {
        sorted_to_original_map[i] = row_metadata[i].first;
    }

    size_t num_slices = (nrows + slice_size - 1) / slice_size;
    host_sell.slice_ptr = new int[num_slices + 1];
    host_sell.slice_ptr[0] = 0;
    size_t current_padded_nnz = 0;

    for (size_t i = 0; i < num_slices; ++i) {
        size_t current_slice_size = std::min(slice_size, nrows - i * slice_size);
        size_t max_nnz_in_slice = 0;

        for (size_t j = 0; j < current_slice_size; ++j) {
            size_t original_row_idx = row_metadata[i * slice_size + j].first;
            max_nnz_in_slice = std::max(max_nnz_in_slice, rows_data[original_row_idx].size());
        }

        current_padded_nnz += current_slice_size * max_nnz_in_slice;
        host_sell.slice_ptr[i + 1] = current_padded_nnz;
    }
    
    host_sell.col_ind = new int[current_padded_nnz];
    host_sell.nz = new double[current_padded_nnz];

    for (size_t i = 0; i < num_slices; ++i) {
        size_t current_slice_size = std::min(slice_size, nrows - i * slice_size);
        size_t max_nnz_in_slice = (host_sell.slice_ptr[i+1] - host_sell.slice_ptr[i]) / current_slice_size;
        size_t slice_base_idx = host_sell.slice_ptr[i];

        for (size_t j = 0; j < current_slice_size; ++j) {
            size_t row_idx_in_slice = j;
            size_t original_row_idx = row_metadata[i * slice_size + j].first;
            const auto& row_data = rows_data[original_row_idx];

            for (size_t k = 0; k < max_nnz_in_slice; ++k) {
                size_t padded_idx = slice_base_idx + k * current_slice_size + row_idx_in_slice;
                if (k < row_data.size()) {
                    host_sell.col_ind[padded_idx] = row_data[k].first;
                    host_sell.nz[padded_idx] = row_data[k].second;
                } else {
                    host_sell.col_ind[padded_idx] = PADDING_VALUE;
                    host_sell.nz[padded_idx] = 0.0;
                }
            }
        }
    }
}

__global__ void csr_spmv_kernel(const Csr* A, const double* x, double* y) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A->nrows) {
        double sum = 0.0;
        for (size_t i = A->row_ptr[row]; i < A->row_ptr[row + 1]; ++i) {
            sum += A->nz[i] * x[A->col_ind[i]];
        }
        y[row] = sum;
    }
}

__global__ void ellpack8_spmv_kernel(const Ellpack8* A, const double* x, double* y) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A->nrows) {
        double sum = 0.0;
        for (size_t i = 0; i < A->max_row_nnz; ++i) {
            size_t idx = row * A->max_row_nnz + i;
            sum += A->nz[idx] * x[A->col_ind[idx]];
        }
        y[row] = sum;
    }
}

__global__ void ellpack7_spmv_kernel(const Ellpack7* A, const double* x, double* y) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < A->nrows) {
        double sum = 0.0;
        for (size_t i = 0; i < 7; ++i) {
            sum += A->nz[i][row] * x[A->col_ind[i][row]];
        }
        y[row] = sum;
    }
}

__global__ void sliced_ellpack_spmv_kernel(
    const SlicedEllpack* A,
    const double* x,
    double* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A->nrows) {
        double sum = 0;
        int slice_index = row / A->slice_size;
        int row_in_slice = row % A->slice_size;
        int max_elements_in_this_slice = A->slice_ptr[slice_index + 1] - A->slice_ptr[slice_index];
        int rows_in_this_slice = (slice_index * A->slice_size + A->slice_size) > A->nrows ? 
                                 (A->nrows - slice_index * A->slice_size) : A->slice_size;
        int max_columns_in_this_slice = max_elements_in_this_slice / rows_in_this_slice;
        int slice_base_index = A->slice_ptr[slice_index] + row_in_slice;

        for (int i = 0; i < max_columns_in_this_slice; ++i) {
            int index = slice_base_index + i * rows_in_this_slice;
            int col = A->col_ind[index];
            sum += A->nz[index] * x[col];
        }
        y[row] = sum;
    }
}

void run_test(size_t N) {
    const size_t block_size = 256;
    const size_t grid_size = (N + block_size - 1) / block_size;
    const size_t slice_size = 256;

    std::vector<size_t> h_row_vec, h_col_vec;
    std::vector<double> h_nz_vec;
    banded_matrix_fill(N, N, h_row_vec, h_col_vec, h_nz_vec);
    int64_t nnz = h_nz_vec.size();
    
    Csr h_csr;
    csr_host_fill(h_row_vec, h_col_vec, h_nz_vec, N, N, h_csr);
    Ellpack8 h_ell8;
    ellpack8_host_fill(h_row_vec, h_col_vec, h_nz_vec, N, N, h_ell8);
    Ellpack7 h_ell7;
    ellpack7_host_fill(h_row_vec, h_col_vec, h_nz_vec, N, N, h_ell7);
    
    std::vector<size_t> sorted_to_original_map(N);
    SlicedEllpack h_sell;
    sliced_ellpack_host_fill(h_row_vec, h_col_vec, h_nz_vec, N, N, h_sell, slice_size, sorted_to_original_map);

    Vector h_x;
    h_x.length = N;
    h_x.vals = new double[N];
    for (size_t i = 0; i < N; ++i) h_x.vals[i] = static_cast<double>(rand()) / RAND_MAX;

    Csr *d_csr;
    CHECK_CUDA(cudaMalloc(&d_csr, sizeof(Csr)));
    int *d_csr_col_ind, *d_csr_row_ptr;
    double *d_csr_nz;
    CHECK_CUDA(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_row_ptr, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_nz, nnz * sizeof(double)));
    
    Ellpack8 *d_ell8;
    CHECK_CUDA(cudaMalloc(&d_ell8, sizeof(Ellpack8)));
    size_t ell8_nnz_padded = h_ell8.nrows * h_ell8.max_row_nnz;
    int *d_ell8_col_ind;
    double *d_ell8_nz;
    CHECK_CUDA(cudaMalloc(&d_ell8_col_ind, ell8_nnz_padded * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ell8_nz, ell8_nnz_padded * sizeof(double)));

    Ellpack7 *d_ell7;
    CHECK_CUDA(cudaMalloc(&d_ell7, sizeof(Ellpack7)));
    int *d_ell7_col_ind[7];
    double *d_ell7_nz[7];
    for (int i = 0; i < 7; ++i) {
        CHECK_CUDA(cudaMalloc(&d_ell7_col_ind[i], h_ell7.nrows_padded * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_ell7_nz[i], h_ell7.nrows_padded * sizeof(double)));
    }

    SlicedEllpack *d_sell;
    CHECK_CUDA(cudaMalloc(&d_sell, sizeof(SlicedEllpack)));
    size_t num_slices = (N + slice_size - 1) / slice_size;
    int total_padded_nnz = h_sell.slice_ptr[num_slices];
    int *d_sell_slice_ptr;
    int *d_sell_col_ind;
    double *d_sell_nz;
    CHECK_CUDA(cudaMalloc(&d_sell_slice_ptr, (num_slices + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sell_col_ind, total_padded_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sell_nz, total_padded_nnz * sizeof(double)));

    double *d_x, *d_y_csr, *d_y_ell8, *d_y_ell7, *d_y_cusparse, *d_y_sell;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_csr, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_ell8, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_ell7, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_cusparse, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_sell, N * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_csr_col_ind, h_csr.col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_row_ptr, h_csr.row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_nz, h_csr.nz, nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    Csr h_d_csr;
    h_d_csr.nrows = h_csr.nrows;
    h_d_csr.ncols = h_csr.ncols;
    h_d_csr.nnz = h_csr.nnz;
    h_d_csr.col_ind = d_csr_col_ind;
    h_d_csr.row_ptr = d_csr_row_ptr;
    h_d_csr.nz = d_csr_nz;
    CHECK_CUDA(cudaMemcpy(d_csr, &h_d_csr, sizeof(Csr), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_ell8_col_ind, h_ell8.col_ind, ell8_nnz_padded * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ell8_nz, h_ell8.nz, ell8_nnz_padded * sizeof(double), cudaMemcpyHostToDevice));
    Ellpack8 h_d_ell8;
    h_d_ell8.nrows = h_ell8.nrows;
    h_d_ell8.ncols = h_ell8.ncols;
    h_d_ell8.max_row_nnz = h_ell8.max_row_nnz;
    h_d_ell8.col_ind = d_ell8_col_ind;
    h_d_ell8.nz = d_ell8_nz;
    CHECK_CUDA(cudaMemcpy(d_ell8, &h_d_ell8, sizeof(Ellpack8), cudaMemcpyHostToDevice));

    Ellpack7 h_d_ell7;
    h_d_ell7.nrows = h_ell7.nrows;
    h_d_ell7.ncols = h_ell7.ncols;
    h_d_ell7.nrows_padded = h_ell7.nrows_padded;
    for (int i = 0; i < 7; ++i) {
        CHECK_CUDA(cudaMemcpy(d_ell7_col_ind[i], h_ell7.col_ind[i], h_ell7.nrows_padded * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_ell7_nz[i], h_ell7.nz[i], h_ell7.nrows_padded * sizeof(double), cudaMemcpyHostToDevice));
        h_d_ell7.col_ind[i] = d_ell7_col_ind[i];
        h_d_ell7.nz[i] = d_ell7_nz[i];
    }
    CHECK_CUDA(cudaMemcpy(d_ell7, &h_d_ell7, sizeof(Ellpack7), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sell_slice_ptr, h_sell.slice_ptr, (num_slices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sell_col_ind, h_sell.col_ind, total_padded_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sell_nz, h_sell.nz, total_padded_nnz * sizeof(double), cudaMemcpyHostToDevice));

    SlicedEllpack h_d_sell;
    h_d_sell.nrows = h_sell.nrows;
    h_d_sell.ncols = h_sell.ncols;
    h_d_sell.slice_size = h_sell.slice_size;
    h_d_sell.slice_ptr = d_sell_slice_ptr;
    h_d_sell.col_ind = d_sell_col_ind;
    h_d_sell.nz = d_sell_nz;
    CHECK_CUDA(cudaMemcpy(d_sell, &h_d_sell, sizeof(SlicedEllpack), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.vals, N * sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float time_csr, time_ell7, time_ell8, time_cusparse, time_sell;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));
    csr_spmv_kernel<<<grid_size, block_size>>>(d_csr, d_x, d_y_csr);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_csr, start, stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    ellpack8_spmv_kernel<<<grid_size, block_size>>>(d_ell8, d_x, d_y_ell8);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ell8, start, stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    ellpack7_spmv_kernel<<<grid_size, block_size>>>(d_ell7, d_x, d_y_ell7);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ell7, start, stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    sliced_ellpack_spmv_kernel<<<grid_size, block_size>>>(d_sell, d_x, d_y_sell);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_sell, start, stop));

    cusparseHandle_t cusparse_handle = NULL;
    cusparseSpMatDescr_t matA_desc;
    cusparseDnVecDescr_t vecX_desc, vecY_desc;
    double alpha = 1.0, beta = 0.0;
    
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&matA_desc, h_csr.nrows, h_csr.ncols, h_csr.nnz,
                                     d_csr_row_ptr, d_csr_col_ind, d_csr_nz,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX_desc, N, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY_desc, N, d_y_cusparse, CUDA_R_64F));

    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_desc, vecX_desc, &beta, vecY_desc,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    
    void* d_buffer = NULL;
    CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));

    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUSPARSE(cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA_desc, vecX_desc, &beta, vecY_desc,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_cusparse, start, stop));

    printf("%lu,%.6f,%.6f,%.6f,%.6f,%.6f\n", N, time_cusparse, time_csr, time_ell8, time_ell7, time_sell);

    delete[] h_csr.col_ind; delete[] h_csr.row_ptr; delete[] h_csr.nz;
    for (int i = 0; i < 7; ++i) {
        delete[] h_ell7.col_ind[i];
        delete[] h_ell7.nz[i];
    }
    delete[] h_ell8.col_ind; delete[] h_ell8.nz;
    delete[] h_sell.slice_ptr; delete[] h_sell.col_ind; delete[] h_sell.nz;
    delete[] h_x.vals;
    
    CHECK_CUDA(cudaFree(d_csr));
    CHECK_CUDA(cudaFree(d_csr_col_ind));
    CHECK_CUDA(cudaFree(d_csr_row_ptr));
    CHECK_CUDA(cudaFree(d_csr_nz));
    
    CHECK_CUDA(cudaFree(d_ell7));
    for (int i = 0; i < 7; ++i) {
        CHECK_CUDA(cudaFree(d_ell7_col_ind[i]));
        CHECK_CUDA(cudaFree(d_ell7_nz[i]));
    }
    CHECK_CUDA(cudaFree(d_ell8));
    CHECK_CUDA(cudaFree(d_ell8_col_ind));
    CHECK_CUDA(cudaFree(d_ell8_nz));
    CHECK_CUDA(cudaFree(d_sell));
    CHECK_CUDA(cudaFree(d_sell_slice_ptr));
    CHECK_CUDA(cudaFree(d_sell_col_ind));
    CHECK_CUDA(cudaFree(d_sell_nz));

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y_csr));
    CHECK_CUDA(cudaFree(d_y_ell7));
    CHECK_CUDA(cudaFree(d_y_ell8));
    CHECK_CUDA(cudaFree(d_y_cusparse));
    CHECK_CUDA(cudaFree(d_y_sell));
    CHECK_CUDA(cudaFree(d_buffer));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUSPARSE(cusparseDestroySpMat(matA_desc));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX_desc));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY_desc));
    CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
}

int main(int argc, char** argv) {
    std::cout << "N,cusparse,csr,ellpack8,ellpack7,sell" << std::endl;
    std::vector<size_t> sizes = {100, 1000, 10000, 100000, 1000000, 10000000};
    for (size_t size : sizes) {
        run_test(size);
    }
    return 0;
}
