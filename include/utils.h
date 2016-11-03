#ifndef _UTILS_H_
#define _UTILS_H_

#define _CRT_RAND_S //for rand_s() windows crypto library fuction (must appear before #include statements)

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "CL/cl.h"
#include "timer.h"
#include "global_constants.h"
#include "config_alt.h"
#include "config_reg.h"
#include "config_mcs.h"
#include "config_tm.h"
#include "config_ga.h"
#include "Random123/u01.h"

typedef struct avg_val
{
    cl_float avg;
    cl_float std_dev;
} avg_val;

enum output_data_types
{
    TYPE_UINT,
    TYPE_FLOAT,
    TYPE_STRING,
    TYPE_AVG
};

typedef struct output_col
{
    char *header;
    cl_uint data_type;
    char *pattern;
} output_col;

typedef struct profiling_data
{
    cl_float *gpu_time;
    cl_float *cpu_time;
    CPerfCounter gpu_timer;
    CPerfCounter cpu_timer;
    
    cl_uint num_fitness_samples;
    cl_float *fitness_samples;
    cl_float *final_fitness;

    cl_uint extra_data_elements;
    cl_float *extra_data; //for whatever the current alg wants
} profiling_data;

void init_profiling_data(
    profiling_data *data,
    void *generic_conf,
    cl_uint extra_data_elements
    );

void release_profiling_data(
    profiling_data *data
    );

void print_timestamp();

void print_header_row(
    output_col *cols,
    cl_uint num_cols,
    profiling_data *prof_data,
    void *generic_conf
    );

void print_result_row(
    output_col *cols,
    cl_uint num_cols,
    void *data_items,
    profiling_data *prof_data,
    void *generic_conf
    );

void get_fitness_value(
    cl_float *samples_buf,
    cl_uint num_samples,
    cl_uint fitness_offset,
    cl_uint num_reps,
    cl_float *avg_fitness,
    cl_float *std_dev
    );

avg_val get_avg_muts(
    void *generic_conf,
    profiling_data *data,
    cl_uint extra_data_offset
    );

void print_profiling_data_alt(
    config_alt *conf, 
    profiling_data *data, 
    cl_uint print_headers,
    char *swarm_config_str
    );

void print_profiling_data_ga(
    config_ga *conf, 
    profiling_data *data, 
    cl_uint print_headers
    );

void print_profiling_data_mcs(
    config_mcs *conf, 
    profiling_data *data, 
    cl_uint print_headers
    );

void print_profiling_data_reg(
    config_reg *conf, 
    profiling_data *data, 
    cl_uint num_reps
    );

void print_profiling_data_tm(
    config_tm *conf, 
    profiling_data *data,
    cl_uint print_headers
    );

void check_error(
    cl_int status_cd,
    char *patt, ...
    );

char *get_kernel_name(
    cl_kernel *kernel
    );

char *get_cl_err_cd_desc(
    cl_int cd
    );

void fill_rand_uint_buf(
    cl_uint *buf,
    cl_uint n,
    cl_uint min_val,
    cl_uint max_val
    );

void fill_rand_buf(
    cl_float *buf,
    cl_uint n,
    cl_float max_val
    );

cl_uint get_seed();

cl_uint make_multiple(
    cl_uint n,
    cl_uint m
    );

size_t get_file_size(
    char *filename
    );

cl_uint read_src(
    char *filename,
    size_t file_size,
    cl_char *src
    );

cl_float *fill_rot_matrix_buf(
    cl_float *matrix,
    cl_uint rot_matrix_dim_len
    );

cl_float *gen_orthogonal_matrix(
    cl_float *matrix,
    cl_uint rot_matrix_dim_len
    );

cl_float *gen_identity_matrix(
    cl_float *matrix,
    cl_uint rot_matrix_dim_len
    );

void fill_perm_buf(
    cl_uint *buf,
    cl_uint n
    );

#endif
