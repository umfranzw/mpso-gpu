#ifndef _BUFFER_UTILS_H_
#define _BUFFER_UTILS_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include <string.h>
#include ALG_HEADER_STR(config)

void fill_optimum_buf(
    cl_mem *buf,
    void *generic_conf,
    device *gpu
    );

void *map_buffer(
    cl_mem *buf,
    cl_map_flags map_flags,
    size_t size,
    device *gpu,
    cl_event *complete
    );

void unmap_buffer(
    cl_mem *buf,
    void *mapped_ptr,
    device *gpu,
    cl_event *complete
    );

void fill_rot_matrix_buf(
    cl_mem *buf,
    cl_uint rot_matrix_dim_len,
    device *gpu
    );

void fill_perm_buf(
    cl_mem *buf,
    void *generic_conf,
    device *gpu
    );

/* float find_best_fitness( */
/*     void *generic_conf, */
/*     cl_mem *sbest_fitnesses, */
/*     device *gpu */
/*     ); */

void gather_final_fitness_data(
    cl_mem *fitness_sample_buf,
    cl_mem *final_fitness_buf,
    cl_mem *extra_data_buf,
    profiling_data *prof_data,
    void *generic_conf,
    device *gpu
    );

#endif
