#ifndef _PRETTY_PRINTER_H_
#define _PRETTY_PRINTER_H_

#include "CL/cl.h"
#include "global_constants.h"
#include "devices.h"
#include "buffer_utils.h"
#include ALG_HEADER_STR(buffers)
#include "buffers_mcs.h"

void print_float_matrix(
    cl_float *buf, 
    cl_uint width, 
    cl_uint height, 
    cl_uint factor
    );
void print_uint_buf(
    cl_uint *buf,
    cl_uint n, 
    cl_uint show_indices
    );
void print_swarm_health_buf(
    char *label,
    cl_mem *mem_buf,
    void *generic_conf,
    device *gpu
    );
void print_crossover_perm_buf(
   char *label,
   config_mcs *conf,
   mpso_bufs_mcs *bufs, 
   device *gpu
   );
void print_float4(
    cl_float4 data
    );
void print_uint4(
    cl_uint4 data
    );
void print_n_floats(
    char *label, 
    cl_uint n,
    cl_uint offset,
    cl_mem *mem_buf, 
    device *gpu
    );

void print_n_uints(
    char *label, 
    cl_uint n,
    cl_mem *mem_buf,
    device *gpu
    );

void print_unvec_positions(
    cl_float *buf,
    void *generic_conf
    );
void print_unvec_velocities(
    cl_float *buf, 
    void *generic_conf
    );
void print_fitnesses(
    cl_float *buf, 
    void *generic_conf
    );
void print_unvec_sbest_positions(
    cl_float *buf,
    void *generic_conf
    );
void print_sbest_fitnesses(
    cl_float *buf, 
    void *generic_conf
    );
void print_indices_buf(
    cl_uint *buf, 
    void *generic_conf
    );

void print_rot_matrix(
    char *label, 
    cl_mem *mem_buf,
    cl_uint dim_len,
    device *gpu
    );
void print_perm_vec(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    );

void print_positions(
    char *label, 
    cl_mem *mem_buf,
    void *generic_conf, 
    device *gpu
    );
void print_velocities(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    );
void print_fitnesses(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    );
void print_sbest_positions(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    );
void print_sbest_fitnesses(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    );
void print_indices_buf(
    char *label, 
    cl_mem *mem_buf, 
    void *generic_conf, 
    device *gpu
    );

#endif
