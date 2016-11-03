#ifndef _UPDATE_FITNESS_SHARED_KERNEL_DRIVER_COMMON_H_
#define _UPDATE_FITNESS_SHARED_KERNEL_DRIVER_COMMON_H_

#include "CL/cl.h"
#include "global_constants.h"
#include ALG_HEADER_STR_COMMON(config)
#include ALG_HEADER_STR_COMMON(buffers)
#include "devices.h"

#include "drivers/common/update_fitness/f1_kernel_driver_common.h"
#include "drivers/common/update_fitness/f2_kernel_driver_common.h"
#include "drivers/common/update_fitness/f3_kernel_driver_common.h"
#include "drivers/common/update_fitness/f4_kernel_driver_common.h"
#include "drivers/common/update_fitness/f5_kernel_driver_common.h"
#include "drivers/common/update_fitness/f6_kernel_driver_common.h"
#include "drivers/common/update_fitness/f7_kernel_driver_common.h"
#include "drivers/common/update_fitness/f8_kernel_driver_common.h"
#include "drivers/common/update_fitness/f9_kernel_driver_common.h"
#include "drivers/common/update_fitness/f10_kernel_driver_common.h"
#include "drivers/common/update_fitness/f11_kernel_driver_common.h"
#include "drivers/common/update_fitness/f12_kernel_driver_common.h"
#include "drivers/common/update_fitness/f13_kernel_driver_common.h"
#include "drivers/common/update_fitness/f14_kernel_driver_common.h"
#include "drivers/common/update_fitness/f15_kernel_driver_common.h"
#include "drivers/common/update_fitness/f16_kernel_driver_common.h"
#include "drivers/common/update_fitness/f17_kernel_driver_common.h"
#include "drivers/common/update_fitness/f18_kernel_driver_common.h"
#include "drivers/common/update_fitness/f19_kernel_driver_common.h"
#include "drivers/common/update_fitness/f20_kernel_driver_common.h"
#include "drivers/common/update_fitness/f21_kernel_driver_common.h"
#include "drivers/common/update_fitness/f22_kernel_driver_common.h"
#include "drivers/common/update_fitness/f23_kernel_driver_common.h"
#include "drivers/common/update_fitness/f24_kernel_driver_common.h"

void launch_update_fitness_shared_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_index,
    device *dev
    );

#endif
