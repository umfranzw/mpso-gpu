#include "drivers/common/update_fitness/update_fitness_shared_kernel_driver_common.h"

void launch_update_fitness_shared_kernel_common(
    void *generic_conf,
    cl_kernel *kernels,
    void *generic_bufs,
    cl_uint iter_index,
    device *dev
    )
{
    ALG_NAME_COMMON(config) *conf = (ALG_NAME_COMMON(config) *) generic_conf;
    ALG_NAME_COMMON(mpso_bufs) *bufs = (ALG_NAME_COMMON(mpso_bufs) *) generic_bufs;

    //static array of function pointers. Each element points to a function that can launch a benchmark kernel.
    static void (*fitness_fcns[])(
        ALG_NAME_COMMON(config)*,
        cl_kernel*,
        ALG_NAME_COMMON(mpso_bufs)*,
        cl_uint,
        device*
        ) = {
        launch_f1_kernel,
        launch_f2_kernel,
        launch_f3_kernel,
        launch_f4_kernel,
        launch_f5_kernel,
        launch_f6_kernel,
        launch_f7_kernel,
        launch_f8_kernel,
        launch_f9_kernel,
        launch_f10_kernel,
        launch_f11_kernel,
        launch_f12_kernel,
        launch_f13_kernel,
        launch_f14_kernel,
        launch_f15_kernel,
        launch_f16_kernel,
        launch_f17_kernel,
        launch_f18_kernel,
        launch_f19_kernel,
        launch_f20_kernel,
        launch_f21_kernel,
        launch_f22_kernel,
        launch_f23_kernel,
        launch_f24_kernel
    };

    fitness_fcns[conf->bench_fcn - 1](
        conf,
        kernels,
        bufs,
        iter_index,
        dev
        );
}
