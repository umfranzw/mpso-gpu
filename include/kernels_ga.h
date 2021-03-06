#ifndef _KERNELS_GA_H_
#define _KERNELS_GA_H_

typedef enum
{
    CROSS_MUT_VEC_KERNEL_GA,

    CROSSOVER_VEC_KERNEL_GA,

    F1_KERNEL_GA,
    F10_KERNEL_GA,
    F11_KERNEL_GA,
    F12_KERNEL_GA,
    F13_KERNEL_GA,
    F14_KERNEL_GA,
    F15_KERNEL_GA,
    F16_KERNEL_GA,
    F17_KERNEL_GA,
    F18_KERNEL_GA,
    F19_KERNEL_GA,
    F2_KERNEL_GA,
    F20_KERNEL_GA,
    F21_KERNEL_GA,
    F22_KERNEL_GA,
    F23_KERNEL_GA,
    F24_KERNEL_GA,
    F3_KERNEL_GA,
    F4_KERNEL_GA,
    F5_KERNEL_GA,
    F6_KERNEL_GA,
    F7_KERNEL_GA,
    F8_KERNEL_GA,
    F9_KERNEL_GA,
    
    FIND_BEST_WORST_ALT_KERNEL_GA,
    FIND_BEST_WORST_ALT2_KERNEL_GA,
    FIND_BEST_WORST_VEC2_KERNEL_GA,

    FIND_MIN_VEC_KERNEL_GA,

    INIT_ROT_MATRIX_VEC_KERNEL_GA,

    MUT_RESTORE_VEC_KERNEL_GA,

    PARTICLE_INIT_VEC_KERNEL_GA,

    PERMUTE_VEC_KERNEL_GA,

    SWAP_PARTICLES_VEC_KERNEL_GA,

    //this is for the GA
    UPDATE_BEST_VALS_VEC_KERNEL_GA,
    
    //this is for MPSO-MCS
    UPDATE_BEST_VALS_VEC_KERNEL_GA2,
    
    UPDATE_POS_VEL_VEC_KERNEL_GA,

    UPDATE_SAMPLES_VEC_KERNEL_GA,
    
    NUM_KERNELS_GA
} kernel_names_ga;

#endif
