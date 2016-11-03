#define KERNELS_FILE "kernels/kernels.cl"

//constants needed for both device and host
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/mpso-gpu/include/global_constants.h"

//Random123 library (does not contain any kernels)
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/mpso-gpu/include/Random123/philox.h"
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/mpso-gpu/include/Random123/threefry.h"
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/mpso-gpu/include/Random123/u01.h"

//This file contains common routines for stuff like rotating and permuting vectors (no kernels).
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/Debug/kernels/common/bench_tools.cl"

#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/Debug/kernels/common/rand_tools.cl"

//These must be included in alphabetical order according to function name.

#if ALG == ALG_ALT
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/Debug/kernels/kernels_alt.cl"
#elif ALG == ALG_REG
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/Debug/kernels/kernels_reg.cl"
#elif ALG == ALG_MCS
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/Debug/kernels/kernels_mcs.cl"
#elif ALG == ALG_TM
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/Debug/kernels/kernels_tm.cl"
#elif ALG == ALG_GA
#include "C:/Users/Wayne/Documents/Visual Studio 2010/Projects/mpso-gpu/Debug/kernels/kernels_ga.cl"
#endif
