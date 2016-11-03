#ifndef _GLOBAL_CONSTANTS_H_
#define _GLOBAL_CONSTANTS_H_

/****************************
 * Path Parameters *
 ****************************/
#define KERNELS_FILENAME "kernels/kernels.cl"

/***********************
  * Control Parameters *
  **********************/
#define LAUNCH_WARNINGS 0 //makes all drivers print a message each time they launch a kernel
#define FIXED_SEED 0
#define ROTATE 1 //enables rotation in the fitness functions (if the function uses it)
#define PERMUTE 1 //enables permutation in fitness functions (if the function uses it)
#define DEBUG 0
#define SWAP_OFFSET 50

/************************
 * Algorithm Parameters *
 ************************/
#define ALG_ALT 0
#define ALG_REG 1
#define ALG_MCS 2
#define ALG_TM 3
#define ALG_GA 4

#define ALG ALG_MCS

typedef enum
{
    TYPE_PSO = 0,
    TYPE_GA
} swarm_types;

typedef enum
{
    PARTICLE_INIT_STREAM,
    UPDATE_POS_VEL_STREAM,
    CROSSOVER_STREAM,
    MUTATION_STREAM
} rand_streams;

#define MIN_OMEGA 0.4f

//These are specific to the TM alg:
#define MIN_MIPS_TM 10
#define MAX_MIPS_TM 500
#define MIN_TASK_INST_TM 1
#define MAX_TASK_INST_TM 1000

/****************
 * Other Macros *
 ****************/
//convert parameter directly to a string constant
#define STR(s) #s
//expands the parameter before converting to a string
#define XSTR(s) STR(s)
#define EXPAND(s) s

//generates an appropriate suffix based on ALG - e.g. test_alt
#if ALG == ALG_ALT
#define ALG_NAME(prefix) prefix ## _alt
#define ALG_NAME_CAPS(prefix) prefix ## _ALT
#define ALG_NAME_COMMON(prefix) ALG_NAME(prefix)
#define ALG_NAME_COMMON_CAPS(prefix) ALG_NAME_CAPS(prefix)

#elif ALG == ALG_REG
#define ALG_NAME(prefix) prefix ## _reg
#define ALG_NAME_CAPS(prefix) prefix ## _REG
#define ALG_NAME_COMMON(prefix) ALG_NAME(prefix)
#define ALG_NAME_COMMON_CAPS(prefix) ALG_NAME_CAPS(prefix)

#elif ALG == ALG_MCS
#define ALG_NAME(prefix) prefix ## _mcs
#define ALG_NAME_CAPS(prefix) prefix ## _MCS
#define ALG_NAME_COMMON(prefix) ALG_NAME(prefix)
#define ALG_NAME_COMMON_CAPS(prefix) ALG_NAME_CAPS(prefix)

#elif ALG == ALG_TM
#define ALG_NAME(prefix) prefix ## _tm
#define ALG_NAME_CAPS(prefix) prefix ## _TM
#define ALG_NAME_COMMON(prefix) prefix ## _reg //to prevent compilation errors in update fitness common - e.g. since buffers_tm doesn't have optimum_buf
#define ALG_NAME_COMMON_CAPS(prefix) prefix ## _REG

#elif ALG == ALG_GA
#define ALG_NAME(prefix) prefix ## _ga
#define ALG_NAME_CAPS(prefix) prefix ## _GA
#define ALG_NAME_COMMON(prefix) ALG_NAME(prefix)
#define ALG_NAME_COMMON_CAPS(prefix) ALG_NAME_CAPS(prefix)
#endif

//generates a config file name (this is not a string) based on the current ALG setting - e.g. config_alt.txt
#define CONFIG_FILE_NAME() ALG_NAME(config) ## .txt
//generates a config file name as a string constant - e.g. "config_alt.txt"
#define CONFIG_FILE_STR() XSTR(config/ ## EXPAND(CONFIG_FILE_NAME()))

//generates a header name (this is not a string) based on the current ALG setting - e.g. test_alt.h
#define ALG_HEADER_NAME(prefix) ALG_NAME(prefix) ## .h
#define ALG_HEADER_NAME_COMMON(prefix) ALG_NAME_COMMON(prefix) ## .h
//generates a header name as a string constant - e.g. "test_alt.h"
#define ALG_HEADER_STR(prefix) XSTR(ALG_HEADER_NAME(prefix))
#define ALG_HEADER_STR_COMMON(prefix) XSTR(ALG_HEADER_NAME_COMMON(prefix))

#endif
