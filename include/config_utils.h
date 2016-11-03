#ifndef _CONFIG_UTILS_H_
#define _CONFIG_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"

cl_uint get_num_configs(FILE *config_file);
cl_int parse_param(FILE *file, char *patt, void *buf);
void skip_whitespace(FILE *file);

#endif