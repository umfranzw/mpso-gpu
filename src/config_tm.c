#include "config_tm.h"

cl_int parse_config_tm(
    FILE *file, 
    config_tm *conf
    )
{
    return parse_param(file, "%u,", (void *) &(conf->num_swarms)) &&
        parse_param(file, "%u,", (void *) &(conf->num_sparticles)) &&
        parse_param(file, "%u,", (void *) &(conf->num_dims)) &&
        parse_param(file, "%u,", (void *) &(conf->max_iters)) &&
        parse_param(file, "%f,", (void *) &(conf->omega)) &&
        parse_param(file, "%f,", (void *) &(conf->omega_decay)) &&
        parse_param(file, "%f,", (void *) &(conf->c1)) &&
        parse_param(file, "%f,", (void *) &(conf->c2)) &&
        parse_param(file, "%u,", (void *) &(conf->exchange_iters)) &&
        parse_param(file, "%u,", (void *) &(conf->num_exchange)) &&
        parse_param(file, "%u,", (void *) &(conf->num_machines)) &&
        parse_param(file, "%u,", (void *) &(conf->num_reps)) &&
        parse_param(file, "%u", (void *) &(conf->fitness_sample_interval));
}

cl_int get_next_config_tm(
    FILE *config_file,
    config_tm *config
    )
{
    cl_int ret_val = 0;
    if (!feof(config_file))
    {
        ret_val = parse_config_tm(
            config_file,
            config
            );
        
        if (!ret_val)
        {
            printf("Error parsing config file.\n");
        }
    }

    return ret_val;
}
