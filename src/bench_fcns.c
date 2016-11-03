#include "bench_fcns.h"

void init_bench_fcn_info(
    bench_fcn_info *info
    )
{
    info[0].max_axis_val = 100;
    info[0].need_rot_matrix = 0;
    info[0].need_perm_vec = 0;
    info[0].need_opt_vec = 1;

    info[1].max_axis_val = 5;
    info[1].need_rot_matrix = 0;
    info[1].need_perm_vec = 0;
    info[1].need_opt_vec = 1;

    info[2].max_axis_val = 32;
    info[2].need_rot_matrix = 0;
    info[2].need_perm_vec = 0;
    info[2].need_opt_vec = 1;

    info[3].max_axis_val = 100;
    info[3].need_rot_matrix = 1;
    info[3].need_perm_vec = 1;
    info[3].need_opt_vec = 1;

    info[4].max_axis_val = 5;
    info[4].need_rot_matrix = 1;
    info[4].need_perm_vec = 1;
    info[4].need_opt_vec = 1;

    info[5].max_axis_val = 32;
    info[5].need_rot_matrix = 1;
    info[5].need_perm_vec = 1;
    info[5].need_opt_vec = 1;

    info[6].max_axis_val = 100;
    info[6].need_rot_matrix = 0;
    info[6].need_perm_vec = 1;
    info[6].need_opt_vec = 1;
    
    info[7].max_axis_val = 100;
    info[7].need_rot_matrix = 0;
    info[7].need_perm_vec = 1;
    info[7].need_opt_vec = 1;

    info[8].max_axis_val = 100;
    info[8].need_rot_matrix = 1;
    info[8].need_perm_vec = 1;
    info[8].need_opt_vec = 1;

    info[9].max_axis_val = 5;
    info[9].need_rot_matrix = 1;
    info[9].need_perm_vec = 1;
    info[9].need_opt_vec = 1;

    info[10].max_axis_val = 32;
    info[10].need_rot_matrix = 1;
    info[10].need_perm_vec = 1;
    info[10].need_opt_vec = 1;

    info[11].max_axis_val = 100;
    info[11].need_rot_matrix = 0;
    info[11].need_perm_vec = 1;
    info[11].need_opt_vec = 1;

    info[12].max_axis_val = 100;
    info[12].need_rot_matrix = 0;
    info[12].need_perm_vec = 1;
    info[12].need_opt_vec = 1;

    info[13].max_axis_val = 100;
    info[13].need_rot_matrix = 1;
    info[13].need_perm_vec = 1;
    info[13].need_opt_vec = 1;

    info[14].max_axis_val = 5;
    info[14].need_rot_matrix = 1;
    info[14].need_perm_vec = 1;
    info[14].need_opt_vec = 1;

    info[15].max_axis_val = 32;
    info[15].need_rot_matrix = 1;
    info[15].need_perm_vec = 1;
    info[15].need_opt_vec = 1;

    info[16].max_axis_val = 100;
    info[16].need_rot_matrix = 0;
    info[16].need_perm_vec = 1;
    info[16].need_opt_vec = 1;

    info[17].max_axis_val = 100;
    info[17].need_rot_matrix = 0;
    info[17].need_perm_vec = 1;
    info[17].need_opt_vec = 1;

    info[18].max_axis_val = 100;
    info[18].need_rot_matrix = 0;
    info[18].need_perm_vec = 0;
    info[18].need_opt_vec = 1;

    info[19].max_axis_val = 100;
    info[19].need_rot_matrix = 0;
    info[19].need_perm_vec = 0;
    info[19].need_opt_vec = 1;

    info[20].max_axis_val = 100;
    info[20].need_rot_matrix = 0;
    info[20].need_perm_vec = 0;
    info[20].need_opt_vec = 0;

    info[21].max_axis_val = 5;
    info[21].need_rot_matrix = 0;
    info[21].need_perm_vec = 0;
    info[21].need_opt_vec = 0;

    info[22].max_axis_val = 32;
    info[22].need_rot_matrix = 0;
    info[22].need_perm_vec = 0;
    info[22].need_opt_vec = 0;

    info[23].max_axis_val = 100;
    info[23].need_rot_matrix = 0;
    info[23].need_perm_vec = 0;
    info[23].need_opt_vec = 0;
}

