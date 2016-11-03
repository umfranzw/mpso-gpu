#include "drivers/wg_sizer.h"

/* size_t calc_local_size( */
/*     size_t global_size, */
/*     size_t pack_unit_size, */
/*     device *dev */
/*     ) */
/* { */
/*     size_t biggest_size = dev->max_workgroup_size - (dev->max_workgroup_size % pack_unit_size); */

/*     size_t local_size; */
/*     for (local_size = biggest_size; (global_size % local_size) != 0 && local_size > 0; local_size -= pack_unit_size); */

/*     return local_size; */
/* } */

size_t calc_local_size(
    size_t global_size,
    size_t pack_unit_size,
    size_t pack_unit_mem,
    device *dev
    )
{
    size_t biggest_size = dev->max_workgroup_size - (dev->max_workgroup_size % pack_unit_size);

    size_t local_size;
    for (local_size = biggest_size;
         ((global_size % local_size) != 0 ||
             dev->max_local_mem < (local_size / pack_unit_size) * pack_unit_mem) &&
             local_size > 0;
         local_size -= pack_unit_size);

    return local_size;
}
