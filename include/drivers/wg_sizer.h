#ifndef _WG_SIZER_H_
#define _WG_SIZER_H_

#include "devices.h"

/* size_t calc_local_size( */
/*     size_t global_size, */
/*     size_t pack_unit_size, */
/*     device *dev */
/*     ); */

size_t calc_local_size(
    size_t global_size,
    size_t pack_unit_size,
    size_t pack_unit_mem,
    device *dev
    );

#endif
