//
//  timing.h
//  pthreads2018
//
//  Created by Patricio Bulic on 10/10/2018.
//  Copyright © 2018 Patricio Bulic. All rights reserved.
//


#ifndef TIMING_H
#define TIMING_H

#include <time.h>
#define BEGCLOCK(name) {clock_t begclock##name = clock ();
#define ENDCLOCK(name) clock_t endclock##name = clock ();              \
fprintf (stderr, "CLOCK(%s): %lf\n", #name,                          \
((double)(endclock##name - begclock##name)) / CLOCKS_PER_SEC);}


#endif