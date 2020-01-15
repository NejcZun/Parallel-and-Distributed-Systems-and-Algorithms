#ifndef TIMING_H
#define TIMING_H

#include <time.h>
#define BEGCLOCK(name) {clock_t begclock##name = clock ();
#define ENDCLOCK(name) clock_t endclock##name = clock ();              \
fprintf (stderr, "CLOCK(%s): %lf\n", #name,                          \
((double)(endclock##name - begclock##name)) / CLOCKS_PER_SEC);}


#endif