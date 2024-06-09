#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct PinkTrombone PinkTrombone;
PinkTrombone* PinkTrombone_new(int n);
void PinkTrombone_delete(PinkTrombone* this_);
int PinkTrombone_control(PinkTrombone* this_, double* data, size_t len);
int PinkTrombone_process(PinkTrombone* this_, double* data, size_t len);

#ifdef __cplusplus
}
#endif
