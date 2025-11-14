#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"{
#endif

typedef unsigned int  WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx{
    WORD h[8];
    BYTE b[32];
} SHA256;

// Host SHA-256
void sha256_init(SHA256 *ctx);
void sha256_transform(SHA256 *ctx, const BYTE *msg);
void sha256(SHA256 *ctx, const BYTE *msg, size_t len);

// Copy constants to device constant memory
void sha256_cuda_init_constants(void);

// Device constants (declared in sha256.cu, used in hw4.cu)
extern __constant__ WORD k_dev[64];
extern __constant__ WORD iv_dev[8];

#ifdef __cplusplus
}
#endif

#endif