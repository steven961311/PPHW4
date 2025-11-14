#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "sha256.h"

#define _rotr(v, s) ((v)>>(s) | (v)<<(32-(s)))
#define _swap(x, y) (((x)^=(y)), ((y)^=(x)), ((x)^=(y)))

#ifdef __cplusplus
extern "C"{
#endif

// Host constants
static const WORD k_host[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// Device constants (visible to other TUs)
__constant__ WORD k_dev[64];
__constant__ WORD iv_dev[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

void sha256_cuda_init_constants(void){
    cudaMemcpyToSymbol(k_dev, k_host, sizeof(k_host));
}

void sha256_init(SHA256 *ctx){
    ctx->h[0]=0x6a09e667; ctx->h[1]=0xbb67ae85; ctx->h[2]=0x3c6ef372; ctx->h[3]=0xa54ff53a;
    ctx->h[4]=0x510e527f; ctx->h[5]=0x9b05688c; ctx->h[6]=0x1f83d9ab; ctx->h[7]=0x5be0cd19;
}

#define _rotr32(x,n) (((x)>>(n)) | ((x)<<(32-(n))))

void sha256_transform(SHA256 *ctx, const BYTE *msg){
    WORD a,b,c,d,e,f,g,h, i,j;
    WORD w[64];

    for(i=0,j=0;i<16;++i,j+=4)
        w[i] = ( (WORD)msg[j]<<24 ) | ( (WORD)msg[j+1]<<16 ) | ( (WORD)msg[j+2]<<8 ) | (WORD)msg[j+3];

    for(i=16;i<64;++i){
        WORD s0 = (_rotr32(w[i-15],7)) ^ (_rotr32(w[i-15],18)) ^ (w[i-15]>>3);
        WORD s1 = (_rotr32(w[i-2],17)) ^ (_rotr32(w[i-2],19))  ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    a=ctx->h[0]; b=ctx->h[1]; c=ctx->h[2]; d=ctx->h[3];
    e=ctx->h[4]; f=ctx->h[5]; g=ctx->h[6]; h=ctx->h[7];

    for(i=0;i<64;++i){
        WORD S0 = (_rotr32(a,2)) ^ (_rotr32(a,13)) ^ (_rotr32(a,22));
        WORD S1 = (_rotr32(e,6)) ^ (_rotr32(e,11)) ^ (_rotr32(e,25));
        WORD ch = (e & f) ^ ((~e) & g);
        WORD maj = (a & b) ^ (a & c) ^ (b & c);
        WORD t1 = h + S1 + ch + k_host[i] + w[i];
        WORD t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    ctx->h[0]+=a; ctx->h[1]+=b; ctx->h[2]+=c; ctx->h[3]+=d;
    ctx->h[4]+=e; ctx->h[5]+=f; ctx->h[6]+=g; ctx->h[7]+=h;
}

void sha256(SHA256 *ctx, const BYTE *msg, size_t len){
    sha256_init(ctx);
    WORD i,j;

    size_t remain = len % 64;
    size_t total_len = len - remain;

    for(i=0;i<total_len;i+=64) sha256_transform(ctx, &msg[i]);

    BYTE m[64] = {};
    for(i=total_len,j=0;i<len;++i,++j) m[j] = msg[i];

    m[j++] = 0x80;
    if(j>56){ sha256_transform(ctx, m); memset(m,0,sizeof(m)); }

    unsigned long long L = len * 8ULL;
    m[63]= (BYTE)L; m[62]=(BYTE)(L>>8); m[61]=(BYTE)(L>>16); m[60]=(BYTE)(L>>24);
    m[59]=(BYTE)(L>>32); m[58]=(BYTE)(L>>40); m[57]=(BYTE)(L>>48); m[56]=(BYTE)(L>>56);
    sha256_transform(ctx, m);

    // present as big-endian bytes in ctx->b
    for(i=0;i<32;i+=4){ _swap(ctx->b[i],ctx->b[i+3]); _swap(ctx->b[i+1],ctx->b[i+2]); }
}

#ifdef __cplusplus
}
#endif

// Device compressor is defined in hw4.cu