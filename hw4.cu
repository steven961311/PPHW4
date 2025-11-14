#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#include "sha256.h" 

typedef unsigned int WORD;

typedef struct _block{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
} HashBlock;

struct DeviceContext {
    WORD *d_midstate = nullptr;
    unsigned char *d_target_be = nullptr;
    unsigned int *d_found_nonce = nullptr;
    int *d_found_flag = nullptr;
    cudaStream_t stream = nullptr;
    bool initialized = false;
} g_dev;
static int g_override_blocks = -1;
static int g_override_threads = -1;

static bool init_device_context(){
    if (g_dev.initialized) return true;
    sha256_cuda_init_constants();
    if (cudaStreamCreateWithFlags(&g_dev.stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    if (cudaMalloc(&g_dev.d_midstate, 8*sizeof(WORD)) != cudaSuccess) return false;
    if (cudaMalloc(&g_dev.d_target_be, 32*sizeof(unsigned char)) != cudaSuccess) return false;
    if (cudaMalloc(&g_dev.d_found_nonce, sizeof(unsigned int)) != cudaSuccess) return false;
    if (cudaMalloc(&g_dev.d_found_flag, sizeof(int)) != cudaSuccess) return false;
    g_dev.initialized = true;
    return true;
}

static void destroy_device_context(){
    if (!g_dev.initialized) return;
    cudaFree(g_dev.d_midstate);
    cudaFree(g_dev.d_target_be);
    cudaFree(g_dev.d_found_nonce);
    cudaFree(g_dev.d_found_flag);
    cudaStreamDestroy(g_dev.stream);
    g_dev = DeviceContext{};
}

// ================= utils =================
static inline unsigned char decode(unsigned char c){
    switch(c){
        case 'a': return 0x0a; case 'b': return 0x0b; case 'c': return 0x0c;
        case 'd': return 0x0d; case 'e': return 0x0e; case 'f': return 0x0f;
        default:  return (unsigned char)(c - '0');
    }
}
static void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len){
    assert(string_len % 2 == 0);
    size_t s=0, b=string_len/2-1;
    for(; s<string_len; s+=2, --b) out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
}
static void print_hex(const unsigned char* hex, size_t len){ for(size_t i=0;i<len;++i) printf("%02x", hex[i]); }
static void print_hex_inverse(const unsigned char* hex, size_t len){ for(int i=(int)len-1;i>=0;--i) printf("%02x", hex[i]); }
static void getline2(char *str, size_t len, FILE *fp){
    int i=0; while(i<(int)len && (str[i]=fgetc(fp))!=EOF && str[i++]!='\n'); str[len-1] = '\0';
}

// ================= host helpers =================
static void double_sha256_host(SHA256 *out, const unsigned char *bytes, size_t len){
    SHA256 tmp; sha256(&tmp,(unsigned char*)bytes,len); sha256(out,(unsigned char*)&tmp,sizeof(tmp));
}
static inline WORD be32(const unsigned char *p){
    return ((WORD)p[0]<<24) | ((WORD)p[1]<<16) | ((WORD)p[2]<<8) | (WORD)p[3];
}

// ================= device SHA-256 compressor (words) =================
extern __constant__ WORD k_dev[64];
extern __constant__ WORD iv_dev[8];

__device__ __forceinline__ WORD rotr32(WORD x, int s){
#if __CUDA_ARCH__ >= 300
    return __funnelshift_r(x, x, s);
#else
    return (x >> s) | (x << (32 - s));
#endif
}
__device__ __forceinline__ WORD Ch(WORD x, WORD y, WORD z){ return (x & y) ^ (~x & z); }
__device__ __forceinline__ WORD Maj(WORD x, WORD y, WORD z){ return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ WORD Sigma0(WORD x){ return rotr32(x,2) ^ rotr32(x,13) ^ rotr32(x,22); }
__device__ __forceinline__ WORD Sigma1(WORD x){ return rotr32(x,6) ^ rotr32(x,11) ^ rotr32(x,25); }
__device__ __forceinline__ WORD sigma0(WORD x){ return rotr32(x,7) ^ rotr32(x,18) ^ (x>>3); }
__device__ __forceinline__ WORD sigma1(WORD x){ return rotr32(x,17)^ rotr32(x,19) ^ (x>>10); }

static __device__ __forceinline__ void sha256_compress_words(WORD state[8], const WORD w_init[16]){
    WORD a=state[0], b=state[1], c=state[2], d=state[3];
    WORD e=state[4], f=state[5], g=state[6], h=state[7];

    WORD w0 = w_init[0];  WORD w1 = w_init[1];  WORD w2 = w_init[2];  WORD w3 = w_init[3];
    WORD w4 = w_init[4];  WORD w5 = w_init[5];  WORD w6 = w_init[6];  WORD w7 = w_init[7];
    WORD w8 = w_init[8];  WORD w9 = w_init[9];  WORD w10= w_init[10]; WORD w11= w_init[11];
    WORD w12= w_init[12]; WORD w13= w_init[13]; WORD w14= w_init[14]; WORD w15= w_init[15];

#pragma unroll 64
    for(int i=0;i<64;i++){
        WORD wt;
        switch(i){
            case 0:  wt=w0; break; case 1:  wt=w1; break; case 2:  wt=w2; break; case 3:  wt=w3; break;
            case 4:  wt=w4; break; case 5:  wt=w5; break; case 6:  wt=w6; break; case 7:  wt=w7; break;
            case 8:  wt=w8; break; case 9:  wt=w9; break; case 10: wt=w10; break; case 11: wt=w11; break;
            case 12: wt=w12; break; case 13: wt=w13; break; case 14: wt=w14; break; case 15: wt=w15; break;
            default:
                wt = w0 + sigma0(w1) + w9 + sigma1(w14);
                w0 = w1;  w1 = w2;  w2 = w3;  w3 = w4;  w4 = w5;  w5 = w6;  w6 = w7;
                w7 = w8;  w8 = w9;  w9 = w10; w10= w11; w11= w12; w12= w13; w13= w14; w14= w15;
                w15 = wt;
                break;
        }
        WORD t1 = h + Sigma1(e) + Ch(e,f,g) + k_dev[i] + wt;
        WORD t2 = Sigma0(a) + Maj(a,b,c);
        h=g; g=f; f=e; e=d + t1; d=c; c=b; b=a; a=t1 + t2;
    }

    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__device__ __forceinline__ WORD bswap32(WORD x){
    x = (x >> 16) | (x << 16);
    return ((x & 0xFF00FF00u) >> 8) | ((x & 0x00FF00FFu) << 8);
}

// ================= kernels =================

__global__ void mine_kernel_midstate(const WORD* __restrict__ midstate,
                                     WORD w0_lastMerkle, WORD w1_ntime, WORD w2_nbits,
                                     const unsigned char* __restrict__ target_be, // 32B BE
                                     unsigned int start_nonce,
                                     unsigned int *found_nonce,
                                     int *found_flag)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    WORD w_base[16];
    w_base[0]=w0_lastMerkle; w_base[1]=w1_ntime; w_base[2]=w2_nbits;
    w_base[4]=0x80000000u; w_base[5]=0; w_base[6]=0; w_base[7]=0;
    w_base[8]=0; w_base[9]=0; w_base[10]=0; w_base[11]=0; w_base[12]=0; w_base[13]=0; w_base[14]=0;
    w_base[15]=0x00000280u;

    const uint64_t SPACE = 0xffffffff;

    for (uint64_t off = tid; off < SPACE; off += stride) {
        if (atomicAdd(found_flag, 0) != 0) return;

        const unsigned int nonce = (unsigned int)(start_nonce + off);
        WORD w[16];
#pragma unroll
        for (int i=0;i<16;++i) w[i] = w_base[i];
        w[3] = bswap32(nonce);

        WORD st1[8];
#pragma unroll
        for(int i=0;i<8;++i) st1[i] = midstate[i];
        sha256_compress_words(st1, w);

        WORD st2[8];
#pragma unroll
        for(int i=0;i<8;++i) st2[i] = iv_dev[i];
        WORD w2b[16];
#pragma unroll
        for(int i=0;i<8;++i) w2b[i] = st1[i];
        w2b[8]=0x80000000u; w2b[9]=0; w2b[10]=0; w2b[11]=0; w2b[12]=0; w2b[13]=0; w2b[14]=0; w2b[15]=0x00000100u;
        sha256_compress_words(st2, w2b);

        // Build BIG-ENDIAN digest bytes (per word, in order)
        unsigned char hb_be[32];
#pragma unroll
        for (int i=0;i<8;++i){
            WORD ww = st2[i];
            hb_be[4*i+0] = (unsigned char)(ww >> 24);
            hb_be[4*i+1] = (unsigned char)(ww >> 16);
            hb_be[4*i+2] = (unsigned char)(ww >>  8);
            hb_be[4*i+3] = (unsigned char)(ww >>  0);
        }

        bool lt=false, gt=false;
        for (int i=0;i<32;++i){
            unsigned char a = hb_be[31-i], b = target_be[i];
            if (a < b){ lt=true; break; }
            if (a > b){ gt=true; break; }
        }
        if (lt || !gt){
            if (atomicCAS(found_flag, 0, 1) == 0) *found_nonce = nonce;
            return;
        }
    }
}

// Debug: write 32B BIG-ENDIAN hash for a specific nonce (midstate path)
__global__ void debug_one_nonce_kernel(const WORD* __restrict__ midstate,
                                       WORD w0_lastMerkle, WORD w1_ntime, WORD w2_nbits,
                                       unsigned int nonce,
                                       unsigned char* __restrict__ out_be32)
{
    if (threadIdx.x==0 && blockIdx.x==0){
        WORD w[16];
        w[0]=w0_lastMerkle; w[1]=w1_ntime; w[2]=w2_nbits; w[3]=bswap32(nonce);
        w[4]=0x80000000u; w[5]=0; w[6]=0; w[7]=0;
        w[8]=0; w[9]=0; w[10]=0; w[11]=0; w[12]=0; w[13]=0; w[14]=0; w[15]=0x00000280u;

        WORD st1[8];
#pragma unroll
        for(int i=0;i<8;++i) st1[i] = midstate[i];
        sha256_compress_words(st1, w);

        WORD st2[8];
#pragma unroll
        for(int i=0;i<8;++i) st2[i] = iv_dev[i];
        WORD w2b[16];
#pragma unroll
        for(int i=0;i<8;++i) w2b[i] = st1[i];
        w2b[8]=0x80000000u; w2b[9]=0; w2b[10]=0; w2b[11]=0; w2b[12]=0; w2b[13]=0; w2b[14]=0; w2b[15]=0x00000100u;
        sha256_compress_words(st2, w2b);

#pragma unroll
        for (int i=0;i<8;++i){
            WORD ww = st2[i];
            out_be32[4*i+0] = (unsigned char)(ww >> 24);
            out_be32[4*i+1] = (unsigned char)(ww >> 16);
            out_be32[4*i+2] = (unsigned char)(ww >>  8);
            out_be32[4*i+3] = (unsigned char)(ww >>  0);
        }

        
    }
}

// ================= merkle (CPU) =================
static void calc_merkle_root(unsigned char *root, int count, char **branch){
    size_t total_count = count;
    unsigned char *raw = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];
    for(int i=0;i<total_count;++i){
        list[i] = raw + i*32;
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }
    list[total_count] = raw + total_count*32;
    while(total_count > 1){
        if(total_count % 2 == 1) memcpy(list[total_count], list[total_count-1], 32);
        int j=0;
        for(int i=0;i<total_count;i+=2,++j) double_sha256_host((SHA256*)list[j], list[i], 64);
        total_count = j;
    }
    memcpy(root, list[0], 32);
    delete[] raw; delete[] list;
}

// ================= solve one block =================
static void solve(FILE *fin, FILE *fout, bool debug_one){
    char version[9], prevhash[65], ntime[9], nbits[9];
    int tx=0;
    getline2(version,9,fin);
    getline2(prevhash,65,fin);
    getline2(ntime,9,fin);
    getline2(nbits,9,fin);
    if (fscanf(fin, "%d\n", &tx) != 1) { fprintf(stderr,"failed to read tx count\n"); return; }

    char *raw_merkle_branch = new char[tx*65];
    char **merkle_branch = new char*[tx];
    for(int i=0;i<tx;++i){
        merkle_branch[i] = raw_merkle_branch + i*65;
        getline2(merkle_branch[i],65,fin);
        merkle_branch[i][64] = '\0';
    }

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    HashBlock block;
    convert_string_to_little_endian_bytes((unsigned char*)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,  prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char*)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char*)&block.ntime, ntime, 8);
    block.nonce = 0;

    // nBits -> target (LE), then convert to big-endian for device compare
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_le[32] = {};
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    target_le[sb    ] = (unsigned char)(mant << rb);
    target_le[sb + 1] = (unsigned char)(mant >> (8-rb));
    target_le[sb + 2] = (unsigned char)(mant >> (16-rb));
    target_le[sb + 3] = (unsigned char)(mant >> (24-rb));

    unsigned char target_be[32];
    for (int i=0;i<32;++i) target_be[i] = target_le[31 - i];

    // Pretty prints
    //printf("merkle root(little): "); print_hex(merkle_root,32); printf("\n");
    //printf("merkle root(big):    "); print_hex_inverse(merkle_root,32); printf("\n");
    //printf("Block info (big): \n");
    //printf("  version:  %s\n", version);
    //printf("  pervhash: %s\n", prevhash);
    //printf("  merkleroot: "); print_hex_inverse(merkle_root,32); printf("\n");
    //printf("  nbits:    %s\n", nbits);
    //printf("  ntime:    %s\n", ntime);
    //printf("Target value (big): "); print_hex_inverse(target_le,32); printf("\n");

    // ---- serialize exact 80-byte header (little-endian fields, as in Bitcoin) ----
    unsigned char header80[80];
    memcpy(header80 + 0,  &block.version,   4);
    memcpy(header80 + 4,  block.prevhash,  32);
    memcpy(header80 + 36, block.merkle_root, 32);
    memcpy(header80 + 68, &block.ntime,     4);
    memcpy(header80 + 72, &block.nbits,     4);
    memcpy(header80 + 76, &block.nonce,     4);

    // midstate (CPU): compress first 64 bytes only
    SHA256 ctx; sha256_init(&ctx);
    sha256_transform(&ctx, header80); // 64-byte chunk
    WORD midstate_h[8]; for(int i=0;i<8;++i) midstate_h[i] = ctx.h[i];

    // chunk1 words (bytes 64..79): [last 4 bytes of merkle_root][ntime][nbits][nonce]
    const unsigned char* tail = header80 + 64;
    WORD w0_lastMerkle = be32(&tail[0]);
    WORD w1_ntime      = be32(&tail[4]);
    WORD w2_nbits      = be32(&tail[8]);

    // ---- GPU setup ----
    if (!init_device_context()){
        fprintf(stderr, "Failed to initialize CUDA context\n");
        delete[] merkle_branch; delete[] raw_merkle_branch;
        return;
    }

    unsigned int zero32=0; int zero=0;
    cudaMemcpyAsync(g_dev.d_midstate,   midstate_h,  8*sizeof(WORD),
                    cudaMemcpyHostToDevice, g_dev.stream);
    cudaMemcpyAsync(g_dev.d_target_be,  target_be,  32*sizeof(unsigned char),
                    cudaMemcpyHostToDevice, g_dev.stream);
    cudaMemcpyAsync(g_dev.d_found_nonce, &zero32, sizeof(zero32),
                    cudaMemcpyHostToDevice, g_dev.stream);
    cudaMemcpyAsync(g_dev.d_found_flag,  &zero,   sizeof(zero),
                    cudaMemcpyHostToDevice, g_dev.stream);

    // Optional: debug known nonce 0x16E80820 first (should print:
    // 0000000000000000000870497004514bd3807cdc98b9f3a57038faf5df04144f)
    if (debug_one){
        unsigned char *d_out=nullptr; cudaMalloc(&d_out, 32);
        debug_one_nonce_kernel<<<1,1,0,g_dev.stream>>>(g_dev.d_midstate, w0_lastMerkle, w1_ntime, w2_nbits,
                                        384305184, d_out);
        cudaStreamSynchronize(g_dev.stream);
        unsigned char hb_be[32]; cudaMemcpyAsync(hb_be, d_out, 32, cudaMemcpyDeviceToHost, g_dev.stream);
        cudaStreamSynchronize(g_dev.stream);
        //printf("[GPU] known nonce 0x16E80820 hash (big): ");
        //for (int i=0;i<32;++i) printf("%02x", hb_be[i]); // already big-endian
        //printf("\n");
        cudaFree(d_out);
    }

        // ---- launch miner ----
        // Choose block/grid sizes tuned to device occupancy. Defaults were (256, 10240).
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);

        // Allow overrides via env vars (for testing/profiling)
        const char* env_block = getenv("HW4_BLOCKDIM");
        const char* env_grid  = getenv("HW4_GRIDDIM");

        int block_size = 256;
        int grid_size  = 10240;

        if (env_block) block_size = atoi(env_block);
        if (env_grid)  grid_size  = atoi(env_grid);
        if (g_override_threads > 0) block_size = g_override_threads;
        if (g_override_blocks > 0)  grid_size  = g_override_blocks;

        // If not overridden, try to compute a good grid size using occupancy API
        if (!env_block && !env_grid && g_override_threads <= 0 && g_override_blocks <= 0) {
            int minGrid, blockPerSm;
            // Query maximum potential block size for our kernel
            cudaError_t occ = cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockPerSm,
                                                                mine_kernel_midstate, 0, 0);
            if (occ == cudaSuccess && blockPerSm > 0) {
                // blockPerSm is a suggested block size; clamp to sensible values
                block_size = blockPerSm;
                if (block_size < 64) block_size = 64;
                if (block_size > 1024) block_size = 1024;

                // Make grid proportional to SM count to saturate device
                // Aim for 64 blocks per SM to cover latency and multi-warp scheduling
                grid_size = prop.multiProcessorCount * 64;
                if (grid_size < 1024) grid_size = 1024; // baseline
            }
        }

        // Round grid_size up to avoid tiny remainder when partitioning 2^32 space
        unsigned int blocks = (unsigned int)grid_size;
        unsigned int threads = (unsigned int)block_size;
        dim3 blockDim(threads);
        dim3 gridDim(blocks);

        //printf("Launching kernel with grid=%u block=%u (SMs=%d, name=%s)\n",
        //       blocks, threads, prop.multiProcessorCount, prop.name);

        mine_kernel_midstate<<<gridDim, blockDim, 0, g_dev.stream>>>(
            g_dev.d_midstate, w0_lastMerkle, w1_ntime, w2_nbits,
            g_dev.d_target_be, 0u, g_dev.d_found_nonce, g_dev.d_found_flag
        );
    int found=0; unsigned int found_nonce=0;
    cudaMemcpyAsync(&found, g_dev.d_found_flag, sizeof(found),
                    cudaMemcpyDeviceToHost, g_dev.stream);
    cudaMemcpyAsync(&found_nonce, g_dev.d_found_nonce, sizeof(found_nonce),
                    cudaMemcpyDeviceToHost, g_dev.stream);
    cudaStreamSynchronize(g_dev.stream);

    if (found) {
        block.nonce = found_nonce;
        SHA256 final_hash; double_sha256_host(&final_hash, (unsigned char*)&block, sizeof(block));

        //printf("Found Solution!!\n");
        //printf("nonce: %10u (0x%08x)\n", block.nonce, block.nonce);
        //final_hash.b from sha256.cu is big-endian bytes
        //printf("hash (big):    "); print_hex(final_hash.b,32); printf("\n");
        //printf("hash (little): "); print_hex_inverse(final_hash.b,32); printf("\n\n");

        // Convert to big-endian representation before writing to match required format
        unsigned int nonce_be =
#if defined(__GNUC__)
            __builtin_bswap32(block.nonce);
#else
            ((block.nonce & 0x000000ffu) << 24) |
            ((block.nonce & 0x0000ff00u) << 8 ) |
            ((block.nonce & 0x00ff0000u) >> 8 ) |
            ((block.nonce & 0xff000000u) >> 24);
#endif
        fprintf(fout, "%08x\n", nonce_be);
    } else {
        fprintf(stderr, "No nonce found (kernel returned without a solution)\n");
    }

    delete[] merkle_branch; delete[] raw_merkle_branch;
}

int main(int argc, char **argv){
    if(argc < 3){
        fprintf(stderr, "usage: cuda_miner <in> <out> [--debug] [--blocks N] [--threads M]\n");
        return 1;
    }
    bool debug_one = false;
    for (int i=3;i<argc;++i){
        if (!strcmp(argv[i], "--debug")){
            debug_one = true;
        } else if (!strcmp(argv[i], "--blocks") && i+1 < argc){
            g_override_blocks = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--threads") && i+1 < argc){
            g_override_threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "warning: unknown or incomplete argument '%s'\n", argv[i]);
        }
    }

    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");
    if(!fin || !fout){ fprintf(stderr,"failed to open input/output\n"); return 2; }

    int totalblock=0;
    if (fscanf(fin, "%d\n", &totalblock) != 1){ fprintf(stderr,"failed to read totalblock\n"); return 3; }
    fprintf(fout, "%d\n", totalblock);

    for(int i=0;i<totalblock;++i) solve(fin, fout, debug_one);

    fclose(fin); fclose(fout);
    destroy_device_context();
    return 0;
}
