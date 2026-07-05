/* spectre_sort.c — Spectre: adaptive, portable, multi-threaded sort core
 * for 32/64-bit signed/unsigned integers. Quill-sort's silent large-n weapon.
 *
 * Build:
 *   POSIX:  cc -O3 -shared -fPIC spectre_sort.c -o libspectre.so    -lpthread
 *   macOS:  cc -O3 -shared -fPIC spectre_sort.c -o libspectre.dylib -lpthread
 *
 * Parallel design — PERSISTENT thread pool + TILE work-stealing radix.
 * Input is cut into many small tiles. Each radix pass runs two work-stealing
 * phases over the tiles on a pool spawned ONCE and reused every pass:
 *   phase 1 (histogram): each tile builds its own [RADIX] histogram
 *   serial merge: bucket-major prefix -> each (tile,bucket) disjoint offset
 *   phase 2 (scatter): each tile scatters into its disjoint output slots
 * Work-stealing (atomic tile counter) lets fast cores grab more tiles than
 * slow ones, so heterogeneous P+E cores don't stall at the barrier. Scatter
 * regions never overlap => lock-free and stable.
 *
 * Returns 0 = OK, 1 = OOM, 2 = n too large (needs n < 2^32).
 */
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS   /* allow getenv() without MSVC deprecation noise */
#endif
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE          /* full BSD namespace: u_int, sysctl,
                                   * _SC_NPROCESSORS_ONLN, clock_gettime */
#elif !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L   /* for clock_gettime / CLOCK_MONOTONIC */
#endif
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <pthread.h>
  #include <unistd.h>
#endif
#if defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

/* ---- atomics + thread-local: real C11 on GCC/Clang/MinGW; native Win32
 * Interlocked intrinsics under MSVC (whose <stdatomic.h> support is recent and
 * flag-gated). The only hot-path atomic is a fetch-add per work tile, which
 * compiles to the same lock-xadd either way, so there is no perf difference. */
#if defined(_MSC_VER)
  #define _Thread_local __declspec(thread)
  typedef volatile long atomic_int;
  #define memory_order_relaxed 0
  static __forceinline int  spectre__load(volatile long*p){ return (int)InterlockedCompareExchange(p,0,0); }
  static __forceinline void spectre__store(volatile long*p,long v){ InterlockedExchange(p,v); }
  static __forceinline int  spectre__fadd(volatile long*p,long v){ return (int)InterlockedExchangeAdd(p,v); }
  #define atomic_load(p)                    spectre__load(p)
  #define atomic_store(p,v)                 spectre__store((p),(v))
  #define atomic_fetch_add(p,v)             spectre__fadd((p),(v))
  #define atomic_load_explicit(p,mo)        spectre__load(p)
  #define atomic_store_explicit(p,v,mo)     spectre__store((p),(v))
#else
  #include <stdatomic.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define PREFETCH(p) __builtin_prefetch((p), 1, 0)
#elif defined(_MSC_VER)
  #include <intrin.h>
  #define PREFETCH(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#else
  #define PREFETCH(p) ((void)0)
#endif

#define SPECTRE_OK 0
#define SPECTRE_ENOMEM 1
#define SPECTRE_TOO_BIG 2

/* Public symbol visibility. On Windows a DLL exports nothing unless marked
 * dllexport (MSVC/MinGW); on GCC/Clang we set default visibility so the
 * symbols survive -fvisibility=hidden builds. Consumers who include a header
 * and link the .lib don't define SPECTRE_BUILD and get dllimport on MSVC. */
#if defined(_WIN32)
  #define SPECTRE_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
  #define SPECTRE_API __attribute__((visibility("default")))
#else
  #define SPECTRE_API
#endif
#ifndef PF_DIST
#define PF_DIST 8
#endif
#define SMALL_N 96
#ifndef RB_MAX
#define RB_MAX 11               /* M2 sweep optimum: k=6 on full-range beats k=5 cache thrash */
#endif
#define RADIX (1 << RB_MAX)     /* table capacity = 4096 */
#define MAX_K 7
#ifndef AUTO_MAX
#define AUTO_MAX 16   /* auto thread ceiling: bandwidth-bound radix stops
                       * benefiting past ~16 threads on current hardware */
#endif
/* runtime digit width chosen per sort to minimize passes; <= RB_MAX */
static _Thread_local int g_rb = 10;
#ifndef RB_PREF
#define RB_PREF 10   /* preferred/default width; sweep can override at compile */
#endif
/* pick the uniform digit width in [10..RB_MAX] that minimizes pass count k;
 * on ties prefer the smaller width (smaller tables = better cache). */
static void plan_rb(int msb, int *rb_out, int *k_out){
    int best_rb=RB_PREF, best_k=(msb+RB_PREF-1)/RB_PREF;
    for(int w=10; w<=RB_MAX; w++){
        int k=(msb+w-1)/w;
        if(k<best_k){best_k=k;best_rb=w;}
    }
    if(best_k<1)best_k=1;
    if(best_k>MAX_K)best_k=MAX_K;
    *rb_out=best_rb; *k_out=best_k;
}
#define MAX_T 256
#ifndef PAR_MIN_N
#define PAR_MIN_N 262144
#endif
#ifndef TILES_PER_THREAD
#define TILES_PER_THREAD 12
#endif
#define MAX_TILES 8192

static _Thread_local const char *g_last_path = "none";
/* optional phase profiling, enabled via SPECTRE_PROFILE=1 */
static atomic_int g_profile = -1;
static double spectre__now_ms(void){
#if defined(_WIN32)
    LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart*1000.0/(double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e3 + ts.tv_nsec/1e6;
#endif
}
static _Thread_local int g_last_k = 0, g_last_threads = 0;
SPECTRE_API const char *spectre_last_path(void){ return g_last_path; }
SPECTRE_API int spectre_last_k(void){ return g_last_k; }
SPECTRE_API int spectre_last_threads(void){ return g_last_threads; }

#if defined(_WIN32)
typedef HANDLE thr_t;
typedef struct { CRITICAL_SECTION m; CONDITION_VARIABLE c; int count,total,gen; } barrier_t;
static void bar_init(barrier_t*b,int n){InitializeCriticalSection(&b->m);InitializeConditionVariable(&b->c);b->count=0;b->total=n;b->gen=0;}
static void bar_wait(barrier_t*b){EnterCriticalSection(&b->m);int g=b->gen;if(++b->count==b->total){b->gen++;b->count=0;WakeAllConditionVariable(&b->c);}else{while(g==b->gen)SleepConditionVariableCS(&b->c,&b->m,INFINITE);}LeaveCriticalSection(&b->m);}
static void bar_destroy(barrier_t*b){DeleteCriticalSection(&b->m);}
#else
typedef pthread_t thr_t;
/* pthread_barrier_* is optional in POSIX and NOT provided by macOS libpthread,
 * so we roll our own from a mutex + condition variable (available everywhere).
 * Sense-reversing generation counter makes it safely reusable. */
typedef struct { pthread_mutex_t m; pthread_cond_t c; int count,total,gen; } barrier_t;
static void bar_init(barrier_t*b,int n){pthread_mutex_init(&b->m,NULL);pthread_cond_init(&b->c,NULL);b->count=0;b->total=n;b->gen=0;}
static void bar_wait(barrier_t*b){
    pthread_mutex_lock(&b->m);
    int g=b->gen;
    if(++b->count==b->total){ b->gen++; b->count=0; pthread_cond_broadcast(&b->c); }
    else { while(g==b->gen) pthread_cond_wait(&b->c,&b->m); }
    pthread_mutex_unlock(&b->m);
}
static void bar_destroy(barrier_t*b){pthread_mutex_destroy(&b->m);pthread_cond_destroy(&b->c);}
#endif

static int cpu_total(void){
#if defined(_WIN32)
    SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
#else
    long n=sysconf(_SC_NPROCESSORS_ONLN); return n>0?(int)n:1;
#endif
}
/* Detect performance-core count across OSs. Returns P-core count, or 0 if the
 * chip is homogeneous / detection is unavailable (caller then uses total). */
static int cpu_perf(void){
#if defined(__APPLE__)
    int v=0; size_t sz=sizeof(v);
    if(sysctlbyname("hw.perflevel0.logicalcpu",&v,&sz,NULL,0)==0 && v>0) return v;
    return 0;
#elif defined(_WIN32)
    DWORD len=0;
    GetLogicalProcessorInformationEx(RelationProcessorCore,NULL,&len);
    if(len==0) return 0;
    char *buf=(char*)malloc(len);
    if(!buf) return 0;
    int perf=0, maxclass=-1;
    if(GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf,&len)){
        char *p=buf, *end=buf+len;
        /* first pass: find the highest efficiency class */
        while(p<end){ PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
            if(e->Relationship==RelationProcessorCore){ int ec=e->Processor.EfficiencyClass; if(ec>maxclass)maxclass=ec; }
            p+=e->Size; }
        p=buf;
        while(p<end){ PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
            if(e->Relationship==RelationProcessorCore && e->Processor.EfficiencyClass==maxclass) perf++;
            p+=e->Size; }
    }
    free(buf);
    return perf;
#elif defined(__linux__)
    /* ARM big.LITTLE / Intel hybrid: /sys/.../cpuN/cpu_capacity. Highest-
     * capacity cores are the P-cores. If unavailable, return 0 (homogeneous). */
    int total=cpu_total(); if(total<=1) return 0;
    long maxcap=-1; int perf=0, found=0;
    char path[128]; long caps[MAX_T]; int nc=0;
    for(int i=0;i<total && nc<MAX_T;i++){
        snprintf(path,sizeof path,"/sys/devices/system/cpu/cpu%d/cpu_capacity",i);
        FILE*f=fopen(path,"r"); if(!f) continue;
        long c=0; if(fscanf(f,"%ld",&c)==1){ caps[nc++]=c; if(c>maxcap)maxcap=c; found=1; }
        fclose(f);
    }
    if(!found) return 0;
    for(int i=0;i<nc;i++) if(caps[i]==maxcap) perf++;
    return perf;
#else
    return 0;
#endif
}
static int g_user_threads = 0;
SPECTRE_API void spectre_set_threads(int t){ g_user_threads = t<0?0:t; }
static int pick_threads(int64_t n){
    if(n<PAR_MIN_N) return 1;
    int t=g_user_threads;
    if(t<=0){
        int total=cpu_total(), p=cpu_perf();
        /* The top-digit scatter is bandwidth-bound and saturates the memory
         * controller at ~8 parallel streams regardless of core count; the
         * cache-resident partition-LSD phase keeps scaling a bit past that.
         * Measured on an 8-core M2 and a 24-core desktop, throughput plateaus
         * by ~T=8 and is flat through ~16, then droops from oversubscription.
         * So auto uses all physical cores but caps at AUTO_MAX where extra
         * threads stop helping bandwidth-bound radix. Users wanting every core
         * on a compute-heavier distribution can still override via set_threads. */
        t = total>0 ? total : (p>0?p:1);
        if(t>AUTO_MAX) t=AUTO_MAX;
        (void)p;
    }
    if(t>MAX_T)t=MAX_T;
    if(t<1)t=1;
    int cap=(int)(n/131072); if(cap<1)cap=1;
    if(t>cap)t=cap;
    return t;
}

static _Thread_local void *g_scratch=NULL;
static _Thread_local size_t g_scratch_cap=0;
static void *get_scratch(size_t bytes){
    if(g_scratch_cap<bytes){free(g_scratch);g_scratch=malloc(bytes);g_scratch_cap=g_scratch?bytes:0;}
    return g_scratch;
}
SPECTRE_API void spectre_free_scratch(void){
    free(g_scratch); g_scratch=NULL; g_scratch_cap=0;
    extern void spectre__free_part_scratch(void);
    spectre__free_part_scratch();
}

typedef struct {
    const void *src; void *dst; int sh; uint64_t mn;
    int is_first,is_last,is_u64,phase;
    int rb; uint32_t rmask; int radix;   /* runtime digit width for this pass */
    uint64_t flip;                       /* domain flip for signed types */
    int k;                               /* total digits (MSD needs low k-1) */
    void *final_dst;                     /* MSD phase-2 writes sorted output here */
    atomic_int next_tile;
} pass_ctx;
typedef struct {
    int T,ntiles;
    int64_t *tlo,*thi;
    uint32_t (*thist)[RADIX];
    uint32_t (*tbase)[RADIX];
    uint32_t *pstart;             /* [radix+1] MSD partition boundaries */
    atomic_int part_ctr;          /* work-steal counter for phase-2 partitions */
    thr_t pool[MAX_T];
    barrier_t start,done;
    atomic_int shutdown;
    pass_ctx ctx;
} pool_t;

static void hist_tile(pool_t*P,pass_ctx*c,int tile){
    uint32_t *h=P->thist[tile]; uint32_t RMASK=c->rmask; memset(h,0,(size_t)c->radix*4);
    int sh=c->sh; uint64_t mn=c->mn; int64_t lo=P->tlo[tile],hi=P->thi[tile];
    if(c->is_u64){ const uint64_t*s=(const uint64_t*)c->src; uint64_t fl=c->flip;
        if(c->is_first) for(int64_t i=lo;i<hi;i++){uint64_t f=(s[i]^fl)-mn;h[(f>>sh)&RMASK]++;}
        else            for(int64_t i=lo;i<hi;i++) h[(s[i]>>sh)&RMASK]++;
    }else{ const uint32_t*s=(const uint32_t*)c->src; uint32_t mn32=(uint32_t)mn; uint32_t fl=(uint32_t)c->flip;
        if(c->is_first) for(int64_t i=lo;i<hi;i++){uint32_t f=(s[i]^fl)-mn32;h[(f>>sh)&RMASK]++;}
        else            for(int64_t i=lo;i<hi;i++) h[(s[i]>>sh)&RMASK]++;
    }
}
static void scatter_tile(pool_t*P,pass_ctx*c,int tile){
    static _Thread_local uint32_t lc[RADIX];
    uint32_t RMASK=c->rmask;
    memcpy(lc,P->tbase[tile],(size_t)c->radix*4);
    int sh=c->sh; uint64_t mn=c->mn; int64_t lo=P->tlo[tile],hi=P->thi[tile];
    int64_t pl=hi>lo+PF_DIST?hi-PF_DIST:lo,i=lo;
    if(c->is_u64){ const uint64_t*s=(const uint64_t*)c->src; uint64_t*d=(uint64_t*)c->dst; uint64_t fl=c->flip;
        if(c->is_first&&c->is_last){
            for(;i<pl;i++){uint64_t fp=(s[i+PF_DIST]^fl)-mn;PREFETCH(&d[lc[(fp>>sh)&RMASK]]);uint64_t f=(s[i]^fl)-mn;d[lc[(f>>sh)&RMASK]++]=(f+mn)^fl;}
            for(;i<hi;i++){uint64_t f=(s[i]^fl)-mn;d[lc[(f>>sh)&RMASK]++]=(f+mn)^fl;}
        }else if(c->is_first){
            for(;i<pl;i++){uint64_t fp=(s[i+PF_DIST]^fl)-mn;PREFETCH(&d[lc[(fp>>sh)&RMASK]]);uint64_t f=(s[i]^fl)-mn;d[lc[(f>>sh)&RMASK]++]=f;}
            for(;i<hi;i++){uint64_t f=(s[i]^fl)-mn;d[lc[(f>>sh)&RMASK]++]=f;}
        }else if(c->is_last){
            for(;i<pl;i++){PREFETCH(&d[lc[(s[i+PF_DIST]>>sh)&RMASK]]);uint64_t f=s[i];d[lc[(f>>sh)&RMASK]++]=(f+mn)^fl;}
            for(;i<hi;i++){uint64_t f=s[i];d[lc[(f>>sh)&RMASK]++]=(f+mn)^fl;}
        }else{
            for(;i<pl;i++){PREFETCH(&d[lc[(s[i+PF_DIST]>>sh)&RMASK]]);uint64_t f=s[i];d[lc[(f>>sh)&RMASK]++]=f;}
            for(;i<hi;i++){uint64_t f=s[i];d[lc[(f>>sh)&RMASK]++]=f;}
        }
    }else{ const uint32_t*s=(const uint32_t*)c->src; uint32_t*d=(uint32_t*)c->dst; uint32_t mn32=(uint32_t)mn; uint32_t fl=(uint32_t)c->flip;
        if(c->is_first&&c->is_last){
            for(;i<pl;i++){uint32_t fp=(s[i+PF_DIST]^fl)-mn32;PREFETCH(&d[lc[(fp>>sh)&RMASK]]);uint32_t f=(s[i]^fl)-mn32;d[lc[(f>>sh)&RMASK]++]=(f+mn32)^fl;}
            for(;i<hi;i++){uint32_t f=(s[i]^fl)-mn32;d[lc[(f>>sh)&RMASK]++]=(f+mn32)^fl;}
        }else if(c->is_first){
            for(;i<pl;i++){uint32_t fp=(s[i+PF_DIST]^fl)-mn32;PREFETCH(&d[lc[(fp>>sh)&RMASK]]);uint32_t f=(s[i]^fl)-mn32;d[lc[(f>>sh)&RMASK]++]=f;}
            for(;i<hi;i++){uint32_t f=(s[i]^fl)-mn32;d[lc[(f>>sh)&RMASK]++]=f;}
        }else if(c->is_last){
            for(;i<pl;i++){PREFETCH(&d[lc[(s[i+PF_DIST]>>sh)&RMASK]]);uint32_t f=s[i];d[lc[(f>>sh)&RMASK]++]=(f+mn32)^fl;}
            for(;i<hi;i++){uint32_t f=s[i];d[lc[(f>>sh)&RMASK]++]=(f+mn32)^fl;}
        }else{
            for(;i<pl;i++){PREFETCH(&d[lc[(s[i+PF_DIST]>>sh)&RMASK]]);uint32_t f=s[i];d[lc[(f>>sh)&RMASK]++]=f;}
            for(;i<hi;i++){uint32_t f=s[i];d[lc[(f>>sh)&RMASK]++]=f;}
        }
    }
}
/* Sort one MSD partition (in `buf[0..n)`, normalized values sharing a top
 * digit) on its low `lowk` digits, writing the final (value+mn)^flip into
 * `dest[0..n)`. Ping-pongs between buf and dest — no extra allocation, and the
 * result lands directly in dest, so the caller needs no global copy-back.
 * `dest` is free to clobber (it's the caller's output slice for this
 * partition, not read after the partition data was moved into buf). */
static _Thread_local uint32_t *g_ser_mem=NULL;
void spectre__free_part_scratch(void){
    free(g_ser_mem); g_ser_mem=NULL;
}
static void part_lsd_u64(uint64_t*buf,uint64_t*dest,int64_t n,int lowk,int rb,uint32_t rmask,uint64_t mn,uint64_t flip){
    if(n<=0)return;
    if(n==1){dest[0]=(buf[0]+mn)^flip;return;}
    if(lowk<1){for(int64_t i=0;i<n;i++)dest[i]=(buf[i]+mn)^flip;return;}
    if(n<64){ /* insertion sort in buf, then emit restored into dest */
        for(int64_t i=1;i<n;i++){uint64_t x=buf[i];int64_t j=i-1;while(j>=0&&buf[j]>x){buf[j+1]=buf[j];j--;}buf[j+1]=x;}
        {for(int64_t i=0;i<n;i++)dest[i]=(buf[i]+mn)^flip;} return;}
    static _Thread_local uint32_t lh[MAX_K][RADIX], lo[RADIX];
    memset(lh,0,(size_t)lowk*RADIX*4);
    for(int64_t i=0;i<n;i++){uint64_t f=buf[i];for(int d=0;d<lowk;d++)lh[d][(f>>(d*rb))&rmask]++;}
    uint64_t*src=buf,*dst=dest;
    for(int p=0;p<lowk;p++){int sh=p*rb;uint32_t acc=0;for(uint32_t b=0;b<=rmask;b++){lo[b]=acc;acc+=lh[p][b];}
        int last=(p==lowk-1); int64_t i=0,lim=n>PF_DIST?n-PF_DIST:0;
        if(last){
            for(;i<lim;i++){PREFETCH(&dst[lo[(src[i+PF_DIST]>>sh)&rmask]]);uint64_t x=src[i];dst[lo[(x>>sh)&rmask]++]=(x+mn)^flip;}
            for(;i<n;i++){uint64_t x=src[i];dst[lo[(x>>sh)&rmask]++]=(x+mn)^flip;}
        }else{
            for(;i<lim;i++){PREFETCH(&dst[lo[(src[i+PF_DIST]>>sh)&rmask]]);uint64_t x=src[i];dst[lo[(x>>sh)&rmask]++]=x;}
            for(;i<n;i++){uint64_t x=src[i];dst[lo[(x>>sh)&rmask]++]=x;}
        }
        uint64_t*t=src;src=dst;dst=t;
    }
    /* result is in `src`; if that isn't dest (even lowk), one in-cache copy */
    if(src!=dest)memcpy(dest,src,n*8);
}
static void part_lsd_u32(uint32_t*buf,uint32_t*dest,int64_t n,int lowk,int rb,uint32_t rmask,uint32_t mn,uint32_t flip){
    if(n<=0)return;
    if(n==1){dest[0]=(buf[0]+mn)^flip;return;}
    if(lowk<1){for(int64_t i=0;i<n;i++)dest[i]=(buf[i]+mn)^flip;return;}
    if(n<64){
        for(int64_t i=1;i<n;i++){uint32_t x=buf[i];int64_t j=i-1;while(j>=0&&buf[j]>x){buf[j+1]=buf[j];j--;}buf[j+1]=x;}
        {for(int64_t i=0;i<n;i++)dest[i]=(buf[i]+mn)^flip;} return;}
    static _Thread_local uint32_t lh[MAX_K][RADIX], lo[RADIX];
    memset(lh,0,(size_t)lowk*RADIX*4);
    for(int64_t i=0;i<n;i++){uint32_t f=buf[i];for(int d=0;d<lowk;d++)lh[d][(f>>(d*rb))&rmask]++;}
    uint32_t*src=buf,*dst=dest;
    for(int p=0;p<lowk;p++){int sh=p*rb;uint32_t acc=0;for(uint32_t b=0;b<=rmask;b++){lo[b]=acc;acc+=lh[p][b];}
        int last=(p==lowk-1); int64_t i=0,lim=n>PF_DIST?n-PF_DIST:0;
        if(last){
            for(;i<lim;i++){PREFETCH(&dst[lo[(src[i+PF_DIST]>>sh)&rmask]]);uint32_t x=src[i];dst[lo[(x>>sh)&rmask]++]=(x+mn)^flip;}
            for(;i<n;i++){uint32_t x=src[i];dst[lo[(x>>sh)&rmask]++]=(x+mn)^flip;}
        }else{
            for(;i<lim;i++){PREFETCH(&dst[lo[(src[i+PF_DIST]>>sh)&rmask]]);uint32_t x=src[i];dst[lo[(x>>sh)&rmask]++]=x;}
            for(;i<n;i++){uint32_t x=src[i];dst[lo[(x>>sh)&rmask]++]=x;}
        }
        uint32_t*t=src;src=dst;dst=t;
    }
    if(src!=dest)memcpy(dest,src,n*4);
}
/* phase-2 worker body: work-steal partitions, LSD-sort each into final dst */
static void part_worker(pool_t*P){
    pass_ctx*c=&P->ctx; int rb=c->rb; uint32_t rmask=c->rmask; int lowk=c->k-1;
    uint64_t mn=c->mn, flip=c->flip; int radix=c->radix;
    for(;;){
        int b=atomic_fetch_add(&P->part_ctr,1); if(b>=radix)break;
        int64_t s=P->pstart[b], e=P->pstart[b+1];
        if(e<=s)continue;
        if(c->is_u64) part_lsd_u64((uint64_t*)c->src+s,(uint64_t*)c->final_dst+s, e-s, lowk, rb, rmask, mn, flip);
        else          part_lsd_u32((uint32_t*)c->src+s,(uint32_t*)c->final_dst+s, e-s, lowk, rb, rmask, (uint32_t)mn, (uint32_t)flip);
    }
}

static void *pool_worker(void *arg){
    pool_t *P=(pool_t*)arg;
    for(;;){
        bar_wait(&P->start);
        if(atomic_load(&P->shutdown)) return NULL;
        pass_ctx *c=&P->ctx;
        if(c->phase==2){ part_worker(P); }         /* work-steal partitions */
        else {                                     /* work-steal tiles */
            for(;;){int tile=atomic_fetch_add(&c->next_tile,1);if(tile>=P->ntiles)break;
                if(c->phase==0)hist_tile(P,c,tile);else scatter_tile(P,c,tile);}
        }
        bar_wait(&P->done);
    }
}

static int pool_start(pool_t*P,int T){
    P->T=T; atomic_store(&P->shutdown,0);
    bar_init(&P->start,T+1); bar_init(&P->done,T+1);
    for(int t=0;t<T;t++){
#if defined(_WIN32)
        P->pool[t]=CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)pool_worker,P,0,NULL);
        if(!P->pool[t]){ P->T=t; goto fail; }
#else
        if(pthread_create(&P->pool[t],NULL,pool_worker,P)){ P->T=t; goto fail; }
#endif
    }
    return 0;
fail:
    /* Some threads were created and are blocked on P->start (which was init'd
     * for T+1). Re-init the barriers to the actual live count so we can release
     * and join them cleanly instead of leaking/hanging. */
    if(P->T>0){
        bar_destroy(&P->start); bar_destroy(&P->done);
        bar_init(&P->start,P->T+1); bar_init(&P->done,P->T+1);
        atomic_store(&P->shutdown,1);
        bar_wait(&P->start);
        for(int t=0;t<P->T;t++){
#if defined(_WIN32)
            WaitForSingleObject(P->pool[t],INFINITE); CloseHandle(P->pool[t]);
#else
            pthread_join(P->pool[t],NULL);
#endif
        }
    }
    bar_destroy(&P->start); bar_destroy(&P->done);
    return 1;
}
static void pool_stop(pool_t*P){
    atomic_store(&P->shutdown,1);
    bar_wait(&P->start);
    for(int t=0;t<P->T;t++){
#if defined(_WIN32)
        WaitForSingleObject(P->pool[t],INFINITE);CloseHandle(P->pool[t]);
#else
        pthread_join(P->pool[t],NULL);
#endif
    }
    bar_destroy(&P->start);bar_destroy(&P->done);
}
static void run_phase(pool_t*P){ atomic_store(&P->ctx.next_tile,0); bar_wait(&P->start); bar_wait(&P->done); }

static void radix_serial(void *a,int64_t n,uint64_t mn,uint64_t mx,int is_u64,uint64_t flip){
    int elem=is_u64?8:4;
    uint64_t range=mx-mn;int msb=0;{uint64_t r=range;while(r){msb++;r>>=1;}}
    int rb,k; plan_rb(msb,&rb,&k);
    const uint32_t RMASK=(1u<<rb)-1u; const int radix=1<<rb;
    g_last_k=k;g_last_threads=1;g_rb=rb;
    /* serial path only runs for small n (< PAR_MIN_N), so the two flip passes
     * here are negligible; keeps the hot loops flip-free and already proven. */
    if(flip){ if(is_u64){uint64_t*s=(uint64_t*)a;for(int64_t i=0;i<n;i++)s[i]^=flip;}
              else{uint32_t*s=(uint32_t*)a;uint32_t fl=(uint32_t)flip;for(int64_t i=0;i<n;i++)s[i]^=fl;} }
    /* histograms in a heap block reached via a single TLS pointer — avoids
     * the ~1.6x per-access penalty of _Thread_local arrays in the hot loop. */
    if(!g_ser_mem) g_ser_mem = (uint32_t*)malloc(((size_t)MAX_K*RADIX + RADIX)*4);
    uint32_t *ser_mem = g_ser_mem;
    uint32_t (*hist)[RADIX] = (uint32_t(*)[RADIX])ser_mem;
    uint32_t *off = ser_mem + (size_t)MAX_K*RADIX;
    memset(hist,0,(size_t)k*RADIX*4);
    if(is_u64){uint64_t*s=(uint64_t*)a;for(int64_t i=0;i<n;i++){uint64_t f=s[i]-mn;for(int d=0;d<k;d++)hist[d][(f>>(d*rb))&RMASK]++;}}
    else{uint32_t*s=(uint32_t*)a;uint32_t mn32=(uint32_t)mn;for(int64_t i=0;i<n;i++){uint32_t f=s[i]-mn32;for(int d=0;d<k;d++)hist[d][(f>>(d*rb))&RMASK]++;}}
    void *buf=get_scratch((size_t)n*elem);if(!buf)return;
    if(is_u64){uint64_t*src=(uint64_t*)a,*dst=(uint64_t*)buf;
        for(int p=0;p<k;p++){int sh=p*rb;uint32_t acc=0;for(int b=0;b<radix;b++){off[b]=acc;acc+=hist[p][b];}
            int64_t lim=n>PF_DIST?n-PF_DIST:0,i=0;
            if(p==0&&k==1){for(;i<lim;i++){uint64_t fp=src[i+PF_DIST]-mn;PREFETCH(&dst[off[(fp>>sh)&RMASK]]);uint64_t f=src[i]-mn;dst[off[(f>>sh)&RMASK]++]=f+mn;}for(;i<n;i++){uint64_t f=src[i]-mn;dst[off[(f>>sh)&RMASK]++]=f+mn;}}
            else if(p==0){for(;i<lim;i++){uint64_t fp=src[i+PF_DIST]-mn;PREFETCH(&dst[off[(fp>>sh)&RMASK]]);uint64_t f=src[i]-mn;dst[off[(f>>sh)&RMASK]++]=f;}for(;i<n;i++){uint64_t f=src[i]-mn;dst[off[(f>>sh)&RMASK]++]=f;}}
            else if(p==k-1){for(;i<lim;i++){PREFETCH(&dst[off[(src[i+PF_DIST]>>sh)&RMASK]]);uint64_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f+mn;}for(;i<n;i++){uint64_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f+mn;}}
            else{for(;i<lim;i++){PREFETCH(&dst[off[(src[i+PF_DIST]>>sh)&RMASK]]);uint64_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f;}for(;i<n;i++){uint64_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f;}}
            uint64_t*t=src;src=dst;dst=t;}
        if(src!=(uint64_t*)a)memcpy(a,src,(size_t)n*8);
    }else{uint32_t*src=(uint32_t*)a,*dst=(uint32_t*)buf;uint32_t mn32=(uint32_t)mn;
        for(int p=0;p<k;p++){int sh=p*rb;uint32_t acc=0;for(int b=0;b<radix;b++){off[b]=acc;acc+=hist[p][b];}
            int64_t lim=n>PF_DIST?n-PF_DIST:0,i=0;
            if(p==0&&k==1){for(;i<lim;i++){uint32_t fp=src[i+PF_DIST]-mn32;PREFETCH(&dst[off[(fp>>sh)&RMASK]]);uint32_t f=src[i]-mn32;dst[off[(f>>sh)&RMASK]++]=f+mn32;}for(;i<n;i++){uint32_t f=src[i]-mn32;dst[off[(f>>sh)&RMASK]++]=f+mn32;}}
            else if(p==0){for(;i<lim;i++){uint32_t fp=src[i+PF_DIST]-mn32;PREFETCH(&dst[off[(fp>>sh)&RMASK]]);uint32_t f=src[i]-mn32;dst[off[(f>>sh)&RMASK]++]=f;}for(;i<n;i++){uint32_t f=src[i]-mn32;dst[off[(f>>sh)&RMASK]++]=f;}}
            else if(p==k-1){for(;i<lim;i++){PREFETCH(&dst[off[(src[i+PF_DIST]>>sh)&RMASK]]);uint32_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f+mn32;}for(;i<n;i++){uint32_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f+mn32;}}
            else{for(;i<lim;i++){PREFETCH(&dst[off[(src[i+PF_DIST]>>sh)&RMASK]]);uint32_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f;}for(;i<n;i++){uint32_t f=src[i];dst[off[(f>>sh)&RMASK]++]=f;}}
            uint32_t*t=src;src=dst;dst=t;}
        if(src!=(uint32_t*)a)memcpy(a,src,(size_t)n*4);
    }
    if(flip){ if(is_u64){uint64_t*s=(uint64_t*)a;for(int64_t i=0;i<n;i++)s[i]^=flip;}
              else{uint32_t*s=(uint32_t*)a;uint32_t fl=(uint32_t)flip;for(int64_t i=0;i<n;i++)s[i]^=fl;} }
}
static int radix_parallel(void *a,int64_t n,uint64_t mn,uint64_t mx,int is_u64,int T,uint64_t flip){
    int elem=is_u64?8:4;
    uint64_t range=mx-mn;int msb=0;{uint64_t r=range;while(r){msb++;r>>=1;}}
    int rb,k; plan_rb(msb,&rb,&k);
    const uint32_t rmask=(1u<<rb)-1u; const int radix=1<<rb;
    g_last_k=k;g_last_threads=T;g_rb=rb;
    void *buf=get_scratch((size_t)n*elem);if(!buf)return SPECTRE_ENOMEM;
    pool_t P;memset(&P,0,sizeof P);
    P.ntiles=T*TILES_PER_THREAD;if(P.ntiles>MAX_TILES)P.ntiles=MAX_TILES;if(P.ntiles<1)P.ntiles=1;
    int64_t tsz=(n+P.ntiles-1)/P.ntiles;
    P.tlo=malloc((size_t)P.ntiles*sizeof(int64_t));
    P.thi=malloc((size_t)P.ntiles*sizeof(int64_t));
    P.thist=malloc((size_t)P.ntiles*RADIX*4);
    P.tbase=malloc((size_t)P.ntiles*RADIX*4);
    P.pstart=malloc(((size_t)radix+1)*sizeof(uint32_t));
    if(!P.tlo||!P.thi||!P.thist||!P.tbase||!P.pstart){free(P.tlo);free(P.thi);free(P.thist);free(P.tbase);free(P.pstart);return SPECTRE_ENOMEM;}
    for(int i=0;i<P.ntiles;i++){P.tlo[i]=(int64_t)i*tsz;P.thi[i]=P.tlo[i]+tsz;if(P.thi[i]>n)P.thi[i]=n;if(P.tlo[i]>n)P.tlo[i]=n;}
    if(pool_start(&P,T)){free(P.tlo);free(P.thi);free(P.thist);free(P.tbase);free(P.pstart);return SPECTRE_ENOMEM;}

    /* fill shared ctx */
    P.ctx.mn=mn;P.ctx.is_u64=is_u64;P.ctx.rb=rb;P.ctx.rmask=rmask;P.ctx.radix=radix;
    P.ctx.flip=flip;P.ctx.k=k;

    if(k==1){
        /* single digit: MSD partition IS the full sort. Histogram + scatter the
         * only digit, restoring the domain on write (is_first && is_last). */
        P.ctx.src=a;P.ctx.dst=buf;P.ctx.sh=0;P.ctx.is_first=1;P.ctx.is_last=1;
        P.ctx.phase=0;run_phase(&P);
        uint32_t acc=0;
        for(int b=0;b<radix;b++){uint32_t run=acc;for(int i=0;i<P.ntiles;i++){P.tbase[i][b]=run;run+=P.thist[i][b];}acc=run;}
        P.ctx.phase=1;run_phase(&P);
        pool_stop(&P);
        memcpy(a,buf,(size_t)n*elem);
        free(P.tlo);free(P.thi);free(P.thist);free(P.tbase);free(P.pstart);
        return SPECTRE_OK;
    }

    /* ---- MSD phase 1: partition by TOP digit (bits [(k-1)*rb, k*rb)) ---- */
    int prof = atomic_load_explicit(&g_profile,memory_order_relaxed); double pt0=0,pt1=0,pt2=0,pt3=0,pt4=0;
    if(prof<0){ const char*e=getenv("SPECTRE_PROFILE"); prof=(e&&e[0]&&e[0]!='0')?1:0; atomic_store_explicit(&g_profile,prof,memory_order_relaxed); }
    if(prof)pt0=spectre__now_ms();
    int topsh=(k-1)*rb;
    P.ctx.src=a;P.ctx.dst=buf;P.ctx.sh=topsh;
    P.ctx.is_first=1;P.ctx.is_last=0;   /* first: normalize on read; NOT last: keep normalized */
    P.ctx.phase=0;run_phase(&P);        /* top-digit histogram */
    if(prof)pt1=spectre__now_ms();
    /* merge: partition boundaries + per-tile scatter bases (top digit) */
    { uint32_t acc=0;
      for(int b=0;b<radix;b++){P.pstart[b]=acc;uint32_t run=acc;
          for(int i=0;i<P.ntiles;i++){P.tbase[i][b]=run;run+=P.thist[i][b];}acc=run;}
      P.pstart[radix]=acc;
    }
    P.ctx.phase=1;run_phase(&P);        /* scatter into buf (normalized) */
    if(prof)pt2=spectre__now_ms();

    /* ---- MSD phase 2: sort each partition on low k-1 digits, landing the
     * result directly in `a` (no global copy-back, no per-thread scratch) ---- */
    P.ctx.src=buf;                       /* partitions live in buf */
    P.ctx.final_dst=a;                   /* sorted output goes straight to a */
    atomic_store(&P.part_ctr,0);
    P.ctx.phase=2;run_phase(&P);
    if(prof)pt3=spectre__now_ms();

    pool_stop(&P);
    if(prof){pt4=spectre__now_ms();
        fprintf(stderr,"[spectre profile n=%lld k=%d T=%d] tophist=%.1f topscatter=%.1f partLSD=%.1f finalcopy=%.1f total=%.1f ms\n",
            (long long)n,k,T,pt1-pt0,pt2-pt1,pt3-pt2,pt4-pt3,pt4-pt0);}
    free(P.tlo);free(P.thi);free(P.thist);free(P.tbase);free(P.pstart);
    return SPECTRE_OK;
}

#define DEFINE_SORT(NAME,T,UT,FLIP,IS_U64)                                     \
static void ins_##NAME(T *a,int64_t n){                                        \
    for(int64_t i=1;i<n;i++){T x=a[i];int64_t j=i-1;                           \
        while(j>=0&&a[j]>x){a[j+1]=a[j];j--;}a[j+1]=x;}}                       \
SPECTRE_API int spectre_sort_##NAME(T *a,int64_t n){                                        \
    if(n<2){g_last_path="trivial";return SPECTRE_OK;}                          \
    if(n<SMALL_N){ins_##NAME(a,n);g_last_path="insertion";return SPECTRE_OK;}  \
    if(n>0xFFFFFFFFll)return SPECTRE_TOO_BIG;                                  \
    UT f0=(UT)a[0]^(UT)FLIP,mn=f0,mx=f0,prev=f0;int up=1,down=1;               \
    for(int64_t i=1;i<n;i++){UT f=(UT)a[i]^(UT)FLIP;                           \
        mn=f<mn?f:mn;mx=f>mx?f:mx;up&=(f>=prev);down&=(f<=prev);prev=f;}       \
    if(up){g_last_path="presorted";return SPECTRE_OK;}                         \
    if(mn==mx){g_last_path="constant";return SPECTRE_OK;}                      \
    if(down){for(int64_t i=0,j=n-1;i<j;i++,j--){T t=a[i];a[i]=a[j];a[j]=t;}    \
        g_last_path="reversed";return SPECTRE_OK;}                             \
    UT range=mx-mn;                                                            \
    if(range<65536u){                                                         \
        uint32_t *cnt=(uint32_t*)get_scratch(((size_t)range+1)*4);            \
        if(!cnt)return SPECTRE_ENOMEM;                                        \
        memset(cnt,0,((size_t)range+1)*4);                                    \
        for(int64_t i=0;i<n;i++)cnt[(uint32_t)(((UT)a[i]^(UT)FLIP)-mn)]++;    \
        T*out=a;for(uint32_t v=0;v<=(uint32_t)range;v++){uint32_t c=cnt[v];   \
            T val=(T)((mn+(UT)v)^(UT)FLIP);while(c--)*out++=val;}             \
        g_last_path="counting";return SPECTRE_OK;}                            \
    if(!get_scratch((size_t)n*sizeof(UT)))return SPECTRE_ENOMEM;              \
    int Tn=pick_threads(n);int rc;                                            \
    if(Tn<=1){radix_serial((void*)a,n,mn,mx,IS_U64,(uint64_t)(UT)FLIP);rc=SPECTRE_OK;} \
    else rc=radix_parallel((void*)a,n,mn,mx,IS_U64,Tn,(uint64_t)(UT)FLIP);    \
    g_last_path="radix";return rc;}                                          \
SPECTRE_API int spectre_sort_desc_##NAME(T *a,int64_t n){                                 \
    int rc=spectre_sort_##NAME(a,n);                                         \
    if(rc==SPECTRE_OK) for(int64_t i=0,j=n-1;i<j;i++,j--){T t=a[i];a[i]=a[j];a[j]=t;} \
    return rc;}
DEFINE_SORT(u32,uint32_t,uint32_t,0u,0)
DEFINE_SORT(i32,int32_t,uint32_t,0x80000000u,0)
DEFINE_SORT(u64,uint64_t,uint64_t,0ull,1)
DEFINE_SORT(i64,int64_t,uint64_t,0x8000000000000000ull,1)
