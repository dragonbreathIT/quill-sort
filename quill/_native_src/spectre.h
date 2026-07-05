/* spectre.h — public API for the Spectre integer sort core.
 *
 * Spectre sorts contiguous 1-D arrays of 32/64-bit signed or unsigned integers.
 * It is adaptive (early-outs for sorted/reversed/constant/small-range data),
 * multi-threaded (portable pthreads / Win32), and portable across
 * x86-64 and ARM64 / Apple Silicon with identical results.
 *
 * All functions sort IN PLACE and return:
 *   SPECTRE_OK       (0)  success
 *   SPECTRE_ENOMEM   (1)  a working allocation failed (array left unmodified
 *                         or partially permuted but NOT corrupted past n)
 *   SPECTRE_TOO_BIG  (2)  n >= 2^32 (unsupported)
 *
 * Thread-safety: independent calls on different arrays from different threads
 * are safe to run concurrently. Telemetry (last_path/last_k/last_threads) is
 * per-thread. Do not call two sorts on the SAME array concurrently.
 */
#ifndef SPECTRE_H
#define SPECTRE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Symbol visibility. The single-file library defines SPECTRE_API itself when
 * compiled (dllexport on Windows). Consumers who include this header get
 * dllimport on MSVC so linking against spectre.lib resolves correctly. */
#if defined(_WIN32)
  #define SPECTRE_API __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
  #define SPECTRE_API __attribute__((visibility("default")))
#else
  #define SPECTRE_API
#endif

#define SPECTRE_OK 0
#define SPECTRE_ENOMEM 1
#define SPECTRE_TOO_BIG 2

/* Ascending in-place sort. */
SPECTRE_API int spectre_sort_i32(int32_t  *a, int64_t n);
SPECTRE_API int spectre_sort_u32(uint32_t *a, int64_t n);
SPECTRE_API int spectre_sort_i64(int64_t  *a, int64_t n);
SPECTRE_API int spectre_sort_u64(uint64_t *a, int64_t n);

/* Descending in-place sort. */
SPECTRE_API int spectre_sort_desc_i32(int32_t  *a, int64_t n);
SPECTRE_API int spectre_sort_desc_u32(uint32_t *a, int64_t n);
SPECTRE_API int spectre_sort_desc_i64(int64_t  *a, int64_t n);
SPECTRE_API int spectre_sort_desc_u64(uint64_t *a, int64_t n);

/* Thread control. 0 = auto (default): all physical cores. */
SPECTRE_API void spectre_set_threads(int t);

/* Telemetry for the most recent sort ON THE CALLING THREAD. */
SPECTRE_API const char *spectre_last_path(void);
SPECTRE_API int spectre_last_k(void);
SPECTRE_API int spectre_last_threads(void);

/* Release this thread's cached scratch buffers (optional). */
SPECTRE_API void spectre_free_scratch(void);

#ifdef __cplusplus
}
#endif
#endif /* SPECTRE_H */
