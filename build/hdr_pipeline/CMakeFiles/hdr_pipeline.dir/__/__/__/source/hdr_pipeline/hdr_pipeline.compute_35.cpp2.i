# 1 "hdr_pipeline.compute_35.cudafe1.gpu"
# 1 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/hdr_pipeline/CMakeFiles/hdr_pipeline.dir/__/__/__/source/hdr_pipeline//"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "hdr_pipeline.compute_35.cudafe1.gpu"
typedef char __nv_bool;
# 1482 "/usr/local/cuda-8.0/include/driver_types.h"
struct CUstream_st;
# 54 "/usr/local/cuda-8.0/include/library_types.h"
enum cudaDataType_t {
# 56 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_R_16F = 2,
# 57 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_C_16F = 6,
# 58 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_R_32F = 0,
# 59 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_C_32F = 4,
# 60 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_R_64F = 1,
# 61 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_C_64F = 5,
# 62 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_R_8I = 3,
# 63 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_C_8I = 7,
# 64 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_R_8U,
# 65 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_C_8U,
# 66 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_R_32I,
# 67 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_C_32I,
# 68 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_R_32U,
# 69 "/usr/local/cuda-8.0/include/library_types.h"
CUDA_C_32U};
# 73 "/usr/local/cuda-8.0/include/library_types.h"
enum libraryPropertyType_t {
# 75 "/usr/local/cuda-8.0/include/library_types.h"
MAJOR_VERSION,
# 76 "/usr/local/cuda-8.0/include/library_types.h"
MINOR_VERSION,
# 77 "/usr/local/cuda-8.0/include/library_types.h"
PATCH_LEVEL};
# 176 "/usr/include/libio.h" 3
enum __codecvt_result {
# 178 "/usr/include/libio.h" 3
__codecvt_ok,
# 179 "/usr/include/libio.h" 3
__codecvt_partial,
# 180 "/usr/include/libio.h" 3
__codecvt_error,
# 181 "/usr/include/libio.h" 3
__codecvt_noconv};
# 51 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
enum idtype_t {
# 52 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
P_ALL,
# 53 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
P_PID,
# 54 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
P_PGID};
# 210 "/usr/include/math.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut_E {
# 211 "/usr/include/math.h" 3
FP_NAN,
# 214 "/usr/include/math.h" 3
FP_INFINITE,
# 217 "/usr/include/math.h" 3
FP_ZERO,
# 220 "/usr/include/math.h" 3
FP_SUBNORMAL,
# 223 "/usr/include/math.h" 3
FP_NORMAL};
# 348 "/usr/include/math.h" 3
enum _LIB_VERSION_TYPE {
# 349 "/usr/include/math.h" 3
_IEEE_ = (-1),
# 350 "/usr/include/math.h" 3
_SVID_,
# 351 "/usr/include/math.h" 3
_XOPEN_,
# 352 "/usr/include/math.h" 3
_POSIX_,
# 353 "/usr/include/math.h" 3
_ISOC_};
# 47 "/usr/include/ctype.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut0_E {
# 48 "/usr/include/ctype.h" 3
_ISupper = 256,
# 49 "/usr/include/ctype.h" 3
_ISlower = 512,
# 50 "/usr/include/ctype.h" 3
_ISalpha = 1024,
# 51 "/usr/include/ctype.h" 3
_ISdigit = 2048,
# 52 "/usr/include/ctype.h" 3
_ISxdigit = 4096,
# 53 "/usr/include/ctype.h" 3
_ISspace = 8192,
# 54 "/usr/include/ctype.h" 3
_ISprint = 16384,
# 55 "/usr/include/ctype.h" 3
_ISgraph = 32768,
# 56 "/usr/include/ctype.h" 3
_ISblank = 1,
# 57 "/usr/include/ctype.h" 3
_IScntrl,
# 58 "/usr/include/ctype.h" 3
_ISpunct = 4,
# 59 "/usr/include/ctype.h" 3
_ISalnum = 8};
# 33 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut1_E {
# 34 "/usr/include/pthread.h" 3
PTHREAD_CREATE_JOINABLE,
# 36 "/usr/include/pthread.h" 3
PTHREAD_CREATE_DETACHED};
# 43 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut2_E {
# 44 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_TIMED_NP,
# 45 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_RECURSIVE_NP,
# 46 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_ERRORCHECK_NP,
# 47 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_ADAPTIVE_NP,
# 50 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_NORMAL = 0,
# 51 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_RECURSIVE,
# 52 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_ERRORCHECK,
# 53 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_DEFAULT = 0,
# 57 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_FAST_NP = 0};
# 65 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut3_E {
# 66 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_STALLED,
# 67 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_STALLED_NP = 0,
# 68 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_ROBUST,
# 69 "/usr/include/pthread.h" 3
PTHREAD_MUTEX_ROBUST_NP = 1};
# 77 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut4_E {
# 78 "/usr/include/pthread.h" 3
PTHREAD_PRIO_NONE,
# 79 "/usr/include/pthread.h" 3
PTHREAD_PRIO_INHERIT,
# 80 "/usr/include/pthread.h" 3
PTHREAD_PRIO_PROTECT};
# 115 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut5_E {
# 116 "/usr/include/pthread.h" 3
PTHREAD_RWLOCK_PREFER_READER_NP,
# 117 "/usr/include/pthread.h" 3
PTHREAD_RWLOCK_PREFER_WRITER_NP,
# 118 "/usr/include/pthread.h" 3
PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
# 119 "/usr/include/pthread.h" 3
PTHREAD_RWLOCK_DEFAULT_NP = 0};
# 156 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut6_E {
# 157 "/usr/include/pthread.h" 3
PTHREAD_INHERIT_SCHED,
# 159 "/usr/include/pthread.h" 3
PTHREAD_EXPLICIT_SCHED};
# 166 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut7_E {
# 167 "/usr/include/pthread.h" 3
PTHREAD_SCOPE_SYSTEM,
# 169 "/usr/include/pthread.h" 3
PTHREAD_SCOPE_PROCESS};
# 176 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut8_E {
# 177 "/usr/include/pthread.h" 3
PTHREAD_PROCESS_PRIVATE,
# 179 "/usr/include/pthread.h" 3
PTHREAD_PROCESS_SHARED};
# 200 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut9_E {
# 201 "/usr/include/pthread.h" 3
PTHREAD_CANCEL_ENABLE,
# 203 "/usr/include/pthread.h" 3
PTHREAD_CANCEL_DISABLE};
# 207 "/usr/include/pthread.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut10_E {
# 208 "/usr/include/pthread.h" 3
PTHREAD_CANCEL_DEFERRED,
# 210 "/usr/include/pthread.h" 3
PTHREAD_CANCEL_ASYNCHRONOUS};
# 72 "/usr/include/wctype.h" 3
enum _ZN53_INTERNAL_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97Ut11_E {
# 73 "/usr/include/wctype.h" 3
__ISwupper,
# 74 "/usr/include/wctype.h" 3
__ISwlower,
# 75 "/usr/include/wctype.h" 3
__ISwalpha,
# 76 "/usr/include/wctype.h" 3
__ISwdigit,
# 77 "/usr/include/wctype.h" 3
__ISwxdigit,
# 78 "/usr/include/wctype.h" 3
__ISwspace,
# 79 "/usr/include/wctype.h" 3
__ISwprint,
# 80 "/usr/include/wctype.h" 3
__ISwgraph,
# 81 "/usr/include/wctype.h" 3
__ISwblank,
# 82 "/usr/include/wctype.h" 3
__ISwcntrl,
# 83 "/usr/include/wctype.h" 3
__ISwpunct,
# 84 "/usr/include/wctype.h" 3
__ISwalnum,
# 86 "/usr/include/wctype.h" 3
_ISwupper = 16777216,
# 87 "/usr/include/wctype.h" 3
_ISwlower = 33554432,
# 88 "/usr/include/wctype.h" 3
_ISwalpha = 67108864,
# 89 "/usr/include/wctype.h" 3
_ISwdigit = 134217728,
# 90 "/usr/include/wctype.h" 3
_ISwxdigit = 268435456,
# 91 "/usr/include/wctype.h" 3
_ISwspace = 536870912,
# 92 "/usr/include/wctype.h" 3
_ISwprint = 1073741824,
# 93 "/usr/include/wctype.h" 3
_ISwgraph = (-2147483647-1),
# 94 "/usr/include/wctype.h" 3
_ISwblank = 65536,
# 95 "/usr/include/wctype.h" 3
_ISwcntrl = 131072,
# 96 "/usr/include/wctype.h" 3
_ISwpunct = 262144,
# 97 "/usr/include/wctype.h" 3
_ISwalnum = 524288};
# 128 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_voidIvEUt_E {
# 128 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt9__is_voidIvE7__valueE = 1};
# 148 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIbEUt_E {
# 148 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIbE7__valueE = 1};
# 155 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIcEUt_E {
# 155 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIcE7__valueE = 1};
# 162 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIaEUt_E {
# 162 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIaE7__valueE = 1};
# 169 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIhEUt_E {
# 169 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIhE7__valueE = 1};
# 177 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIwEUt_E {
# 177 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIwE7__valueE = 1};
# 186 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIDsEUt_E {
# 186 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIDsE7__valueE = 1};
# 193 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIDiEUt_E {
# 193 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIDiE7__valueE = 1};
# 201 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIsEUt_E {
# 201 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIsE7__valueE = 1};
# 208 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerItEUt_E {
# 208 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerItE7__valueE = 1};
# 215 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIiEUt_E {
# 215 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIiE7__valueE = 1};
# 222 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIjEUt_E {
# 222 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIjE7__valueE = 1};
# 229 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIlEUt_E {
# 229 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIlE7__valueE = 1};
# 236 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerImEUt_E {
# 236 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerImE7__valueE = 1};
# 243 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIxEUt_E {
# 243 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIxE7__valueE = 1};
# 250 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIyEUt_E {
# 250 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIyE7__valueE = 1};
# 268 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIfEUt_E {
# 268 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt13__is_floatingIfE7__valueE = 1};
# 275 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIdEUt_E {
# 275 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt13__is_floatingIdE7__valueE = 1};
# 282 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIeEUt_E {
# 282 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt13__is_floatingIeE7__valueE = 1};
# 358 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIcEUt_E {
# 358 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt9__is_charIcE7__valueE = 1};
# 366 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIwEUt_E {
# 366 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt9__is_charIwE7__valueE = 1};
# 381 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIcEUt_E {
# 381 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt9__is_byteIcE7__valueE = 1};
# 388 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIaEUt_E {
# 388 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt9__is_byteIaE7__valueE = 1};
# 395 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIhEUt_E {
# 395 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt9__is_byteIhE7__valueE = 1};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIeEUt_E {
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIeE7__valueE};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIdEUt_E {
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIdE7__valueE};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIfEUt_E {
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIfE7__valueE};
# 113 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIffEUt_E {
# 113 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIffE7__valueE = 1};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIfdEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIfdE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIdfEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIdfE7__valueE};
# 113 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIddEUt_E {
# 113 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIddE7__valueE = 1};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIefEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIefE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIedEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIedE7__valueE};
# 475 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZNSt6locale5_ImplE;
# 304 "/usr/include/c++/4.8/bits/locale_classes.h" 3
enum _ZNSt6localeUt_E {
# 304 "/usr/include/c++/4.8/bits/locale_classes.h" 3
_ZNSt6locale18_S_categories_sizeE = 12};
# 62 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZSt6locale;
# 51 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt13_Ios_Fmtflags {
# 53 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt12_S_boolalpha = 1,
# 54 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_dec,
# 55 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt8_S_fixed = 4,
# 56 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_hex = 8,
# 57 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt11_S_internal = 16,
# 58 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt7_S_left = 32,
# 59 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_oct = 64,
# 60 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt8_S_right = 128,
# 61 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt13_S_scientific = 256,
# 62 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt11_S_showbase = 512,
# 63 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt12_S_showpoint = 1024,
# 64 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt10_S_showpos = 2048,
# 65 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt9_S_skipws = 4096,
# 66 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt10_S_unitbuf = 8192,
# 67 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt12_S_uppercase = 16384,
# 68 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt14_S_adjustfield = 176,
# 69 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt12_S_basefield = 74,
# 70 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt13_S_floatfield = 260,
# 71 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt19_S_ios_fmtflags_end = 65536,
# 72 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt19_S_ios_fmtflags_max = 2147483647,
# 73 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt19_S_ios_fmtflags_min = (-2147483647-1)};
# 105 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt13_Ios_Openmode {
# 107 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_app = 1,
# 108 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_ate,
# 109 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_bin = 4,
# 110 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt5_S_in = 8,
# 111 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_out = 16,
# 112 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt8_S_trunc = 32,
# 113 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt19_S_ios_openmode_end = 65536,
# 114 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt19_S_ios_openmode_max = 2147483647,
# 115 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt19_S_ios_openmode_min = (-2147483647-1)};
# 147 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt12_Ios_Iostate {
# 149 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt10_S_goodbit,
# 150 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt9_S_badbit,
# 151 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt9_S_eofbit,
# 152 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt10_S_failbit = 4,
# 153 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt18_S_ios_iostate_end = 65536,
# 154 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt18_S_ios_iostate_max = 2147483647,
# 155 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt18_S_ios_iostate_min = (-2147483647-1)};
# 187 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZSt12_Ios_Seekdir {
# 189 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_beg,
# 190 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_cur,
# 191 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt6_S_end,
# 192 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt18_S_ios_seekdir_end = 65536};
# 425 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZNSt8ios_base5eventE {
# 427 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZNSt8ios_base11erase_eventE,
# 428 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZNSt8ios_base11imbue_eventE,
# 429 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZNSt8ios_base13copyfmt_eventE};
# 466 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base14_Callback_listE;
# 505 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE;
# 517 "/usr/include/c++/4.8/bits/ios_base.h" 3
enum _ZNSt8ios_baseUt_E {
# 517 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZNSt8ios_base18_S_local_word_sizeE = 8};
# 539 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base4InitE;
# 205 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZSt8ios_base;
# 1524 "/usr/include/c++/4.8/bits/locale_facets.h" 3
enum _ZNSt10__num_baseUt_E {
# 1525 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base9_S_ominusE,
# 1526 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base8_S_oplusE,
# 1527 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_oxE,
# 1528 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_oXE,
# 1529 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base10_S_odigitsE,
# 1530 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base14_S_odigits_endE = 20,
# 1531 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base11_S_oudigitsE = 20,
# 1532 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base15_S_oudigits_endE = 36,
# 1533 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_oeE = 18,
# 1534 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_oEE = 34,
# 1535 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base7_S_oendE = 36};
# 1550 "/usr/include/c++/4.8/bits/locale_facets.h" 3
enum _ZNSt10__num_baseUt0_E {
# 1551 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base9_S_iminusE,
# 1552 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base8_S_iplusE,
# 1553 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_ixE,
# 1554 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_iXE,
# 1555 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base8_S_izeroE,
# 1556 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_ieE = 18,
# 1557 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base5_S_iEE = 24,
# 1558 "/usr/include/c++/4.8/bits/locale_facets.h" 3
_ZNSt10__num_base7_S_iendE = 26};
# 113 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIiiEUt_E {
# 113 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIiiE7__valueE = 1};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIliEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIliE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameImiEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameImiE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIxiEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIxiE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIyiEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIyiE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIfiEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIfiE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIdiEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIdiE7__valueE};
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt10__are_sameIeiEUt_E {
# 106 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt10__are_sameIeiE7__valueE};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIPcEUt_E {
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIPcE7__valueE};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIPwEUt_E {
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIPwE7__valueE};
# 34 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
struct _ZN4math6vectorIfLj3EEE;
# 34 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
struct _ZN4math6vectorIfLj2EEE;
# 212 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3
typedef unsigned long size_t;
# 1 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 1 3
# 38 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 3
# 1 "/usr/local/cuda-8.0/include/host_defines.h" 1 3
# 39 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 2 3





typedef __attribute__((device_builtin_texture_type)) unsigned long long __texture_type__;
typedef __attribute__((device_builtin_surface_type)) unsigned long long __surface_type__;
# 196 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 3
extern __attribute__((device)) __attribute__((used)) void* malloc(size_t);
extern __attribute__((device)) __attribute__((used)) void free(void*);


static __attribute__((device)) void __nv_sized_free(void *p, size_t sz) { free(p); }
static __attribute__((device)) void __nv_sized_array_free(void *p, size_t sz) { free(p); }


extern __attribute__((device)) void __assertfail(
  const void *message,
  const void *file,
  unsigned int line,
  const void *function,
  size_t charsize);
# 254 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 3
static __attribute__((device)) void __assert_fail(
  const char *__assertion,
  const char *__file,
  unsigned int __line,
  const char *__function)
{
  __assertfail(
    (const void *)__assertion,
    (const void *)__file,
                  __line,
    (const void *)__function,
    sizeof(char));
}
# 284 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 3
# 1 "/usr/local/cuda-8.0/include/builtin_types.h" 1 3
# 56 "/usr/local/cuda-8.0/include/builtin_types.h" 3
# 1 "/usr/local/cuda-8.0/include/device_types.h" 1 3
# 53 "/usr/local/cuda-8.0/include/device_types.h" 3
# 1 "/usr/local/cuda-8.0/include/host_defines.h" 1 3
# 54 "/usr/local/cuda-8.0/include/device_types.h" 2 3







enum __attribute__((device_builtin)) cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};
# 57 "/usr/local/cuda-8.0/include/builtin_types.h" 2 3


# 1 "/usr/local/cuda-8.0/include/driver_types.h" 1 3
# 156 "/usr/local/cuda-8.0/include/driver_types.h" 3
enum __attribute__((device_builtin)) cudaError
{





    cudaSuccess = 0,





    cudaErrorMissingConfiguration = 1,





    cudaErrorMemoryAllocation = 2,





    cudaErrorInitializationError = 3,
# 191 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorLaunchFailure = 4,
# 200 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorPriorLaunchFailure = 5,
# 210 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorLaunchTimeout = 6,
# 219 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorLaunchOutOfResources = 7,





    cudaErrorInvalidDeviceFunction = 8,
# 234 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorInvalidConfiguration = 9,





    cudaErrorInvalidDevice = 10,





    cudaErrorInvalidValue = 11,





    cudaErrorInvalidPitchValue = 12,





    cudaErrorInvalidSymbol = 13,




    cudaErrorMapBufferObjectFailed = 14,




    cudaErrorUnmapBufferObjectFailed = 15,





    cudaErrorInvalidHostPointer = 16,





    cudaErrorInvalidDevicePointer = 17,





    cudaErrorInvalidTexture = 18,





    cudaErrorInvalidTextureBinding = 19,






    cudaErrorInvalidChannelDescriptor = 20,





    cudaErrorInvalidMemcpyDirection = 21,
# 315 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorAddressOfConstant = 22,
# 324 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorTextureFetchFailed = 23,
# 333 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorTextureNotBound = 24,
# 342 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorSynchronizationError = 25,





    cudaErrorInvalidFilterSetting = 26,





    cudaErrorInvalidNormSetting = 27,







    cudaErrorMixedDeviceExecution = 28,






    cudaErrorCudartUnloading = 29,




    cudaErrorUnknown = 30,







    cudaErrorNotYetImplemented = 31,
# 391 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorMemoryValueTooLarge = 32,






    cudaErrorInvalidResourceHandle = 33,







    cudaErrorNotReady = 34,






    cudaErrorInsufficientDriver = 35,
# 426 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorSetOnActiveProcess = 36,





    cudaErrorInvalidSurface = 37,





    cudaErrorNoDevice = 38,





    cudaErrorECCUncorrectable = 39,




    cudaErrorSharedObjectSymbolNotFound = 40,




    cudaErrorSharedObjectInitFailed = 41,





    cudaErrorUnsupportedLimit = 42,





    cudaErrorDuplicateVariableName = 43,





    cudaErrorDuplicateTextureName = 44,





    cudaErrorDuplicateSurfaceName = 45,
# 488 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorDevicesUnavailable = 46,




    cudaErrorInvalidKernelImage = 47,







    cudaErrorNoKernelImageForDevice = 48,
# 514 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorIncompatibleDriverContext = 49,






    cudaErrorPeerAccessAlreadyEnabled = 50,






    cudaErrorPeerAccessNotEnabled = 51,





    cudaErrorDeviceAlreadyInUse = 54,






    cudaErrorProfilerDisabled = 55,







    cudaErrorProfilerNotInitialized = 56,






    cudaErrorProfilerAlreadyStarted = 57,






     cudaErrorProfilerAlreadyStopped = 58,







    cudaErrorAssert = 59,






    cudaErrorTooManyPeers = 60,





    cudaErrorHostMemoryAlreadyRegistered = 61,





    cudaErrorHostMemoryNotRegistered = 62,




    cudaErrorOperatingSystem = 63,





    cudaErrorPeerAccessUnsupported = 64,






    cudaErrorLaunchMaxDepthExceeded = 65,







    cudaErrorLaunchFileScopedTex = 66,







    cudaErrorLaunchFileScopedSurf = 67,
# 639 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorSyncDepthExceeded = 68,
# 651 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorLaunchPendingCountExceeded = 69,




    cudaErrorNotPermitted = 70,





    cudaErrorNotSupported = 71,
# 671 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorHardwareStackError = 72,







    cudaErrorIllegalInstruction = 73,
# 688 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorMisalignedAddress = 74,
# 699 "/usr/local/cuda-8.0/include/driver_types.h" 3
    cudaErrorInvalidAddressSpace = 75,







    cudaErrorInvalidPc = 76,







    cudaErrorIllegalAddress = 77,





    cudaErrorInvalidPtx = 78,




    cudaErrorInvalidGraphicsContext = 79,





    cudaErrorNvlinkUncorrectable = 80,




    cudaErrorStartupFailure = 0x7f,







    cudaErrorApiFailureBase = 10000
};




enum __attribute__((device_builtin)) cudaChannelFormatKind
{
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3
};




struct __attribute__((device_builtin)) cudaChannelFormatDesc
{
    int x;
    int y;
    int z;
    int w;
    enum cudaChannelFormatKind f;
};




typedef struct cudaArray *cudaArray_t;




typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;




typedef struct cudaMipmappedArray *cudaMipmappedArray_t;




typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;




enum __attribute__((device_builtin)) cudaMemoryType
{
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2
};




enum __attribute__((device_builtin)) cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};






struct __attribute__((device_builtin)) cudaPitchedPtr
{
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};






struct __attribute__((device_builtin)) cudaExtent
{
    size_t width;
    size_t height;
    size_t depth;
};






struct __attribute__((device_builtin)) cudaPos
{
    size_t x;
    size_t y;
    size_t z;
};




struct __attribute__((device_builtin)) cudaMemcpy3DParms
{
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;

    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;

    struct cudaExtent extent;
    enum cudaMemcpyKind kind;
};




struct __attribute__((device_builtin)) cudaMemcpy3DPeerParms
{
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    int srcDevice;

    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    int dstDevice;

    struct cudaExtent extent;
};




struct cudaGraphicsResource;




enum __attribute__((device_builtin)) cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone = 0,
    cudaGraphicsRegisterFlagsReadOnly = 1,
    cudaGraphicsRegisterFlagsWriteDiscard = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    cudaGraphicsRegisterFlagsTextureGather = 8
};




enum __attribute__((device_builtin)) cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone = 0,
    cudaGraphicsMapFlagsReadOnly = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2
};




enum __attribute__((device_builtin)) cudaGraphicsCubeFace
{
    cudaGraphicsCubeFacePositiveX = 0x00,
    cudaGraphicsCubeFaceNegativeX = 0x01,
    cudaGraphicsCubeFacePositiveY = 0x02,
    cudaGraphicsCubeFaceNegativeY = 0x03,
    cudaGraphicsCubeFacePositiveZ = 0x04,
    cudaGraphicsCubeFaceNegativeZ = 0x05
};




enum __attribute__((device_builtin)) cudaResourceType
{
    cudaResourceTypeArray = 0x00,
    cudaResourceTypeMipmappedArray = 0x01,
    cudaResourceTypeLinear = 0x02,
    cudaResourceTypePitch2D = 0x03
};




enum __attribute__((device_builtin)) cudaResourceViewFormat
{
    cudaResViewFormatNone = 0x00,
    cudaResViewFormatUnsignedChar1 = 0x01,
    cudaResViewFormatUnsignedChar2 = 0x02,
    cudaResViewFormatUnsignedChar4 = 0x03,
    cudaResViewFormatSignedChar1 = 0x04,
    cudaResViewFormatSignedChar2 = 0x05,
    cudaResViewFormatSignedChar4 = 0x06,
    cudaResViewFormatUnsignedShort1 = 0x07,
    cudaResViewFormatUnsignedShort2 = 0x08,
    cudaResViewFormatUnsignedShort4 = 0x09,
    cudaResViewFormatSignedShort1 = 0x0a,
    cudaResViewFormatSignedShort2 = 0x0b,
    cudaResViewFormatSignedShort4 = 0x0c,
    cudaResViewFormatUnsignedInt1 = 0x0d,
    cudaResViewFormatUnsignedInt2 = 0x0e,
    cudaResViewFormatUnsignedInt4 = 0x0f,
    cudaResViewFormatSignedInt1 = 0x10,
    cudaResViewFormatSignedInt2 = 0x11,
    cudaResViewFormatSignedInt4 = 0x12,
    cudaResViewFormatHalf1 = 0x13,
    cudaResViewFormatHalf2 = 0x14,
    cudaResViewFormatHalf4 = 0x15,
    cudaResViewFormatFloat1 = 0x16,
    cudaResViewFormatFloat2 = 0x17,
    cudaResViewFormatFloat4 = 0x18,
    cudaResViewFormatUnsignedBlockCompressed1 = 0x19,
    cudaResViewFormatUnsignedBlockCompressed2 = 0x1a,
    cudaResViewFormatUnsignedBlockCompressed3 = 0x1b,
    cudaResViewFormatUnsignedBlockCompressed4 = 0x1c,
    cudaResViewFormatSignedBlockCompressed4 = 0x1d,
    cudaResViewFormatUnsignedBlockCompressed5 = 0x1e,
    cudaResViewFormatSignedBlockCompressed5 = 0x1f,
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20,
    cudaResViewFormatSignedBlockCompressed6H = 0x21,
    cudaResViewFormatUnsignedBlockCompressed7 = 0x22
};




struct __attribute__((device_builtin)) cudaResourceDesc {
 enum cudaResourceType resType;

 union {
  struct {
   cudaArray_t array;
  } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
  struct {
   void *devPtr;
   struct cudaChannelFormatDesc desc;
   size_t sizeInBytes;
  } linear;
  struct {
   void *devPtr;
   struct cudaChannelFormatDesc desc;
   size_t width;
   size_t height;
   size_t pitchInBytes;
  } pitch2D;
 } res;
};




struct __attribute__((device_builtin)) cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
};




struct __attribute__((device_builtin)) cudaPointerAttributes
{




    enum cudaMemoryType memoryType;
# 1044 "/usr/local/cuda-8.0/include/driver_types.h" 3
    int device;





    void *devicePointer;





    void *hostPointer;




    int isManaged;
};




struct __attribute__((device_builtin)) cudaFuncAttributes
{





   size_t sharedSizeBytes;





   size_t constSizeBytes;




   size_t localSizeBytes;






   int maxThreadsPerBlock;




   int numRegs;






   int ptxVersion;






   int binaryVersion;





   int cacheModeCA;
};




enum __attribute__((device_builtin)) cudaFuncCache
{
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3
};





enum __attribute__((device_builtin)) cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2
};




enum __attribute__((device_builtin)) cudaComputeMode
{
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3
};




enum __attribute__((device_builtin)) cudaLimit
{
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02,
    cudaLimitDevRuntimeSyncDepth = 0x03,
    cudaLimitDevRuntimePendingLaunchCount = 0x04
};




enum __attribute__((device_builtin)) cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6
};




enum __attribute__((device_builtin)) cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4
};




enum __attribute__((device_builtin)) cudaOutputMode
{
    cudaKeyValuePair = 0x00,
    cudaCSV = 0x01
};




enum __attribute__((device_builtin)) cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91
};





enum __attribute__((device_builtin)) cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank = 1,
    cudaDevP2PAttrAccessSupported = 2,
    cudaDevP2PAttrNativeAtomicSupported = 3
};



struct __attribute__((device_builtin)) cudaDeviceProp
{
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
};
# 1455 "/usr/local/cuda-8.0/include/driver_types.h" 3
typedef __attribute__((device_builtin)) struct __attribute__((device_builtin)) cudaIpcEventHandle_st
{
    char reserved[64];
}cudaIpcEventHandle_t;




typedef __attribute__((device_builtin)) struct __attribute__((device_builtin)) cudaIpcMemHandle_st
{
    char reserved[64];
}cudaIpcMemHandle_t;
# 1477 "/usr/local/cuda-8.0/include/driver_types.h" 3
typedef __attribute__((device_builtin)) enum cudaError cudaError_t;




typedef __attribute__((device_builtin)) struct CUstream_st *cudaStream_t;




typedef __attribute__((device_builtin)) struct CUevent_st *cudaEvent_t;




typedef __attribute__((device_builtin)) struct cudaGraphicsResource *cudaGraphicsResource_t;




typedef __attribute__((device_builtin)) struct CUuuid_st cudaUUID_t;




typedef __attribute__((device_builtin)) enum cudaOutputMode cudaOutputMode_t;
# 60 "/usr/local/cuda-8.0/include/builtin_types.h" 2 3


# 1 "/usr/local/cuda-8.0/include/surface_types.h" 1 3
# 84 "/usr/local/cuda-8.0/include/surface_types.h" 3
enum __attribute__((device_builtin)) cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp = 1,
    cudaBoundaryModeTrap = 2
};




enum __attribute__((device_builtin)) cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,
    cudaFormatModeAuto = 1
};




struct __attribute__((device_builtin)) surfaceReference
{



    struct cudaChannelFormatDesc channelDesc;
};




typedef __attribute__((device_builtin)) unsigned long long cudaSurfaceObject_t;
# 63 "/usr/local/cuda-8.0/include/builtin_types.h" 2 3
# 1 "/usr/local/cuda-8.0/include/texture_types.h" 1 3
# 84 "/usr/local/cuda-8.0/include/texture_types.h" 3
enum __attribute__((device_builtin)) cudaTextureAddressMode
{
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3
};




enum __attribute__((device_builtin)) cudaTextureFilterMode
{
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1
};




enum __attribute__((device_builtin)) cudaTextureReadMode
{
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1
};




struct __attribute__((device_builtin)) textureReference
{



    int normalized;



    enum cudaTextureFilterMode filterMode;



    enum cudaTextureAddressMode addressMode[3];



    struct cudaChannelFormatDesc channelDesc;



    int sRGB;



    unsigned int maxAnisotropy;



    enum cudaTextureFilterMode mipmapFilterMode;



    float mipmapLevelBias;



    float minMipmapLevelClamp;



    float maxMipmapLevelClamp;
    int __cudaReserved[15];
};




struct __attribute__((device_builtin)) cudaTextureDesc
{



    enum cudaTextureAddressMode addressMode[3];



    enum cudaTextureFilterMode filterMode;



    enum cudaTextureReadMode readMode;



    int sRGB;



    float borderColor[4];



    int normalizedCoords;



    unsigned int maxAnisotropy;



    enum cudaTextureFilterMode mipmapFilterMode;



    float mipmapLevelBias;



    float minMipmapLevelClamp;



    float maxMipmapLevelClamp;
};




typedef __attribute__((device_builtin)) unsigned long long cudaTextureObject_t;
# 64 "/usr/local/cuda-8.0/include/builtin_types.h" 2 3
# 1 "/usr/local/cuda-8.0/include/vector_types.h" 1 3
# 61 "/usr/local/cuda-8.0/include/vector_types.h" 3
# 1 "/usr/local/cuda-8.0/include/builtin_types.h" 1 3
# 64 "/usr/local/cuda-8.0/include/builtin_types.h" 3
# 1 "/usr/local/cuda-8.0/include/vector_types.h" 1 3
# 64 "/usr/local/cuda-8.0/include/builtin_types.h" 2 3
# 62 "/usr/local/cuda-8.0/include/vector_types.h" 2 3
# 98 "/usr/local/cuda-8.0/include/vector_types.h" 3
struct __attribute__((device_builtin)) char1
{
    signed char x;
};

struct __attribute__((device_builtin)) uchar1
{
    unsigned char x;
};


struct __attribute__((device_builtin)) __attribute__((aligned(2))) char2
{
    signed char x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2))) uchar2
{
    unsigned char x, y;
};

struct __attribute__((device_builtin)) char3
{
    signed char x, y, z;
};

struct __attribute__((device_builtin)) uchar3
{
    unsigned char x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) char4
{
    signed char x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) uchar4
{
    unsigned char x, y, z, w;
};

struct __attribute__((device_builtin)) short1
{
    short x;
};

struct __attribute__((device_builtin)) ushort1
{
    unsigned short x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) short2
{
    short x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) ushort2
{
    unsigned short x, y;
};

struct __attribute__((device_builtin)) short3
{
    short x, y, z;
};

struct __attribute__((device_builtin)) ushort3
{
    unsigned short x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) short4 { short x; short y; short z; short w; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct __attribute__((device_builtin)) int1
{
    int x;
};

struct __attribute__((device_builtin)) uint1
{
    unsigned int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) int2 { int x; int y; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) uint2 { unsigned int x; unsigned int y; };

struct __attribute__((device_builtin)) int3
{
    int x, y, z;
};

struct __attribute__((device_builtin)) uint3
{
    unsigned int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) int4
{
    int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) uint4
{
    unsigned int x, y, z, w;
};

struct __attribute__((device_builtin)) long1
{
    long int x;
};

struct __attribute__((device_builtin)) ulong1
{
    unsigned long x;
};






struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(long int)))) long2
{
    long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(unsigned long int)))) ulong2
{
    unsigned long int x, y;
};



struct __attribute__((device_builtin)) long3
{
    long int x, y, z;
};

struct __attribute__((device_builtin)) ulong3
{
    unsigned long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) long4
{
    long int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulong4
{
    unsigned long int x, y, z, w;
};

struct __attribute__((device_builtin)) float1
{
    float x;
};
# 274 "/usr/local/cuda-8.0/include/vector_types.h" 3
struct __attribute__((device_builtin)) __attribute__((aligned(8))) float2 { float x; float y; };




struct __attribute__((device_builtin)) float3
{
    float x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) float4
{
    float x, y, z, w;
};

struct __attribute__((device_builtin)) longlong1
{
    long long int x;
};

struct __attribute__((device_builtin)) ulonglong1
{
    unsigned long long int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong2
{
    long long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong2
{
    unsigned long long int x, y;
};

struct __attribute__((device_builtin)) longlong3
{
    long long int x, y, z;
};

struct __attribute__((device_builtin)) ulonglong3
{
    unsigned long long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong4
{
    long long int x, y, z ,w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __attribute__((device_builtin)) double1
{
    double x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double2
{
    double x, y;
};

struct __attribute__((device_builtin)) double3
{
    double x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double4
{
    double x, y, z, w;
};
# 362 "/usr/local/cuda-8.0/include/vector_types.h" 3
typedef __attribute__((device_builtin)) struct char1 char1;
typedef __attribute__((device_builtin)) struct uchar1 uchar1;
typedef __attribute__((device_builtin)) struct char2 char2;
typedef __attribute__((device_builtin)) struct uchar2 uchar2;
typedef __attribute__((device_builtin)) struct char3 char3;
typedef __attribute__((device_builtin)) struct uchar3 uchar3;
typedef __attribute__((device_builtin)) struct char4 char4;
typedef __attribute__((device_builtin)) struct uchar4 uchar4;
typedef __attribute__((device_builtin)) struct short1 short1;
typedef __attribute__((device_builtin)) struct ushort1 ushort1;
typedef __attribute__((device_builtin)) struct short2 short2;
typedef __attribute__((device_builtin)) struct ushort2 ushort2;
typedef __attribute__((device_builtin)) struct short3 short3;
typedef __attribute__((device_builtin)) struct ushort3 ushort3;
typedef __attribute__((device_builtin)) struct short4 short4;
typedef __attribute__((device_builtin)) struct ushort4 ushort4;
typedef __attribute__((device_builtin)) struct int1 int1;
typedef __attribute__((device_builtin)) struct uint1 uint1;
typedef __attribute__((device_builtin)) struct int2 int2;
typedef __attribute__((device_builtin)) struct uint2 uint2;
typedef __attribute__((device_builtin)) struct int3 int3;
typedef __attribute__((device_builtin)) struct uint3 uint3;
typedef __attribute__((device_builtin)) struct int4 int4;
typedef __attribute__((device_builtin)) struct uint4 uint4;
typedef __attribute__((device_builtin)) struct long1 long1;
typedef __attribute__((device_builtin)) struct ulong1 ulong1;
typedef __attribute__((device_builtin)) struct long2 long2;
typedef __attribute__((device_builtin)) struct ulong2 ulong2;
typedef __attribute__((device_builtin)) struct long3 long3;
typedef __attribute__((device_builtin)) struct ulong3 ulong3;
typedef __attribute__((device_builtin)) struct long4 long4;
typedef __attribute__((device_builtin)) struct ulong4 ulong4;
typedef __attribute__((device_builtin)) struct float1 float1;
typedef __attribute__((device_builtin)) struct float2 float2;
typedef __attribute__((device_builtin)) struct float3 float3;
typedef __attribute__((device_builtin)) struct float4 float4;
typedef __attribute__((device_builtin)) struct longlong1 longlong1;
typedef __attribute__((device_builtin)) struct ulonglong1 ulonglong1;
typedef __attribute__((device_builtin)) struct longlong2 longlong2;
typedef __attribute__((device_builtin)) struct ulonglong2 ulonglong2;
typedef __attribute__((device_builtin)) struct longlong3 longlong3;
typedef __attribute__((device_builtin)) struct ulonglong3 ulonglong3;
typedef __attribute__((device_builtin)) struct longlong4 longlong4;
typedef __attribute__((device_builtin)) struct ulonglong4 ulonglong4;
typedef __attribute__((device_builtin)) struct double1 double1;
typedef __attribute__((device_builtin)) struct double2 double2;
typedef __attribute__((device_builtin)) struct double3 double3;
typedef __attribute__((device_builtin)) struct double4 double4;







struct __attribute__((device_builtin)) dim3
{
    unsigned int x, y, z;





};

typedef __attribute__((device_builtin)) struct dim3 dim3;
# 64 "/usr/local/cuda-8.0/include/builtin_types.h" 2 3
# 285 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 2 3
# 1 "/usr/local/cuda-8.0/include/device_launch_parameters.h" 1 3
# 71 "/usr/local/cuda-8.0/include/device_launch_parameters.h" 3
uint3 __attribute__((device_builtin)) extern const threadIdx;
uint3 __attribute__((device_builtin)) extern const blockIdx;
dim3 __attribute__((device_builtin)) extern const blockDim;
dim3 __attribute__((device_builtin)) extern const gridDim;
int __attribute__((device_builtin)) extern const warpSize;
# 286 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 2 3
# 1 "/usr/local/cuda-8.0/include/crt/storage_class.h" 1 3
# 286 "/usr/local/cuda-8.0/include/crt/device_runtime.h" 2 3
# 214 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 2 3
# 187 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/c++config.h" 3
typedef long _ZSt9ptrdiff_t;
# 98 "/usr/include/c++/4.8/bits/postypes.h" 3
typedef _ZSt9ptrdiff_t _ZSt10streamsize;
# 62 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZSt6locale {
# 280 "/usr/include/c++/4.8/bits/locale_classes.h" 3
struct _ZNSt6locale5_ImplE *_M_impl;};
# 261 "/usr/include/c++/4.8/bits/ios_base.h" 3
typedef enum _ZSt13_Ios_Fmtflags _ZNSt8ios_base8fmtflagsE;
# 336 "/usr/include/c++/4.8/bits/ios_base.h" 3
typedef enum _ZSt12_Ios_Iostate _ZNSt8ios_base7iostateE;
# 505 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE {
# 507 "/usr/include/c++/4.8/bits/ios_base.h" 3
void *_M_pword;
# 508 "/usr/include/c++/4.8/bits/ios_base.h" 3
long _M_iword;};
# 539 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base4InitE {char __nv_no_debug_dummy_end_padding_0;};
# 205 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZSt8ios_base { const long *__vptr;
# 458 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt10streamsize _M_precision;
# 459 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZSt10streamsize _M_width;
# 460 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZNSt8ios_base8fmtflagsE _M_flags;
# 461 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZNSt8ios_base7iostateE _M_exception;
# 462 "/usr/include/c++/4.8/bits/ios_base.h" 3
_ZNSt8ios_base7iostateE _M_streambuf_state;
# 496 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base14_Callback_listE *_M_callbacks;
# 513 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE _M_word_zero;
# 518 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE _M_local_word[8];
# 521 "/usr/include/c++/4.8/bits/ios_base.h" 3
int _M_word_size;
# 522 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZNSt8ios_base6_WordsE *_M_word;
# 528 "/usr/include/c++/4.8/bits/ios_base.h" 3
struct _ZSt6locale _M_ios_locale;};
# 989 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
typedef struct _ZN4math6vectorIfLj2EEE _ZN4math6float2E;
# 990 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
typedef struct _ZN4math6vectorIfLj3EEE _ZN4math6float3E;
# 34 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
struct _ZN4math6vectorIfLj3EEE {
# 301 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float x;
# 302 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float y;
# 303 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float z;};
# 34 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
struct _ZN4math6vectorIfLj2EEE {
# 43 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float x;
# 44 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float y;};
# 8550 "/usr/local/cuda-8.0/include/math_functions.h"
 __attribute__((device_builtin)) extern __attribute__((device)) __attribute__((__nothrow__)) float powf(float, float);
# 196 "/usr/local/cuda-8.0/include/device_functions.h"
 __attribute__((device_builtin)) extern __attribute__((device)) float __saturatef(float);
# 7 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z9luminanceRKN4math6vectorIfLj3EEE(const _ZN4math6float3E *);
# 12 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z17Uncharted2Tonemapf(float);
# 24 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z7tonemapff(float, float);
# 30 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) _ZN4math6float3E _Z7tonemapRKN4math6vectorIfLj3EEEf(const _ZN4math6float3E *, float);
# 35 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) unsigned char _Z9toLinear8f(float);
# 40 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) unsigned char _Z7toSRGB8f(float);
# 45 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z11fromLinear8h(unsigned char);
# 50 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z9fromSRGB8h(unsigned char);
# 62 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__attribute__((global)) extern void _Z16luminance_kernelPfPKfjj(float *, const float *, unsigned, unsigned);
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__attribute__((global)) extern void _Z17downsample_kernelPfPKfjj(float *, const float *, unsigned, unsigned);
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__attribute__((global)) extern void _Z14tonemap_kernelP6uchar4S0_PKfjjff(struct uchar4 *, struct uchar4 *, const float *, unsigned, unsigned, float, float);
# 312 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj3EEC1Efff(struct _ZN4math6vectorIfLj3EEE *const, float, float, float);
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj3EEC2Efff(struct _ZN4math6vectorIfLj3EEE *const, float, float, float);
# 53 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj2EEC1Eff(struct _ZN4math6vectorIfLj2EEE *const, float, float);
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj2EEC2Eff(struct _ZN4math6vectorIfLj2EEE *const, float, float);
# 1 "/usr/local/cuda-8.0/include/common_functions.h" 1
# 249 "/usr/local/cuda-8.0/include/common_functions.h"
# 1 "/usr/local/cuda-8.0/include/math_functions.h" 1 3
# 10327 "/usr/local/cuda-8.0/include/math_functions.h" 3
# 1 "/usr/local/cuda-8.0/include/math_functions.hpp" 1 3
# 10328 "/usr/local/cuda-8.0/include/math_functions.h" 2 3



# 1 "/usr/local/cuda-8.0/include/math_functions_dbl_ptx3.h" 1 3
# 270 "/usr/local/cuda-8.0/include/math_functions_dbl_ptx3.h" 3
# 1 "/usr/local/cuda-8.0/include/math_functions_dbl_ptx3.hpp" 1 3
# 271 "/usr/local/cuda-8.0/include/math_functions_dbl_ptx3.h" 2 3
# 10332 "/usr/local/cuda-8.0/include/math_functions.h" 2 3
# 250 "/usr/local/cuda-8.0/include/common_functions.h" 2
# 56 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h" 2
# 7 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z9luminanceRKN4math6vectorIfLj3EEE(
# 7 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
const _ZN4math6float3E *color){
# 8 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{ _ZN4math6float3E __T20;
 const struct _ZN4math6vectorIfLj3EEE *__T21;
# 9 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return (__T21 = ((_ZN4math6vectorIfLj3EEC1Efff((&__T20), (0.2125999928F), (0.715200007F), (0.07220000029F))) , (((const struct _ZN4math6vectorIfLj3EEE *)&__T20)))) , ((((color->x) * (__T21->x)) + ((color->y) * (__T21->y))) + ((color->z) * (__T21->z)));
# 10 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 12 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z17Uncharted2Tonemapf(
# 12 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
float x){
# 13 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{
# 15 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
 float __cuda_local_var_38554_18_const_A;
# 16 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
 float __cuda_local_var_38555_18_const_B;
# 17 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
 float __cuda_local_var_38556_18_const_C;
# 18 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
 float __cuda_local_var_38557_18_const_D;
# 19 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
 float __cuda_local_var_38558_18_const_E;
# 20 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
 float __cuda_local_var_38559_18_const_F;
# 15 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__cuda_local_var_38554_18_const_A = (0.150000006F);
# 16 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__cuda_local_var_38555_18_const_B = (0.5F);
# 17 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__cuda_local_var_38556_18_const_C = (0.1000000015F);
# 18 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__cuda_local_var_38557_18_const_D = (0.200000003F);
# 19 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__cuda_local_var_38558_18_const_E = (0.01999999955F);
# 20 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__cuda_local_var_38559_18_const_F = (0.3000000119F);
# 21 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return ( fdividef(((x * (((0.150000006F) * x) + (0.05000000075F))) + (0.00400000019F)) , ((x * (((0.150000006F) * x) + (0.5F))) + (0.06000000238F)))) - (0.06666666269F);
# 22 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 24 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z7tonemapff(
# 24 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
float c,
# 24 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
float exposure){
# 25 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{
# 26 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
 float __cuda_local_var_38565_18_const_W;
# 26 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__cuda_local_var_38565_18_const_W = (11.19999981F);
# 27 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return fdividef((_Z17Uncharted2Tonemapf(((c * exposure) * (2.0F)))) , (_Z17Uncharted2Tonemapf((11.19999981F))));
# 28 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 30 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) _ZN4math6float3E _Z7tonemapRKN4math6vectorIfLj3EEEf(
# 30 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
const _ZN4math6float3E *c,
# 30 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
float exposure){
# 31 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{ struct _ZN4math6vectorIfLj3EEE __T22;
 float __T23;
 float __T24;
 float __T25;
# 31 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{ __T23 = (_Z7tonemapff((c->x), exposure)); __T24 = (_Z7tonemapff((c->y), exposure)); __T25 = (_Z7tonemapff((c->z), exposure));
# 32 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
_ZN4math6vectorIfLj3EEC1Efff((&__T22), __T23, __T24, __T25);
# 32 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return __T22; }
# 33 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 35 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) unsigned char _Z9toLinear8f(
# 35 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
float c){
# 36 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{
# 37 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return (unsigned char)__float2uint_rz((float)(((__saturatef(c)) * (255.0F))));
# 38 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 40 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) unsigned char _Z7toSRGB8f(
# 40 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
float c){
# 41 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{
# 42 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return _Z9toLinear8f((powf(c, (0.4545454383F))));
# 43 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 45 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z11fromLinear8h(
# 45 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
unsigned char c){
# 46 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{
# 47 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return ((float)c) * (0.003921568859F);
# 48 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 50 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
static __attribute__((device)) float _Z9fromSRGB8h(
# 50 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
unsigned char c){
# 51 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
{
# 52 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
return powf((_Z11fromLinear8h(c)), (2.200000048F));
# 53 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
}}
# 62 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__attribute__((global)) void _Z16luminance_kernelPfPKfjj(
# 62 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
float *dest,
# 62 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
const float *input,
# 62 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
unsigned width,
# 62 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
unsigned height){
# 63 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{
# 65 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 unsigned __cuda_local_var_38652_15_non_const_x;
# 66 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 unsigned __cuda_local_var_38653_15_non_const_y;
# 65 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38652_15_non_const_x = (((blockIdx.x) * (blockDim.x)) + (threadIdx.x));
# 66 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38653_15_non_const_y = (((blockIdx.y) * (blockDim.y)) + (threadIdx.y));
# 71 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
if ((__cuda_local_var_38652_15_non_const_x < width) && (__cuda_local_var_38653_15_non_const_y < height))
# 71 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{
# 72 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 const float *__cuda_local_var_38659_16_non_const_input_pixel;
# 74 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 float __cuda_local_var_38661_9_non_const_lum;
# 72 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38659_16_non_const_input_pixel = (input + (3U * ((width * __cuda_local_var_38653_15_non_const_y) + __cuda_local_var_38652_15_non_const_x)));
# 74 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38661_9_non_const_lum = ((((0.2099999934F) * (__cuda_local_var_38659_16_non_const_input_pixel[0])) + ((0.7200000286F) * (__cuda_local_var_38659_16_non_const_input_pixel[1]))) + ((0.0700000003F) * (__cuda_local_var_38659_16_non_const_input_pixel[2])));
# 76 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
(dest[((width * __cuda_local_var_38653_15_non_const_y) + __cuda_local_var_38652_15_non_const_x)]) = __cuda_local_var_38661_9_non_const_lum;
# 77 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}
# 78 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}}
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__attribute__((global)) void _Z17downsample_kernelPfPKfjj(
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
float *dest,
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
const float *input,
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
unsigned width,
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
unsigned height){
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{
# 100 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 unsigned __cuda_local_var_38687_16_non_const_x;
# 101 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 unsigned __cuda_local_var_38688_16_non_const_y;
# 103 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 int __cuda_local_var_38690_7_non_const_F;
# 108 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 float __cuda_local_var_38695_9_non_const_sum;
# 100 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38687_16_non_const_x = (((blockIdx.x) * (blockDim.x)) + (threadIdx.x));
# 101 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38688_16_non_const_y = (((blockIdx.y) * (blockDim.y)) + (threadIdx.y));
# 103 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38690_7_non_const_F = 2;
# 105 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
if ((__cuda_local_var_38687_16_non_const_x >= (width / ((unsigned)__cuda_local_var_38690_7_non_const_F))) || (__cuda_local_var_38688_16_non_const_y >= (height / ((unsigned)__cuda_local_var_38690_7_non_const_F)))) {
# 106 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
return; }
# 108 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38695_9_non_const_sum = (0.0F); {
# 111 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 int j;
# 111 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
j = 0;
# 111 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
for (; (j < __cuda_local_var_38690_7_non_const_F); j++)
# 111 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{
# 111 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{
# 112 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 int i;
# 112 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
i = 0;
# 112 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
for (; (i < __cuda_local_var_38690_7_non_const_F); i++)
# 112 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{
# 114 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38695_9_non_const_sum += (input[(((((__cuda_local_var_38688_16_non_const_y * ((unsigned)__cuda_local_var_38690_7_non_const_F)) + ((unsigned)j)) * width) + (__cuda_local_var_38687_16_non_const_x * ((unsigned)__cuda_local_var_38690_7_non_const_F))) + ((unsigned)i))]);
# 115 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
} }
# 116 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
} }
# 117 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
(dest[(((__cuda_local_var_38688_16_non_const_y * width) / ((unsigned)__cuda_local_var_38690_7_non_const_F)) + __cuda_local_var_38687_16_non_const_x)]) = ( fdividef(__cuda_local_var_38695_9_non_const_sum , ((float)(__cuda_local_var_38690_7_non_const_F * __cuda_local_var_38690_7_non_const_F))));
# 124 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}}
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__attribute__((global)) void _Z14tonemap_kernelP6uchar4S0_PKfjjff(
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
struct uchar4 *tonemapped,
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
struct uchar4 *brightpass,
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
const float *src,
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
unsigned width,
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
unsigned height,
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
float exposure,
# 142 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
float brightpass_thdesthold){
# 143 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{ static const struct uchar4 __T26 = {((unsigned char)0U),((unsigned char)0U),((unsigned char)0U),((unsigned char)255U)};
# 144 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 unsigned __cuda_local_var_38731_15_non_const_x;
# 145 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 unsigned __cuda_local_var_38732_15_non_const_y;
# 144 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38731_15_non_const_x = (((blockIdx.x) * (blockDim.x)) + (threadIdx.x));
# 145 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38732_15_non_const_y = (((blockIdx.y) * (blockDim.y)) + (threadIdx.y));
# 147 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
if ((__cuda_local_var_38731_15_non_const_x < width) && (__cuda_local_var_38732_15_non_const_y < height))
# 148 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{ float __T27;
 float __T28;
 float __T29;
 struct uchar4 __T210;
# 150 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 _ZN4math6float3E __cuda_local_var_38737_16_non_const_c;
# 153 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 _ZN4math6float3E __cuda_local_var_38740_16_non_const_c_t;
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
 struct uchar4 __cuda_local_var_38743_10_non_const_out;
__T27 = (src[((3U * ((__cuda_local_var_38732_15_non_const_y * width) + __cuda_local_var_38731_15_non_const_x)) + 0U)]); __T28 = (src[((3U * ((__cuda_local_var_38732_15_non_const_y * width) + __cuda_local_var_38731_15_non_const_x)) + 1U)]); __T29 = (src[((3U * ((__cuda_local_var_38732_15_non_const_y * width) + __cuda_local_var_38731_15_non_const_x)) + 2U)]);
# 150 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
_ZN4math6vectorIfLj3EEC1Efff((&__cuda_local_var_38737_16_non_const_c), __T27, __T28, __T29);
# 153 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
memset((char *)&__cuda_local_var_38740_16_non_const_c_t, 0,sizeof(__cuda_local_var_38740_16_non_const_c_t));
# 153 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38740_16_non_const_c_t = (_Z7tonemapRKN4math6vectorIfLj3EEEf((((const _ZN4math6float3E *)&__cuda_local_var_38737_16_non_const_c)), exposure));
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
memset((char *)&__cuda_local_var_38743_10_non_const_out, 0,sizeof(__cuda_local_var_38743_10_non_const_out));
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38743_10_non_const_out.x = ((unsigned char)0U);
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38743_10_non_const_out.y = ((unsigned char)0U);
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38743_10_non_const_out.z = ((unsigned char)0U);
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
__cuda_local_var_38743_10_non_const_out.w = ((unsigned char)255U);
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
(__cuda_local_var_38743_10_non_const_out.x) = (_Z7toSRGB8f((__cuda_local_var_38740_16_non_const_c_t.x)));
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
(__cuda_local_var_38743_10_non_const_out.y) = (_Z7toSRGB8f((__cuda_local_var_38740_16_non_const_c_t.y)));
# 156 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
(__cuda_local_var_38743_10_non_const_out.z) = (_Z7toSRGB8f((__cuda_local_var_38740_16_non_const_c_t.z)));
# 157 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
(tonemapped[((__cuda_local_var_38732_15_non_const_y * width) + __cuda_local_var_38731_15_non_const_x)]) = __cuda_local_var_38743_10_non_const_out;
# 158 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
(brightpass[((__cuda_local_var_38732_15_non_const_y * width) + __cuda_local_var_38731_15_non_const_x)]) = ((((_Z9luminanceRKN4math6vectorIfLj3EEE((((const _ZN4math6float3E *)&__cuda_local_var_38740_16_non_const_c_t)))) > brightpass_thdesthold) ? ((void)(__T210 = __cuda_local_var_38743_10_non_const_out)) : ((void)(__T210 = __T26))) , __T210);
# 159 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}
# 160 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}}
__asm__(".align 2");
# 312 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj3EEC1Efff( struct _ZN4math6vectorIfLj3EEE *const this,
# 312 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float x,
# 312 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float y,
# 312 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float z){
# 314 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
{
# 314 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
(this->x) = x;
# 314 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
(this->y) = y;
# 314 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
(this->z) = z;
# 315 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
}}
__asm__(".align 2");
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj3EEC2Efff( struct _ZN4math6vectorIfLj3EEE *const this, float __T211, float __T212, float __T213){ { _ZN4math6vectorIfLj3EEC1Efff(this, __T211, __T212, __T213); }}
__asm__(".align 2");
# 53 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj2EEC1Eff( struct _ZN4math6vectorIfLj2EEE *const this,
# 53 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float x,
# 53 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
float y){
# 55 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
{
# 55 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
(this->x) = x;
# 55 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
(this->y) = y;
# 56 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
}}
__asm__(".align 2");
static __attribute__((device)) __inline__ void _ZN4math6vectorIfLj2EEC2Eff( struct _ZN4math6vectorIfLj2EEE *const this, float __T214, float __T215){ { _ZN4math6vectorIfLj2EEC1Eff(this, __T214, __T215); }}
