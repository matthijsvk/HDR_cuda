# 1 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false

# 1
# 56 "/usr/local/cuda-8.0/include/cuda_runtime.h"
#pragma GCC diagnostic push
# 59
#pragma GCC diagnostic ignored "-Wunused-function"
# 61 "/usr/local/cuda-8.0/include/device_types.h"
#if 0
# 61
enum cudaRoundMode { 
# 63
cudaRoundNearest, 
# 64
cudaRoundZero, 
# 65
cudaRoundPosInf, 
# 66
cudaRoundMinInf
# 67
}; 
#endif
# 147 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3
typedef long ptrdiff_t; 
# 212
typedef unsigned long size_t; 
#include "crt/host_runtime.h"
# 425
typedef 
# 422
struct { 
# 423
long long __max_align_ll __attribute((__aligned__(__alignof__(long long)))); 
# 424
long double __max_align_ld __attribute((__aligned__(__alignof__(long double)))); 
# 425
} max_align_t; 
# 432
typedef __decltype((nullptr)) nullptr_t; 
# 156 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 156
enum cudaError { 
# 163
cudaSuccess, 
# 169
cudaErrorMissingConfiguration, 
# 175
cudaErrorMemoryAllocation, 
# 181
cudaErrorInitializationError, 
# 191
cudaErrorLaunchFailure, 
# 200
cudaErrorPriorLaunchFailure, 
# 210
cudaErrorLaunchTimeout, 
# 219
cudaErrorLaunchOutOfResources, 
# 225
cudaErrorInvalidDeviceFunction, 
# 234
cudaErrorInvalidConfiguration, 
# 240
cudaErrorInvalidDevice, 
# 246
cudaErrorInvalidValue, 
# 252
cudaErrorInvalidPitchValue, 
# 258
cudaErrorInvalidSymbol, 
# 263
cudaErrorMapBufferObjectFailed, 
# 268
cudaErrorUnmapBufferObjectFailed, 
# 274
cudaErrorInvalidHostPointer, 
# 280
cudaErrorInvalidDevicePointer, 
# 286
cudaErrorInvalidTexture, 
# 292
cudaErrorInvalidTextureBinding, 
# 299
cudaErrorInvalidChannelDescriptor, 
# 305
cudaErrorInvalidMemcpyDirection, 
# 315
cudaErrorAddressOfConstant, 
# 324
cudaErrorTextureFetchFailed, 
# 333
cudaErrorTextureNotBound, 
# 342
cudaErrorSynchronizationError, 
# 348
cudaErrorInvalidFilterSetting, 
# 354
cudaErrorInvalidNormSetting, 
# 362
cudaErrorMixedDeviceExecution, 
# 369
cudaErrorCudartUnloading, 
# 374
cudaErrorUnknown, 
# 382
cudaErrorNotYetImplemented, 
# 391
cudaErrorMemoryValueTooLarge, 
# 398
cudaErrorInvalidResourceHandle, 
# 406
cudaErrorNotReady, 
# 413
cudaErrorInsufficientDriver, 
# 426
cudaErrorSetOnActiveProcess, 
# 432
cudaErrorInvalidSurface, 
# 438
cudaErrorNoDevice, 
# 444
cudaErrorECCUncorrectable, 
# 449
cudaErrorSharedObjectSymbolNotFound, 
# 454
cudaErrorSharedObjectInitFailed, 
# 460
cudaErrorUnsupportedLimit, 
# 466
cudaErrorDuplicateVariableName, 
# 472
cudaErrorDuplicateTextureName, 
# 478
cudaErrorDuplicateSurfaceName, 
# 488
cudaErrorDevicesUnavailable, 
# 493
cudaErrorInvalidKernelImage, 
# 501
cudaErrorNoKernelImageForDevice, 
# 514
cudaErrorIncompatibleDriverContext, 
# 521
cudaErrorPeerAccessAlreadyEnabled, 
# 528
cudaErrorPeerAccessNotEnabled, 
# 534
cudaErrorDeviceAlreadyInUse = 54, 
# 541
cudaErrorProfilerDisabled, 
# 549
cudaErrorProfilerNotInitialized, 
# 556
cudaErrorProfilerAlreadyStarted, 
# 563
cudaErrorProfilerAlreadyStopped, 
# 571
cudaErrorAssert, 
# 578
cudaErrorTooManyPeers, 
# 584
cudaErrorHostMemoryAlreadyRegistered, 
# 590
cudaErrorHostMemoryNotRegistered, 
# 595
cudaErrorOperatingSystem, 
# 601
cudaErrorPeerAccessUnsupported, 
# 608
cudaErrorLaunchMaxDepthExceeded, 
# 616
cudaErrorLaunchFileScopedTex, 
# 624
cudaErrorLaunchFileScopedSurf, 
# 639
cudaErrorSyncDepthExceeded, 
# 651
cudaErrorLaunchPendingCountExceeded, 
# 656
cudaErrorNotPermitted, 
# 662
cudaErrorNotSupported, 
# 671
cudaErrorHardwareStackError, 
# 679
cudaErrorIllegalInstruction, 
# 688
cudaErrorMisalignedAddress, 
# 699
cudaErrorInvalidAddressSpace, 
# 707
cudaErrorInvalidPc, 
# 715
cudaErrorIllegalAddress, 
# 721
cudaErrorInvalidPtx, 
# 726
cudaErrorInvalidGraphicsContext, 
# 732
cudaErrorNvlinkUncorrectable, 
# 737
cudaErrorStartupFailure = 127, 
# 745
cudaErrorApiFailureBase = 10000
# 746
}; 
#endif
# 751 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 751
enum cudaChannelFormatKind { 
# 753
cudaChannelFormatKindSigned, 
# 754
cudaChannelFormatKindUnsigned, 
# 755
cudaChannelFormatKindFloat, 
# 756
cudaChannelFormatKindNone
# 757
}; 
#endif
# 762 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 762
struct cudaChannelFormatDesc { 
# 764
int x; 
# 765
int y; 
# 766
int z; 
# 767
int w; 
# 768
cudaChannelFormatKind f; 
# 769
}; 
#endif
# 774 "/usr/local/cuda-8.0/include/driver_types.h"
typedef struct cudaArray *cudaArray_t; 
# 779
typedef const cudaArray *cudaArray_const_t; 
# 781
struct cudaArray; 
# 786
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 791
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 793
struct cudaMipmappedArray; 
# 798
#if 0
# 798
enum cudaMemoryType { 
# 800
cudaMemoryTypeHost = 1, 
# 801
cudaMemoryTypeDevice
# 802
}; 
#endif
# 807 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 807
enum cudaMemcpyKind { 
# 809
cudaMemcpyHostToHost, 
# 810
cudaMemcpyHostToDevice, 
# 811
cudaMemcpyDeviceToHost, 
# 812
cudaMemcpyDeviceToDevice, 
# 813
cudaMemcpyDefault
# 814
}; 
#endif
# 821 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 821
struct cudaPitchedPtr { 
# 823
void *ptr; 
# 824
size_t pitch; 
# 825
size_t xsize; 
# 826
size_t ysize; 
# 827
}; 
#endif
# 834 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 834
struct cudaExtent { 
# 836
size_t width; 
# 837
size_t height; 
# 838
size_t depth; 
# 839
}; 
#endif
# 846 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 846
struct cudaPos { 
# 848
size_t x; 
# 849
size_t y; 
# 850
size_t z; 
# 851
}; 
#endif
# 856 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 856
struct cudaMemcpy3DParms { 
# 858
cudaArray_t srcArray; 
# 859
cudaPos srcPos; 
# 860
cudaPitchedPtr srcPtr; 
# 862
cudaArray_t dstArray; 
# 863
cudaPos dstPos; 
# 864
cudaPitchedPtr dstPtr; 
# 866
cudaExtent extent; 
# 867
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 868
}; 
#endif
# 873 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 873
struct cudaMemcpy3DPeerParms { 
# 875
cudaArray_t srcArray; 
# 876
cudaPos srcPos; 
# 877
cudaPitchedPtr srcPtr; 
# 878
int srcDevice; 
# 880
cudaArray_t dstArray; 
# 881
cudaPos dstPos; 
# 882
cudaPitchedPtr dstPtr; 
# 883
int dstDevice; 
# 885
cudaExtent extent; 
# 886
}; 
#endif
# 891 "/usr/local/cuda-8.0/include/driver_types.h"
struct cudaGraphicsResource; 
# 896
#if 0
# 896
enum cudaGraphicsRegisterFlags { 
# 898
cudaGraphicsRegisterFlagsNone, 
# 899
cudaGraphicsRegisterFlagsReadOnly, 
# 900
cudaGraphicsRegisterFlagsWriteDiscard, 
# 901
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 902
cudaGraphicsRegisterFlagsTextureGather = 8
# 903
}; 
#endif
# 908 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 908
enum cudaGraphicsMapFlags { 
# 910
cudaGraphicsMapFlagsNone, 
# 911
cudaGraphicsMapFlagsReadOnly, 
# 912
cudaGraphicsMapFlagsWriteDiscard
# 913
}; 
#endif
# 918 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 918
enum cudaGraphicsCubeFace { 
# 920
cudaGraphicsCubeFacePositiveX, 
# 921
cudaGraphicsCubeFaceNegativeX, 
# 922
cudaGraphicsCubeFacePositiveY, 
# 923
cudaGraphicsCubeFaceNegativeY, 
# 924
cudaGraphicsCubeFacePositiveZ, 
# 925
cudaGraphicsCubeFaceNegativeZ
# 926
}; 
#endif
# 931 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 931
enum cudaResourceType { 
# 933
cudaResourceTypeArray, 
# 934
cudaResourceTypeMipmappedArray, 
# 935
cudaResourceTypeLinear, 
# 936
cudaResourceTypePitch2D
# 937
}; 
#endif
# 942 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 942
enum cudaResourceViewFormat { 
# 944
cudaResViewFormatNone, 
# 945
cudaResViewFormatUnsignedChar1, 
# 946
cudaResViewFormatUnsignedChar2, 
# 947
cudaResViewFormatUnsignedChar4, 
# 948
cudaResViewFormatSignedChar1, 
# 949
cudaResViewFormatSignedChar2, 
# 950
cudaResViewFormatSignedChar4, 
# 951
cudaResViewFormatUnsignedShort1, 
# 952
cudaResViewFormatUnsignedShort2, 
# 953
cudaResViewFormatUnsignedShort4, 
# 954
cudaResViewFormatSignedShort1, 
# 955
cudaResViewFormatSignedShort2, 
# 956
cudaResViewFormatSignedShort4, 
# 957
cudaResViewFormatUnsignedInt1, 
# 958
cudaResViewFormatUnsignedInt2, 
# 959
cudaResViewFormatUnsignedInt4, 
# 960
cudaResViewFormatSignedInt1, 
# 961
cudaResViewFormatSignedInt2, 
# 962
cudaResViewFormatSignedInt4, 
# 963
cudaResViewFormatHalf1, 
# 964
cudaResViewFormatHalf2, 
# 965
cudaResViewFormatHalf4, 
# 966
cudaResViewFormatFloat1, 
# 967
cudaResViewFormatFloat2, 
# 968
cudaResViewFormatFloat4, 
# 969
cudaResViewFormatUnsignedBlockCompressed1, 
# 970
cudaResViewFormatUnsignedBlockCompressed2, 
# 971
cudaResViewFormatUnsignedBlockCompressed3, 
# 972
cudaResViewFormatUnsignedBlockCompressed4, 
# 973
cudaResViewFormatSignedBlockCompressed4, 
# 974
cudaResViewFormatUnsignedBlockCompressed5, 
# 975
cudaResViewFormatSignedBlockCompressed5, 
# 976
cudaResViewFormatUnsignedBlockCompressed6H, 
# 977
cudaResViewFormatSignedBlockCompressed6H, 
# 978
cudaResViewFormatUnsignedBlockCompressed7
# 979
}; 
#endif
# 984 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 984
struct cudaResourceDesc { 
# 985
cudaResourceType resType; 
# 987
union { 
# 988
struct { 
# 989
cudaArray_t array; 
# 990
} array; 
# 991
struct { 
# 992
cudaMipmappedArray_t mipmap; 
# 993
} mipmap; 
# 994
struct { 
# 995
void *devPtr; 
# 996
cudaChannelFormatDesc desc; 
# 997
size_t sizeInBytes; 
# 998
} linear; 
# 999
struct { 
# 1000
void *devPtr; 
# 1001
cudaChannelFormatDesc desc; 
# 1002
size_t width; 
# 1003
size_t height; 
# 1004
size_t pitchInBytes; 
# 1005
} pitch2D; 
# 1006
} res; 
# 1007
}; 
#endif
# 1012 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1012
struct cudaResourceViewDesc { 
# 1014
cudaResourceViewFormat format; 
# 1015
size_t width; 
# 1016
size_t height; 
# 1017
size_t depth; 
# 1018
unsigned firstMipmapLevel; 
# 1019
unsigned lastMipmapLevel; 
# 1020
unsigned firstLayer; 
# 1021
unsigned lastLayer; 
# 1022
}; 
#endif
# 1027 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1027
struct cudaPointerAttributes { 
# 1033
cudaMemoryType memoryType; 
# 1044
int device; 
# 1050
void *devicePointer; 
# 1056
void *hostPointer; 
# 1061
int isManaged; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1062
}; 
#endif
# 1067 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1067
struct cudaFuncAttributes { 
# 1074
size_t sharedSizeBytes; 
# 1080
size_t constSizeBytes; 
# 1085
size_t localSizeBytes; 
# 1092
int maxThreadsPerBlock; 
# 1097
int numRegs; 
# 1104
int ptxVersion; 
# 1111
int binaryVersion; 
# 1117
int cacheModeCA; 
# 1118
}; 
#endif
# 1123 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1123
enum cudaFuncCache { 
# 1125
cudaFuncCachePreferNone, 
# 1126
cudaFuncCachePreferShared, 
# 1127
cudaFuncCachePreferL1, 
# 1128
cudaFuncCachePreferEqual
# 1129
}; 
#endif
# 1135 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1135
enum cudaSharedMemConfig { 
# 1137
cudaSharedMemBankSizeDefault, 
# 1138
cudaSharedMemBankSizeFourByte, 
# 1139
cudaSharedMemBankSizeEightByte
# 1140
}; 
#endif
# 1145 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1145
enum cudaComputeMode { 
# 1147
cudaComputeModeDefault, 
# 1148
cudaComputeModeExclusive, 
# 1149
cudaComputeModeProhibited, 
# 1150
cudaComputeModeExclusiveProcess
# 1151
}; 
#endif
# 1156 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1156
enum cudaLimit { 
# 1158
cudaLimitStackSize, 
# 1159
cudaLimitPrintfFifoSize, 
# 1160
cudaLimitMallocHeapSize, 
# 1161
cudaLimitDevRuntimeSyncDepth, 
# 1162
cudaLimitDevRuntimePendingLaunchCount
# 1163
}; 
#endif
# 1168 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1168
enum cudaMemoryAdvise { 
# 1170
cudaMemAdviseSetReadMostly = 1, 
# 1171
cudaMemAdviseUnsetReadMostly, 
# 1172
cudaMemAdviseSetPreferredLocation, 
# 1173
cudaMemAdviseUnsetPreferredLocation, 
# 1174
cudaMemAdviseSetAccessedBy, 
# 1175
cudaMemAdviseUnsetAccessedBy
# 1176
}; 
#endif
# 1181 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1181
enum cudaMemRangeAttribute { 
# 1183
cudaMemRangeAttributeReadMostly = 1, 
# 1184
cudaMemRangeAttributePreferredLocation, 
# 1185
cudaMemRangeAttributeAccessedBy, 
# 1186
cudaMemRangeAttributeLastPrefetchLocation
# 1187
}; 
#endif
# 1192 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1192
enum cudaOutputMode { 
# 1194
cudaKeyValuePair, 
# 1195
cudaCSV
# 1196
}; 
#endif
# 1201 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1201
enum cudaDeviceAttr { 
# 1203
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1204
cudaDevAttrMaxBlockDimX, 
# 1205
cudaDevAttrMaxBlockDimY, 
# 1206
cudaDevAttrMaxBlockDimZ, 
# 1207
cudaDevAttrMaxGridDimX, 
# 1208
cudaDevAttrMaxGridDimY, 
# 1209
cudaDevAttrMaxGridDimZ, 
# 1210
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1211
cudaDevAttrTotalConstantMemory, 
# 1212
cudaDevAttrWarpSize, 
# 1213
cudaDevAttrMaxPitch, 
# 1214
cudaDevAttrMaxRegistersPerBlock, 
# 1215
cudaDevAttrClockRate, 
# 1216
cudaDevAttrTextureAlignment, 
# 1217
cudaDevAttrGpuOverlap, 
# 1218
cudaDevAttrMultiProcessorCount, 
# 1219
cudaDevAttrKernelExecTimeout, 
# 1220
cudaDevAttrIntegrated, 
# 1221
cudaDevAttrCanMapHostMemory, 
# 1222
cudaDevAttrComputeMode, 
# 1223
cudaDevAttrMaxTexture1DWidth, 
# 1224
cudaDevAttrMaxTexture2DWidth, 
# 1225
cudaDevAttrMaxTexture2DHeight, 
# 1226
cudaDevAttrMaxTexture3DWidth, 
# 1227
cudaDevAttrMaxTexture3DHeight, 
# 1228
cudaDevAttrMaxTexture3DDepth, 
# 1229
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1230
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1231
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1232
cudaDevAttrSurfaceAlignment, 
# 1233
cudaDevAttrConcurrentKernels, 
# 1234
cudaDevAttrEccEnabled, 
# 1235
cudaDevAttrPciBusId, 
# 1236
cudaDevAttrPciDeviceId, 
# 1237
cudaDevAttrTccDriver, 
# 1238
cudaDevAttrMemoryClockRate, 
# 1239
cudaDevAttrGlobalMemoryBusWidth, 
# 1240
cudaDevAttrL2CacheSize, 
# 1241
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1242
cudaDevAttrAsyncEngineCount, 
# 1243
cudaDevAttrUnifiedAddressing, 
# 1244
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1245
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1246
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1247
cudaDevAttrMaxTexture2DGatherHeight, 
# 1248
cudaDevAttrMaxTexture3DWidthAlt, 
# 1249
cudaDevAttrMaxTexture3DHeightAlt, 
# 1250
cudaDevAttrMaxTexture3DDepthAlt, 
# 1251
cudaDevAttrPciDomainId, 
# 1252
cudaDevAttrTexturePitchAlignment, 
# 1253
cudaDevAttrMaxTextureCubemapWidth, 
# 1254
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1255
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1256
cudaDevAttrMaxSurface1DWidth, 
# 1257
cudaDevAttrMaxSurface2DWidth, 
# 1258
cudaDevAttrMaxSurface2DHeight, 
# 1259
cudaDevAttrMaxSurface3DWidth, 
# 1260
cudaDevAttrMaxSurface3DHeight, 
# 1261
cudaDevAttrMaxSurface3DDepth, 
# 1262
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1263
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1264
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1265
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1266
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1267
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1268
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1269
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1270
cudaDevAttrMaxTexture1DLinearWidth, 
# 1271
cudaDevAttrMaxTexture2DLinearWidth, 
# 1272
cudaDevAttrMaxTexture2DLinearHeight, 
# 1273
cudaDevAttrMaxTexture2DLinearPitch, 
# 1274
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1275
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1276
cudaDevAttrComputeCapabilityMajor, 
# 1277
cudaDevAttrComputeCapabilityMinor, 
# 1278
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1279
cudaDevAttrStreamPrioritiesSupported, 
# 1280
cudaDevAttrGlobalL1CacheSupported, 
# 1281
cudaDevAttrLocalL1CacheSupported, 
# 1282
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1283
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1284
cudaDevAttrManagedMemory, 
# 1285
cudaDevAttrIsMultiGpuBoard, 
# 1286
cudaDevAttrMultiGpuBoardGroupID, 
# 1287
cudaDevAttrHostNativeAtomicSupported, 
# 1288
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1289
cudaDevAttrPageableMemoryAccess, 
# 1290
cudaDevAttrConcurrentManagedAccess, 
# 1291
cudaDevAttrComputePreemptionSupported, 
# 1292
cudaDevAttrCanUseHostPointerForRegisteredMem
# 1293
}; 
#endif
# 1299 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1299
enum cudaDeviceP2PAttr { 
# 1300
cudaDevP2PAttrPerformanceRank = 1, 
# 1301
cudaDevP2PAttrAccessSupported, 
# 1302
cudaDevP2PAttrNativeAtomicSupported
# 1303
}; 
#endif
# 1307 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
# 1307
struct cudaDeviceProp { 
# 1309
char name[256]; 
# 1310
size_t totalGlobalMem; 
# 1311
size_t sharedMemPerBlock; 
# 1312
int regsPerBlock; 
# 1313
int warpSize; 
# 1314
size_t memPitch; 
# 1315
int maxThreadsPerBlock; 
# 1316
int maxThreadsDim[3]; 
# 1317
int maxGridSize[3]; 
# 1318
int clockRate; 
# 1319
size_t totalConstMem; 
# 1320
int major; 
# 1321
int minor; 
# 1322
size_t textureAlignment; 
# 1323
size_t texturePitchAlignment; 
# 1324
int deviceOverlap; 
# 1325
int multiProcessorCount; 
# 1326
int kernelExecTimeoutEnabled; 
# 1327
int integrated; 
# 1328
int canMapHostMemory; 
# 1329
int computeMode; 
# 1330
int maxTexture1D; 
# 1331
int maxTexture1DMipmap; 
# 1332
int maxTexture1DLinear; 
# 1333
int maxTexture2D[2]; 
# 1334
int maxTexture2DMipmap[2]; 
# 1335
int maxTexture2DLinear[3]; 
# 1336
int maxTexture2DGather[2]; 
# 1337
int maxTexture3D[3]; 
# 1338
int maxTexture3DAlt[3]; 
# 1339
int maxTextureCubemap; 
# 1340
int maxTexture1DLayered[2]; 
# 1341
int maxTexture2DLayered[3]; 
# 1342
int maxTextureCubemapLayered[2]; 
# 1343
int maxSurface1D; 
# 1344
int maxSurface2D[2]; 
# 1345
int maxSurface3D[3]; 
# 1346
int maxSurface1DLayered[2]; 
# 1347
int maxSurface2DLayered[3]; 
# 1348
int maxSurfaceCubemap; 
# 1349
int maxSurfaceCubemapLayered[2]; 
# 1350
size_t surfaceAlignment; 
# 1351
int concurrentKernels; 
# 1352
int ECCEnabled; 
# 1353
int pciBusID; 
# 1354
int pciDeviceID; 
# 1355
int pciDomainID; 
# 1356
int tccDriver; 
# 1357
int asyncEngineCount; 
# 1358
int unifiedAddressing; 
# 1359
int memoryClockRate; 
# 1360
int memoryBusWidth; 
# 1361
int l2CacheSize; 
# 1362
int maxThreadsPerMultiProcessor; 
# 1363
int streamPrioritiesSupported; 
# 1364
int globalL1CacheSupported; 
# 1365
int localL1CacheSupported; 
# 1366
size_t sharedMemPerMultiprocessor; 
# 1367
int regsPerMultiprocessor; 
# 1368
int managedMemory; 
# 1369
int isMultiGpuBoard; 
# 1370
int multiGpuBoardGroupID; 
# 1371
int hostNativeAtomicSupported; 
# 1372
int singleToDoublePrecisionPerfRatio; 
# 1373
int pageableMemoryAccess; 
# 1374
int concurrentManagedAccess; 
# 1375
}; 
#endif
# 1458 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef 
# 1455
struct cudaIpcEventHandle_st { 
# 1457
char reserved[64]; 
# 1458
} cudaIpcEventHandle_t; 
#endif
# 1466 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef 
# 1463
struct cudaIpcMemHandle_st { 
# 1465
char reserved[64]; 
# 1466
} cudaIpcMemHandle_t; 
#endif
# 1477 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef cudaError 
# 1477
cudaError_t; 
#endif
# 1482 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef struct CUstream_st *
# 1482
cudaStream_t; 
#endif
# 1487 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef struct CUevent_st *
# 1487
cudaEvent_t; 
#endif
# 1492 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef cudaGraphicsResource *
# 1492
cudaGraphicsResource_t; 
#endif
# 1497 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef struct CUuuid_st 
# 1497
cudaUUID_t; 
#endif
# 1502 "/usr/local/cuda-8.0/include/driver_types.h"
#if 0
typedef cudaOutputMode 
# 1502
cudaOutputMode_t; 
#endif
# 84 "/usr/local/cuda-8.0/include/surface_types.h"
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/usr/local/cuda-8.0/include/surface_types.h"
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/usr/local/cuda-8.0/include/surface_types.h"
#if 0
# 103
struct surfaceReference { 
# 108
cudaChannelFormatDesc channelDesc; 
# 109
}; 
#endif
# 114 "/usr/local/cuda-8.0/include/surface_types.h"
#if 0
typedef unsigned long long 
# 114
cudaSurfaceObject_t; 
#endif
# 84 "/usr/local/cuda-8.0/include/texture_types.h"
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/usr/local/cuda-8.0/include/texture_types.h"
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/usr/local/cuda-8.0/include/texture_types.h"
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/usr/local/cuda-8.0/include/texture_types.h"
#if 0
# 113
struct textureReference { 
# 118
int normalized; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureAddressMode addressMode[3]; 
# 130
cudaChannelFormatDesc channelDesc; 
# 134
int sRGB; 
# 138
unsigned maxAnisotropy; 
# 142
cudaTextureFilterMode mipmapFilterMode; 
# 146
float mipmapLevelBias; 
# 150
float minMipmapLevelClamp; 
# 154
float maxMipmapLevelClamp; 
# 155
int __cudaReserved[15]; 
# 156
}; 
#endif
# 161 "/usr/local/cuda-8.0/include/texture_types.h"
#if 0
# 161
struct cudaTextureDesc { 
# 166
cudaTextureAddressMode addressMode[3]; 
# 170
cudaTextureFilterMode filterMode; 
# 174
cudaTextureReadMode readMode; 
# 178
int sRGB; 
# 182
float borderColor[4]; 
# 186
int normalizedCoords; 
# 190
unsigned maxAnisotropy; 
# 194
cudaTextureFilterMode mipmapFilterMode; 
# 198
float mipmapLevelBias; 
# 202
float minMipmapLevelClamp; 
# 206
float maxMipmapLevelClamp; 
# 207
}; 
#endif
# 212 "/usr/local/cuda-8.0/include/texture_types.h"
#if 0
typedef unsigned long long 
# 212
cudaTextureObject_t; 
#endif
# 98 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 98
struct char1 { 
# 100
signed char x; 
# 101
}; 
#endif
# 103 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 103
struct uchar1 { 
# 105
unsigned char x; 
# 106
}; 
#endif
# 109 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 109
struct __attribute((aligned(2))) char2 { 
# 111
signed char x, y; 
# 112
}; 
#endif
# 114 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 114
struct __attribute((aligned(2))) uchar2 { 
# 116
unsigned char x, y; 
# 117
}; 
#endif
# 119 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 119
struct char3 { 
# 121
signed char x, y, z; 
# 122
}; 
#endif
# 124 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 124
struct uchar3 { 
# 126
unsigned char x, y, z; 
# 127
}; 
#endif
# 129 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 129
struct __attribute((aligned(4))) char4 { 
# 131
signed char x, y, z, w; 
# 132
}; 
#endif
# 134 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 134
struct __attribute((aligned(4))) uchar4 { 
# 136
unsigned char x, y, z, w; 
# 137
}; 
#endif
# 139 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 139
struct short1 { 
# 141
short x; 
# 142
}; 
#endif
# 144 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 144
struct ushort1 { 
# 146
unsigned short x; 
# 147
}; 
#endif
# 149 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 149
struct __attribute((aligned(4))) short2 { 
# 151
short x, y; 
# 152
}; 
#endif
# 154 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 154
struct __attribute((aligned(4))) ushort2 { 
# 156
unsigned short x, y; 
# 157
}; 
#endif
# 159 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 159
struct short3 { 
# 161
short x, y, z; 
# 162
}; 
#endif
# 164 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 164
struct ushort3 { 
# 166
unsigned short x, y, z; 
# 167
}; 
#endif
# 169 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 169
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 170 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 170
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 172 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 172
struct int1 { 
# 174
int x; 
# 175
}; 
#endif
# 177 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 177
struct uint1 { 
# 179
unsigned x; 
# 180
}; 
#endif
# 182 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 182
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 183 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 183
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 185 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 185
struct int3 { 
# 187
int x, y, z; 
# 188
}; 
#endif
# 190 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 190
struct uint3 { 
# 192
unsigned x, y, z; 
# 193
}; 
#endif
# 195 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 195
struct __attribute((aligned(16))) int4 { 
# 197
int x, y, z, w; 
# 198
}; 
#endif
# 200 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 200
struct __attribute((aligned(16))) uint4 { 
# 202
unsigned x, y, z, w; 
# 203
}; 
#endif
# 205 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 205
struct long1 { 
# 207
long x; 
# 208
}; 
#endif
# 210 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 210
struct ulong1 { 
# 212
unsigned long x; 
# 213
}; 
#endif
# 220 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 220
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 222
long x, y; 
# 223
}; 
#endif
# 225 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 225
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 227
unsigned long x, y; 
# 228
}; 
#endif
# 232 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 232
struct long3 { 
# 234
long x, y, z; 
# 235
}; 
#endif
# 237 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 237
struct ulong3 { 
# 239
unsigned long x, y, z; 
# 240
}; 
#endif
# 242 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 242
struct __attribute((aligned(16))) long4 { 
# 244
long x, y, z, w; 
# 245
}; 
#endif
# 247 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 247
struct __attribute((aligned(16))) ulong4 { 
# 249
unsigned long x, y, z, w; 
# 250
}; 
#endif
# 252 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 252
struct float1 { 
# 254
float x; 
# 255
}; 
#endif
# 274 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 274
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 279 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 279
struct float3 { 
# 281
float x, y, z; 
# 282
}; 
#endif
# 284 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 284
struct __attribute((aligned(16))) float4 { 
# 286
float x, y, z, w; 
# 287
}; 
#endif
# 289 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 289
struct longlong1 { 
# 291
long long x; 
# 292
}; 
#endif
# 294 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 294
struct ulonglong1 { 
# 296
unsigned long long x; 
# 297
}; 
#endif
# 299 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 299
struct __attribute((aligned(16))) longlong2 { 
# 301
long long x, y; 
# 302
}; 
#endif
# 304 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 304
struct __attribute((aligned(16))) ulonglong2 { 
# 306
unsigned long long x, y; 
# 307
}; 
#endif
# 309 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 309
struct longlong3 { 
# 311
long long x, y, z; 
# 312
}; 
#endif
# 314 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 314
struct ulonglong3 { 
# 316
unsigned long long x, y, z; 
# 317
}; 
#endif
# 319 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 319
struct __attribute((aligned(16))) longlong4 { 
# 321
long long x, y, z, w; 
# 322
}; 
#endif
# 324 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 324
struct __attribute((aligned(16))) ulonglong4 { 
# 326
unsigned long long x, y, z, w; 
# 327
}; 
#endif
# 329 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 329
struct double1 { 
# 331
double x; 
# 332
}; 
#endif
# 334 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 334
struct __attribute((aligned(16))) double2 { 
# 336
double x, y; 
# 337
}; 
#endif
# 339 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 339
struct double3 { 
# 341
double x, y, z; 
# 342
}; 
#endif
# 344 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 344
struct __attribute((aligned(16))) double4 { 
# 346
double x, y, z, w; 
# 347
}; 
#endif
# 362 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef char1 
# 362
char1; 
#endif
# 363 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uchar1 
# 363
uchar1; 
#endif
# 364 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef char2 
# 364
char2; 
#endif
# 365 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uchar2 
# 365
uchar2; 
#endif
# 366 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef char3 
# 366
char3; 
#endif
# 367 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uchar3 
# 367
uchar3; 
#endif
# 368 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef char4 
# 368
char4; 
#endif
# 369 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uchar4 
# 369
uchar4; 
#endif
# 370 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef short1 
# 370
short1; 
#endif
# 371 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ushort1 
# 371
ushort1; 
#endif
# 372 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef short2 
# 372
short2; 
#endif
# 373 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ushort2 
# 373
ushort2; 
#endif
# 374 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef short3 
# 374
short3; 
#endif
# 375 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ushort3 
# 375
ushort3; 
#endif
# 376 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef short4 
# 376
short4; 
#endif
# 377 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ushort4 
# 377
ushort4; 
#endif
# 378 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef int1 
# 378
int1; 
#endif
# 379 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uint1 
# 379
uint1; 
#endif
# 380 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef int2 
# 380
int2; 
#endif
# 381 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uint2 
# 381
uint2; 
#endif
# 382 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef int3 
# 382
int3; 
#endif
# 383 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uint3 
# 383
uint3; 
#endif
# 384 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef int4 
# 384
int4; 
#endif
# 385 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef uint4 
# 385
uint4; 
#endif
# 386 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef long1 
# 386
long1; 
#endif
# 387 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulong1 
# 387
ulong1; 
#endif
# 388 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef long2 
# 388
long2; 
#endif
# 389 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulong2 
# 389
ulong2; 
#endif
# 390 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef long3 
# 390
long3; 
#endif
# 391 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulong3 
# 391
ulong3; 
#endif
# 392 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef long4 
# 392
long4; 
#endif
# 393 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulong4 
# 393
ulong4; 
#endif
# 394 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef float1 
# 394
float1; 
#endif
# 395 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef float2 
# 395
float2; 
#endif
# 396 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef float3 
# 396
float3; 
#endif
# 397 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef float4 
# 397
float4; 
#endif
# 398 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef longlong1 
# 398
longlong1; 
#endif
# 399 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulonglong1 
# 399
ulonglong1; 
#endif
# 400 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef longlong2 
# 400
longlong2; 
#endif
# 401 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulonglong2 
# 401
ulonglong2; 
#endif
# 402 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef longlong3 
# 402
longlong3; 
#endif
# 403 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulonglong3 
# 403
ulonglong3; 
#endif
# 404 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef longlong4 
# 404
longlong4; 
#endif
# 405 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef ulonglong4 
# 405
ulonglong4; 
#endif
# 406 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef double1 
# 406
double1; 
#endif
# 407 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef double2 
# 407
double2; 
#endif
# 408 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef double3 
# 408
double3; 
#endif
# 409 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef double4 
# 409
double4; 
#endif
# 417 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
# 417
struct dim3 { 
# 419
unsigned x, y, z; 
# 425
}; 
#endif
# 427 "/usr/local/cuda-8.0/include/vector_types.h"
#if 0
typedef dim3 
# 427
dim3; 
#endif
# 70 "/usr/local/cuda-8.0/include/library_types.h"
typedef 
# 54
enum cudaDataType_t { 
# 56
CUDA_R_16F = 2, 
# 57
CUDA_C_16F = 6, 
# 58
CUDA_R_32F = 0, 
# 59
CUDA_C_32F = 4, 
# 60
CUDA_R_64F = 1, 
# 61
CUDA_C_64F = 5, 
# 62
CUDA_R_8I = 3, 
# 63
CUDA_C_8I = 7, 
# 64
CUDA_R_8U, 
# 65
CUDA_C_8U, 
# 66
CUDA_R_32I, 
# 67
CUDA_C_32I, 
# 68
CUDA_R_32U, 
# 69
CUDA_C_32U
# 70
} cudaDataType; 
# 78
typedef 
# 73
enum libraryPropertyType_t { 
# 75
MAJOR_VERSION, 
# 76
MINOR_VERSION, 
# 77
PATCH_LEVEL
# 78
} libraryPropertyType; 
# 104 "/usr/local/cuda-8.0/include/cuda_device_runtime_api.h"
extern "C" {
# 106
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 107
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 108
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 109
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 110
extern cudaError_t cudaDeviceSynchronize(); 
# 111
extern cudaError_t cudaGetLastError(); 
# 112
extern cudaError_t cudaPeekAtLastError(); 
# 113
extern const char *cudaGetErrorString(cudaError_t error); 
# 114
extern const char *cudaGetErrorName(cudaError_t error); 
# 115
extern cudaError_t cudaGetDeviceCount(int * count); 
# 116
extern cudaError_t cudaGetDevice(int * device); 
# 117
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 118
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 119
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 120
__attribute__((unused)) extern cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 121
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 122
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
# 123
__attribute__((unused)) extern cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 124
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 125
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 126
extern cudaError_t cudaFree(void * devPtr); 
# 127
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 128
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 129
__attribute__((unused)) extern cudaError_t cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 130
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 131
__attribute__((unused)) extern cudaError_t cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 132
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 133
__attribute__((unused)) extern cudaError_t cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 134
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 135
__attribute__((unused)) extern cudaError_t cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 136
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 137
__attribute__((unused)) extern cudaError_t cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 138
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 139
__attribute__((unused)) extern cudaError_t cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 140
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 161
__attribute__((unused)) extern void *cudaGetParameterBuffer(size_t alignment, size_t size); 
# 189
__attribute__((unused)) extern void *cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 190
__attribute__((unused)) extern cudaError_t cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 191
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 209
__attribute__((unused)) extern cudaError_t cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 210
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 213
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 214
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 216
}
# 218
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 219
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 220
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 221
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 219 "/usr/local/cuda-8.0/include/cuda_runtime_api.h"
extern "C" {
# 252
extern cudaError_t cudaDeviceReset(); 
# 269
extern cudaError_t cudaDeviceSynchronize(); 
# 344
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 373
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 404
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 439
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 481
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 510
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 552
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 575
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 602
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 644
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 679
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 717
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 767
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 797
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 837
extern cudaError_t cudaThreadExit(); 
# 861
extern cudaError_t cudaThreadSynchronize(); 
# 908
extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 939
extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 974
extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1020
extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1074
extern cudaError_t cudaGetLastError(); 
# 1115
extern cudaError_t cudaPeekAtLastError(); 
# 1130
extern const char *cudaGetErrorName(cudaError_t error); 
# 1145
extern const char *cudaGetErrorString(cudaError_t error); 
# 1175
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1421
extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
# 1593
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1628
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 1647
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 1681
extern cudaError_t cudaSetDevice(int device); 
# 1698
extern cudaError_t cudaGetDevice(int * device); 
# 1727
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 1789
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 1830
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 1867
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 1896
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 1939
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 1963
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 1984
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2005
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2037
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 2051
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2108
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2128
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2149
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2220
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2256
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 2290
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 2321
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 2353
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 2385
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 2410
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 2451
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 2510
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 2560
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 2614
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 2648
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 2671
extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 2694
extern cudaError_t cudaSetDoubleForHost(double * d); 
# 2750
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 2794
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 2845
extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0); 
# 2874
extern cudaError_t cudaSetupArgument(const void * arg, size_t size, size_t offset); 
# 2912
extern cudaError_t cudaLaunch(const void * func); 
# 3030
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 3056
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 3085
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 3124
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 3166
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 3193
extern cudaError_t cudaFree(void * devPtr); 
# 3213
extern cudaError_t cudaFreeHost(void * ptr); 
# 3235
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 3257
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 3316
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 3393
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 3412
extern cudaError_t cudaHostUnregister(void * ptr); 
# 3454
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 3473
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 3508
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 3643
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 3764
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 3790
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 3890
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 3918
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 4029
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 4052
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 4071
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 4092
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 4130
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 4162
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 4200
extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 4237
extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 4275
extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 4320
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 4366
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 4412
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 4455
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 4493
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 4531
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 4582
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4614
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 4660
extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4705
extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4765
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4820
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4874
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4920
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4966
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4992
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 5022
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 5065
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 5097
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 5134
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 5184
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 5207
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 5229
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 5296
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 5382
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 5438
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 5474
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 5627
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 5665
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 5704
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 5723
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 5783
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 5815
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 5851
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 5883
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 5912
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 5946
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 5971
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 6011
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 6046
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 6093
extern cudaError_t cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = ((2147483647) * 2U) + 1U); 
# 6144
extern cudaError_t cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
# 6172
extern cudaError_t cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 6200
extern cudaError_t cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
# 6221
extern cudaError_t cudaUnbindTexture(const textureReference * texref); 
# 6246
extern cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
# 6271
extern cudaError_t cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
# 6311
extern cudaError_t cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 6330
extern cudaError_t cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
# 6556
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 6571
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 6587
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 6603
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 6620
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 6659
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 6674
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 6689
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 6716
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 6733
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 6738
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 6964
}
# 107 "/usr/local/cuda-8.0/include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 108
{ 
# 109
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 110
} 
# 112
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 113
{ 
# 114
int e = (((int)sizeof(unsigned short)) * 8); 
# 116
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 117
} 
# 119
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 120
{ 
# 121
int e = (((int)sizeof(unsigned short)) * 8); 
# 123
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 124
} 
# 126
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 127
{ 
# 128
int e = (((int)sizeof(unsigned short)) * 8); 
# 130
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 131
} 
# 133
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 134
{ 
# 135
int e = (((int)sizeof(unsigned short)) * 8); 
# 137
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 138
} 
# 140
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 141
{ 
# 142
int e = (((int)sizeof(char)) * 8); 
# 147
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 149
} 
# 151
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 152
{ 
# 153
int e = (((int)sizeof(signed char)) * 8); 
# 155
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 156
} 
# 158
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 159
{ 
# 160
int e = (((int)sizeof(unsigned char)) * 8); 
# 162
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 163
} 
# 165
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 166
{ 
# 167
int e = (((int)sizeof(signed char)) * 8); 
# 169
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 170
} 
# 172
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 173
{ 
# 174
int e = (((int)sizeof(unsigned char)) * 8); 
# 176
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 177
} 
# 179
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 180
{ 
# 181
int e = (((int)sizeof(signed char)) * 8); 
# 183
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 184
} 
# 186
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 187
{ 
# 188
int e = (((int)sizeof(unsigned char)) * 8); 
# 190
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 191
} 
# 193
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 194
{ 
# 195
int e = (((int)sizeof(signed char)) * 8); 
# 197
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 198
} 
# 200
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 201
{ 
# 202
int e = (((int)sizeof(unsigned char)) * 8); 
# 204
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 205
} 
# 207
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 208
{ 
# 209
int e = (((int)sizeof(short)) * 8); 
# 211
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 212
} 
# 214
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 215
{ 
# 216
int e = (((int)sizeof(unsigned short)) * 8); 
# 218
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 219
} 
# 221
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 222
{ 
# 223
int e = (((int)sizeof(short)) * 8); 
# 225
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 226
} 
# 228
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 229
{ 
# 230
int e = (((int)sizeof(unsigned short)) * 8); 
# 232
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 233
} 
# 235
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 236
{ 
# 237
int e = (((int)sizeof(short)) * 8); 
# 239
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 240
} 
# 242
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 243
{ 
# 244
int e = (((int)sizeof(unsigned short)) * 8); 
# 246
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 247
} 
# 249
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 250
{ 
# 251
int e = (((int)sizeof(short)) * 8); 
# 253
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 254
} 
# 256
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 257
{ 
# 258
int e = (((int)sizeof(unsigned short)) * 8); 
# 260
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 261
} 
# 263
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 264
{ 
# 265
int e = (((int)sizeof(int)) * 8); 
# 267
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 268
} 
# 270
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 271
{ 
# 272
int e = (((int)sizeof(unsigned)) * 8); 
# 274
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 275
} 
# 277
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 278
{ 
# 279
int e = (((int)sizeof(int)) * 8); 
# 281
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 282
} 
# 284
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 285
{ 
# 286
int e = (((int)sizeof(unsigned)) * 8); 
# 288
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 289
} 
# 291
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 292
{ 
# 293
int e = (((int)sizeof(int)) * 8); 
# 295
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 296
} 
# 298
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 299
{ 
# 300
int e = (((int)sizeof(unsigned)) * 8); 
# 302
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 303
} 
# 305
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 306
{ 
# 307
int e = (((int)sizeof(int)) * 8); 
# 309
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 310
} 
# 312
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 313
{ 
# 314
int e = (((int)sizeof(unsigned)) * 8); 
# 316
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 317
} 
# 379
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 380
{ 
# 381
int e = (((int)sizeof(float)) * 8); 
# 383
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 384
} 
# 386
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 387
{ 
# 388
int e = (((int)sizeof(float)) * 8); 
# 390
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 391
} 
# 393
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 394
{ 
# 395
int e = (((int)sizeof(float)) * 8); 
# 397
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 398
} 
# 400
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 401
{ 
# 402
int e = (((int)sizeof(float)) * 8); 
# 404
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 405
} 
# 79 "/usr/local/cuda-8.0/include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 75 "/usr/local/cuda-8.0/include/vector_functions.hpp"
static inline char1 make_char1(signed char x) 
# 76
{ 
# 77
char1 t; (t.x) = x; return t; 
# 78
} 
# 80
static inline uchar1 make_uchar1(unsigned char x) 
# 81
{ 
# 82
uchar1 t; (t.x) = x; return t; 
# 83
} 
# 85
static inline char2 make_char2(signed char x, signed char y) 
# 86
{ 
# 87
char2 t; (t.x) = x; (t.y) = y; return t; 
# 88
} 
# 90
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 91
{ 
# 92
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 93
} 
# 95
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 96
{ 
# 97
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 98
} 
# 100
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 101
{ 
# 102
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 103
} 
# 105
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 106
{ 
# 107
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 108
} 
# 110
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 111
{ 
# 112
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 113
} 
# 115
static inline short1 make_short1(short x) 
# 116
{ 
# 117
short1 t; (t.x) = x; return t; 
# 118
} 
# 120
static inline ushort1 make_ushort1(unsigned short x) 
# 121
{ 
# 122
ushort1 t; (t.x) = x; return t; 
# 123
} 
# 125
static inline short2 make_short2(short x, short y) 
# 126
{ 
# 127
short2 t; (t.x) = x; (t.y) = y; return t; 
# 128
} 
# 130
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 131
{ 
# 132
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 133
} 
# 135
static inline short3 make_short3(short x, short y, short z) 
# 136
{ 
# 137
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 138
} 
# 140
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 141
{ 
# 142
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 143
} 
# 145
static inline short4 make_short4(short x, short y, short z, short w) 
# 146
{ 
# 147
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 148
} 
# 150
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 151
{ 
# 152
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 153
} 
# 155
static inline int1 make_int1(int x) 
# 156
{ 
# 157
int1 t; (t.x) = x; return t; 
# 158
} 
# 160
static inline uint1 make_uint1(unsigned x) 
# 161
{ 
# 162
uint1 t; (t.x) = x; return t; 
# 163
} 
# 165
static inline int2 make_int2(int x, int y) 
# 166
{ 
# 167
int2 t; (t.x) = x; (t.y) = y; return t; 
# 168
} 
# 170
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 171
{ 
# 172
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 173
} 
# 175
static inline int3 make_int3(int x, int y, int z) 
# 176
{ 
# 177
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 178
} 
# 180
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 181
{ 
# 182
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 183
} 
# 185
static inline int4 make_int4(int x, int y, int z, int w) 
# 186
{ 
# 187
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 188
} 
# 190
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 191
{ 
# 192
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 193
} 
# 195
static inline long1 make_long1(long x) 
# 196
{ 
# 197
long1 t; (t.x) = x; return t; 
# 198
} 
# 200
static inline ulong1 make_ulong1(unsigned long x) 
# 201
{ 
# 202
ulong1 t; (t.x) = x; return t; 
# 203
} 
# 205
static inline long2 make_long2(long x, long y) 
# 206
{ 
# 207
long2 t; (t.x) = x; (t.y) = y; return t; 
# 208
} 
# 210
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 211
{ 
# 212
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 213
} 
# 215
static inline long3 make_long3(long x, long y, long z) 
# 216
{ 
# 217
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 218
} 
# 220
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 221
{ 
# 222
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 223
} 
# 225
static inline long4 make_long4(long x, long y, long z, long w) 
# 226
{ 
# 227
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 228
} 
# 230
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 231
{ 
# 232
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 233
} 
# 235
static inline float1 make_float1(float x) 
# 236
{ 
# 237
float1 t; (t.x) = x; return t; 
# 238
} 
# 240
static inline float2 make_float2(float x, float y) 
# 241
{ 
# 242
float2 t; (t.x) = x; (t.y) = y; return t; 
# 243
} 
# 245
static inline float3 make_float3(float x, float y, float z) 
# 246
{ 
# 247
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 248
} 
# 250
static inline float4 make_float4(float x, float y, float z, float w) 
# 251
{ 
# 252
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 253
} 
# 255
static inline longlong1 make_longlong1(long long x) 
# 256
{ 
# 257
longlong1 t; (t.x) = x; return t; 
# 258
} 
# 260
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 261
{ 
# 262
ulonglong1 t; (t.x) = x; return t; 
# 263
} 
# 265
static inline longlong2 make_longlong2(long long x, long long y) 
# 266
{ 
# 267
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 268
} 
# 270
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 271
{ 
# 272
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 273
} 
# 275
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 276
{ 
# 277
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 278
} 
# 280
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 281
{ 
# 282
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 283
} 
# 285
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 286
{ 
# 287
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 288
} 
# 290
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 291
{ 
# 292
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 293
} 
# 295
static inline double1 make_double1(double x) 
# 296
{ 
# 297
double1 t; (t.x) = x; return t; 
# 298
} 
# 300
static inline double2 make_double2(double x, double y) 
# 301
{ 
# 302
double2 t; (t.x) = x; (t.y) = y; return t; 
# 303
} 
# 305
static inline double3 make_double3(double x, double y, double z) 
# 306
{ 
# 307
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 308
} 
# 310
static inline double4 make_double4(double x, double y, double z, double w) 
# 311
{ 
# 312
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 313
} 
# 27 "/usr/include/string.h" 3
extern "C" {
# 42
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 43
 __attribute((__nonnull__(1, 2))); 
# 46
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 47
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 62
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 65
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 66
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 70
extern "C++" {
# 72
extern void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 73
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 74
extern const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 75
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 90
}
# 101
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 103
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 104
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 114
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 115
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 125
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 126
 __attribute((__nonnull__(1, 2))); 
# 128
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 130
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 136
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 137
 __attribute((__nonnull__(1, 2))); 
# 140
extern int strcmp(const char * __s1, const char * __s2) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 143
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 144
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern int strcoll(const char * __s1, const char * __s2) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 150
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 152
 __attribute((__nonnull__(2))); 
# 39 "/usr/include/xlocale.h" 3
typedef 
# 27
struct __locale_struct { 
# 30
struct __locale_data *__locales[13]; 
# 33
const unsigned short *__ctype_b; 
# 34
const int *__ctype_tolower; 
# 35
const int *__ctype_toupper; 
# 38
const char *__names[13]; 
# 39
} *__locale_t; 
# 42
typedef __locale_t locale_t; 
# 162 "/usr/include/string.h" 3
extern int strcoll_l(const char * __s1, const char * __s2, __locale_t __l) throw()
# 163
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 165
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, __locale_t __l) throw()
# 166
 __attribute((__nonnull__(2, 4))); 
# 171
extern char *strdup(const char * __s) throw()
# 172
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 179
extern char *strndup(const char * __string, size_t __n) throw()
# 180
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 209
extern "C++" {
# 211
extern char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 212
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 213
extern const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 214
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 229
}
# 236
extern "C++" {
# 238
extern char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 239
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 240
extern const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 241
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 256
}
# 267
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 268
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 269
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 270
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 280
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 281
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 284
extern size_t strspn(const char * __s, const char * __accept) throw()
# 285
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 288
extern "C++" {
# 290
extern char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 291
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 292
extern const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 293
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 308
}
# 315
extern "C++" {
# 317
extern char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 318
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 319
extern const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 320
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 335
}
# 343
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 344
 __attribute((__nonnull__(2))); 
# 349
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 352
 __attribute((__nonnull__(2, 3))); 
# 354
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 356
 __attribute((__nonnull__(2, 3))); 
# 362
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 363
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 364
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 366
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 377
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 379
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 383
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 385
 __attribute((__nonnull__(1, 2))); 
# 386
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 388
 __attribute((__nonnull__(1, 2))); 
# 394
extern size_t strlen(const char * __s) throw()
# 395
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 401
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 402
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 408
extern char *strerror(int __errnum) throw(); 
# 433
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 434
 __attribute((__nonnull__(2))); 
# 440
extern char *strerror_l(int __errnum, __locale_t __l) throw(); 
# 446
extern void __bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 450
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 454
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 457
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 458
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 462
extern "C++" {
# 464
extern char *index(char * __s, int __c) throw() __asm__("index")
# 465
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 466
extern const char *index(const char * __s, int __c) throw() __asm__("index")
# 467
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 482
}
# 490
extern "C++" {
# 492
extern char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 493
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 494
extern const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 495
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 510
}
# 518
extern int ffs(int __i) throw() __attribute((const)); 
# 523
extern int ffsl(long __l) throw() __attribute((const)); 
# 524
__extension__ extern int ffsll(long long __ll) throw()
# 525
 __attribute((const)); 
# 529
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 530
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 533
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 534
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 540
extern int strcasecmp_l(const char * __s1, const char * __s2, __locale_t __loc) throw()
# 542
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 544
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, __locale_t __loc) throw()
# 546
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 552
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 554
 __attribute((__nonnull__(1, 2))); 
# 559
extern char *strsignal(int __sig) throw(); 
# 562
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 563
 __attribute((__nonnull__(1, 2))); 
# 564
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 565
 __attribute((__nonnull__(1, 2))); 
# 569
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 571
 __attribute((__nonnull__(1, 2))); 
# 572
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 574
 __attribute((__nonnull__(1, 2))); 
# 579
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 580
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 583
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 586
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 594
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 595
 __attribute((__nonnull__(1))); 
# 596
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 597
 __attribute((__nonnull__(1))); 
# 658
}
# 29 "/usr/include/time.h" 3
extern "C" {
# 30 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef unsigned char __u_char; 
# 31
typedef unsigned short __u_short; 
# 32
typedef unsigned __u_int; 
# 33
typedef unsigned long __u_long; 
# 36
typedef signed char __int8_t; 
# 37
typedef unsigned char __uint8_t; 
# 38
typedef signed short __int16_t; 
# 39
typedef unsigned short __uint16_t; 
# 40
typedef signed int __int32_t; 
# 41
typedef unsigned __uint32_t; 
# 43
typedef signed long __int64_t; 
# 44
typedef unsigned long __uint64_t; 
# 52
typedef long __quad_t; 
# 53
typedef unsigned long __u_quad_t; 
# 124
typedef unsigned long __dev_t; 
# 125
typedef unsigned __uid_t; 
# 126
typedef unsigned __gid_t; 
# 127
typedef unsigned long __ino_t; 
# 128
typedef unsigned long __ino64_t; 
# 129
typedef unsigned __mode_t; 
# 130
typedef unsigned long __nlink_t; 
# 131
typedef long __off_t; 
# 132
typedef long __off64_t; 
# 133
typedef int __pid_t; 
# 134
typedef struct { int __val[2]; } __fsid_t; 
# 135
typedef long __clock_t; 
# 136
typedef unsigned long __rlim_t; 
# 137
typedef unsigned long __rlim64_t; 
# 138
typedef unsigned __id_t; 
# 139
typedef long __time_t; 
# 140
typedef unsigned __useconds_t; 
# 141
typedef long __suseconds_t; 
# 143
typedef int __daddr_t; 
# 144
typedef int __key_t; 
# 147
typedef int __clockid_t; 
# 150
typedef void *__timer_t; 
# 153
typedef long __blksize_t; 
# 158
typedef long __blkcnt_t; 
# 159
typedef long __blkcnt64_t; 
# 162
typedef unsigned long __fsblkcnt_t; 
# 163
typedef unsigned long __fsblkcnt64_t; 
# 166
typedef unsigned long __fsfilcnt_t; 
# 167
typedef unsigned long __fsfilcnt64_t; 
# 170
typedef long __fsword_t; 
# 172
typedef long __ssize_t; 
# 175
typedef long __syscall_slong_t; 
# 177
typedef unsigned long __syscall_ulong_t; 
# 181
typedef __off64_t __loff_t; 
# 182
typedef __quad_t *__qaddr_t; 
# 183
typedef char *__caddr_t; 
# 186
typedef long __intptr_t; 
# 189
typedef unsigned __socklen_t; 
# 30 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
struct timeval { 
# 32
__time_t tv_sec; 
# 33
__suseconds_t tv_usec; 
# 34
}; 
# 25 "/usr/include/x86_64-linux-gnu/bits/timex.h" 3
struct timex { 
# 27
unsigned modes; 
# 28
__syscall_slong_t offset; 
# 29
__syscall_slong_t freq; 
# 30
__syscall_slong_t maxerror; 
# 31
__syscall_slong_t esterror; 
# 32
int status; 
# 33
__syscall_slong_t constant; 
# 34
__syscall_slong_t precision; 
# 35
__syscall_slong_t tolerance; 
# 36
timeval time; 
# 37
__syscall_slong_t tick; 
# 38
__syscall_slong_t ppsfreq; 
# 39
__syscall_slong_t jitter; 
# 40
int shift; 
# 41
__syscall_slong_t stabil; 
# 42
__syscall_slong_t jitcnt; 
# 43
__syscall_slong_t calcnt; 
# 44
__syscall_slong_t errcnt; 
# 45
__syscall_slong_t stbcnt; 
# 47
int tai; 
# 50
int:32; int:32; int:32; int:32; 
# 51
int:32; int:32; int:32; int:32; 
# 52
int:32; int:32; int:32; 
# 53
}; 
# 90 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
extern "C" {
# 93
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 95
}
# 59 "/usr/include/time.h" 3
typedef __clock_t clock_t; 
# 75
typedef __time_t time_t; 
# 91
typedef __clockid_t clockid_t; 
# 103
typedef __timer_t timer_t; 
# 120
struct timespec { 
# 122
__time_t tv_sec; 
# 123
__syscall_slong_t tv_nsec; 
# 124
}; 
# 133
struct tm { 
# 135
int tm_sec; 
# 136
int tm_min; 
# 137
int tm_hour; 
# 138
int tm_mday; 
# 139
int tm_mon; 
# 140
int tm_year; 
# 141
int tm_wday; 
# 142
int tm_yday; 
# 143
int tm_isdst; 
# 146
long tm_gmtoff; 
# 147
const char *tm_zone; 
# 152
}; 
# 161
struct itimerspec { 
# 163
timespec it_interval; 
# 164
timespec it_value; 
# 165
}; 
# 168
struct sigevent; 
# 174
typedef __pid_t pid_t; 
# 189
extern clock_t clock() throw(); 
# 192
extern time_t time(time_t * __timer) throw(); 
# 195
extern double difftime(time_t __time1, time_t __time0) throw()
# 196
 __attribute((const)); 
# 199
extern time_t mktime(tm * __tp) throw(); 
# 205
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 213
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 223
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, __locale_t __loc) throw(); 
# 230
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, __locale_t __loc) throw(); 
# 239
extern tm *gmtime(const time_t * __timer) throw(); 
# 243
extern tm *localtime(const time_t * __timer) throw(); 
# 249
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 254
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 261
extern char *asctime(const tm * __tp) throw(); 
# 264
extern char *ctime(const time_t * __timer) throw(); 
# 272
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 276
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 282
extern char *__tzname[2]; 
# 283
extern int __daylight; 
# 284
extern long __timezone; 
# 289
extern char *tzname[2]; 
# 293
extern void tzset() throw(); 
# 297
extern int daylight; 
# 298
extern long timezone; 
# 304
extern int stime(const time_t * __when) throw(); 
# 319
extern time_t timegm(tm * __tp) throw(); 
# 322
extern time_t timelocal(tm * __tp) throw(); 
# 325
extern int dysize(int __year) throw() __attribute((const)); 
# 334
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 339
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 342
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 345
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 353
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 358
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 363
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 368
extern int timer_delete(timer_t __timerid) throw(); 
# 371
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 376
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 380
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 386
extern int timespec_get(timespec * __ts, int __base) throw()
# 387
 __attribute((__nonnull__(1))); 
# 403
extern int getdate_err; 
# 412
extern tm *getdate(const char * __string); 
# 426
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 430
}
# 68 "/usr/local/cuda-8.0/include/common_functions.h"
extern "C" {
# 71
extern clock_t clock() throw(); 
# 72
extern void *memset(void *, int, size_t) throw(); 
# 73
extern void *memcpy(void *, const void *, size_t) throw(); 
# 75
}
# 93 "/usr/local/cuda-8.0/include/math_functions.h"
extern "C" {
# 164
extern int abs(int) throw(); 
# 165
extern long labs(long) throw(); 
# 166
extern long long llabs(long long) throw(); 
# 216
extern double fabs(double x) throw(); 
# 257
extern float fabsf(float x) throw(); 
# 261
extern inline int min(int, int); 
# 263
extern inline unsigned umin(unsigned, unsigned); 
# 264
extern inline long long llmin(long long, long long); 
# 265
extern inline unsigned long long ullmin(unsigned long long, unsigned long long); 
# 286
extern float fminf(float x, float y) throw(); 
# 306
extern double fmin(double x, double y) throw(); 
# 313
extern inline int max(int, int); 
# 315
extern inline unsigned umax(unsigned, unsigned); 
# 316
extern inline long long llmax(long long, long long); 
# 317
extern inline unsigned long long ullmax(unsigned long long, unsigned long long); 
# 338
extern float fmaxf(float x, float y) throw(); 
# 358
extern double fmax(double, double) throw(); 
# 402
extern double sin(double x) throw(); 
# 435
extern double cos(double x) throw(); 
# 454
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 470
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 515
extern double tan(double x) throw(); 
# 584
extern double sqrt(double x) throw(); 
# 656
extern double rsqrt(double x); 
# 726
extern float rsqrtf(float x); 
# 782
extern double log2(double x) throw(); 
# 807
extern double exp2(double x) throw(); 
# 832
extern float exp2f(float x) throw(); 
# 859
extern double exp10(double x) throw(); 
# 882
extern float exp10f(float x) throw(); 
# 928
extern double expm1(double x) throw(); 
# 973
extern float expm1f(float x) throw(); 
# 1028
extern float log2f(float x) throw(); 
# 1082
extern double log10(double x) throw(); 
# 1153
extern double log(double x) throw(); 
# 1247
extern double log1p(double x) throw(); 
# 1344
extern float log1pf(float x) throw(); 
# 1419
extern double floor(double x) throw(); 
# 1458
extern double exp(double x) throw(); 
# 1489
extern double cosh(double x) throw(); 
# 1519
extern double sinh(double x) throw(); 
# 1549
extern double tanh(double x) throw(); 
# 1584
extern double acosh(double x) throw(); 
# 1622
extern float acoshf(float x) throw(); 
# 1638
extern double asinh(double x) throw(); 
# 1654
extern float asinhf(float x) throw(); 
# 1708
extern double atanh(double x) throw(); 
# 1762
extern float atanhf(float x) throw(); 
# 1821
extern double ldexp(double x, int exp) throw(); 
# 1877
extern float ldexpf(float x, int exp) throw(); 
# 1929
extern double logb(double x) throw(); 
# 1984
extern float logbf(float x) throw(); 
# 2014
extern int ilogb(double x) throw(); 
# 2044
extern int ilogbf(float x) throw(); 
# 2120
extern double scalbn(double x, int n) throw(); 
# 2196
extern float scalbnf(float x, int n) throw(); 
# 2272
extern double scalbln(double x, long n) throw(); 
# 2348
extern float scalblnf(float x, long n) throw(); 
# 2426
extern double frexp(double x, int * nptr) throw(); 
# 2501
extern float frexpf(float x, int * nptr) throw(); 
# 2515
extern double round(double x) throw(); 
# 2532
extern float roundf(float x) throw(); 
# 2550
extern long lround(double x) throw(); 
# 2568
extern long lroundf(float x) throw(); 
# 2586
extern long long llround(double x) throw(); 
# 2604
extern long long llroundf(float x) throw(); 
# 2656
extern float rintf(float x) throw(); 
# 2672
extern long lrint(double x) throw(); 
# 2688
extern long lrintf(float x) throw(); 
# 2704
extern long long llrint(double x) throw(); 
# 2720
extern long long llrintf(float x) throw(); 
# 2773
extern double nearbyint(double x) throw(); 
# 2826
extern float nearbyintf(float x) throw(); 
# 2888
extern double ceil(double x) throw(); 
# 2900
extern double trunc(double x) throw(); 
# 2915
extern float truncf(float x) throw(); 
# 2941
extern double fdim(double x, double y) throw(); 
# 2967
extern float fdimf(float x, float y) throw(); 
# 3003
extern double atan2(double y, double x) throw(); 
# 3034
extern double atan(double x) throw(); 
# 3057
extern double acos(double x) throw(); 
# 3089
extern double asin(double x) throw(); 
# 3135
extern double hypot(double x, double y) throw(); 
# 3187
extern double rhypot(double x, double y) throw(); 
# 3233
extern float hypotf(float x, float y) throw(); 
# 3285
extern float rhypotf(float x, float y) throw(); 
# 3332
extern double norm3d(double a, double b, double c) throw(); 
# 3383
extern double rnorm3d(double a, double b, double c) throw(); 
# 3432
extern double norm4d(double a, double b, double c, double d) throw(); 
# 3488
extern double rnorm4d(double a, double b, double c, double d) throw(); 
# 3533
extern double norm(int dim, const double * t) throw(); 
# 3584
extern double rnorm(int dim, const double * t) throw(); 
# 3636
extern float rnormf(int dim, const float * a) throw(); 
# 3680
extern float normf(int dim, const float * a) throw(); 
# 3725
extern float norm3df(float a, float b, float c) throw(); 
# 3776
extern float rnorm3df(float a, float b, float c) throw(); 
# 3825
extern float norm4df(float a, float b, float c, float d) throw(); 
# 3881
extern float rnorm4df(float a, float b, float c, float d) throw(); 
# 3965
extern double cbrt(double x) throw(); 
# 4051
extern float cbrtf(float x) throw(); 
# 4106
extern double rcbrt(double x); 
# 4156
extern float rcbrtf(float x); 
# 4216
extern double sinpi(double x); 
# 4276
extern float sinpif(float x); 
# 4328
extern double cospi(double x); 
# 4380
extern float cospif(float x); 
# 4410
extern void sincospi(double x, double * sptr, double * cptr); 
# 4440
extern void sincospif(float x, float * sptr, float * cptr); 
# 4752
extern double pow(double x, double y) throw(); 
# 4808
extern double modf(double x, double * iptr) throw(); 
# 4867
extern double fmod(double x, double y) throw(); 
# 4953
extern double remainder(double x, double y) throw(); 
# 5043
extern float remainderf(float x, float y) throw(); 
# 5097
extern double remquo(double x, double y, int * quo) throw(); 
# 5151
extern float remquof(float x, float y, int * quo) throw(); 
# 5192
extern double j0(double x) throw(); 
# 5234
extern float j0f(float x) throw(); 
# 5295
extern double j1(double x) throw(); 
# 5356
extern float j1f(float x) throw(); 
# 5399
extern double jn(int n, double x) throw(); 
# 5442
extern float jnf(int n, float x) throw(); 
# 5494
extern double y0(double x) throw(); 
# 5546
extern float y0f(float x) throw(); 
# 5598
extern double y1(double x) throw(); 
# 5650
extern float y1f(float x) throw(); 
# 5703
extern double yn(int n, double x) throw(); 
# 5756
extern float ynf(int n, float x) throw(); 
# 5783
extern double cyl_bessel_i0(double x) throw(); 
# 5809
extern float cyl_bessel_i0f(float x) throw(); 
# 5836
extern double cyl_bessel_i1(double x) throw(); 
# 5862
extern float cyl_bessel_i1f(float x) throw(); 
# 5945
extern double erf(double x) throw(); 
# 6027
extern float erff(float x) throw(); 
# 6091
extern double erfinv(double y); 
# 6148
extern float erfinvf(float y); 
# 6187
extern double erfc(double x) throw(); 
# 6225
extern float erfcf(float x) throw(); 
# 6353
extern double lgamma(double x) throw(); 
# 6416
extern double erfcinv(double y); 
# 6472
extern float erfcinvf(float y); 
# 6530
extern double normcdfinv(double y); 
# 6588
extern float normcdfinvf(float y); 
# 6631
extern double normcdf(double y); 
# 6674
extern float normcdff(float y); 
# 6749
extern double erfcx(double x); 
# 6824
extern float erfcxf(float x); 
# 6958
extern float lgammaf(float x) throw(); 
# 7067
extern double tgamma(double x) throw(); 
# 7176
extern float tgammaf(float x) throw(); 
# 7189
extern double copysign(double x, double y) throw(); 
# 7202
extern float copysignf(float x, float y) throw(); 
# 7239
extern double nextafter(double x, double y) throw(); 
# 7276
extern float nextafterf(float x, float y) throw(); 
# 7292
extern double nan(const char * tagp) throw(); 
# 7308
extern float nanf(const char * tagp) throw(); 
# 7315
extern int __isinff(float) throw(); 
# 7316
extern int __isnanf(float) throw(); 
# 7326
extern int __finite(double) throw(); 
# 7327
extern int __finitef(float) throw(); 
# 7328
extern int __signbit(double) throw(); 
# 7329
extern int __isnan(double) throw(); 
# 7330
extern int __isinf(double) throw(); 
# 7333
extern int __signbitf(float) throw(); 
# 7492
extern double fma(double x, double y, double z) throw(); 
# 7650
extern float fmaf(float x, float y, float z) throw(); 
# 7661
extern int __signbitl(long double) throw(); 
# 7667
extern int __finitel(long double) throw(); 
# 7668
extern int __isinfl(long double) throw(); 
# 7669
extern int __isnanl(long double) throw(); 
# 7719
extern float acosf(float x) throw(); 
# 7759
extern float asinf(float x) throw(); 
# 7799
extern float atanf(float x) throw(); 
# 7832
extern float atan2f(float y, float x) throw(); 
# 7856
extern float cosf(float x) throw(); 
# 7898
extern float sinf(float x) throw(); 
# 7940
extern float tanf(float x) throw(); 
# 7964
extern float coshf(float x) throw(); 
# 8005
extern float sinhf(float x) throw(); 
# 8035
extern float tanhf(float x) throw(); 
# 8086
extern float logf(float x) throw(); 
# 8136
extern float expf(float x) throw(); 
# 8187
extern float log10f(float x) throw(); 
# 8242
extern float modff(float x, float * iptr) throw(); 
# 8550
extern float powf(float x, float y) throw(); 
# 8619
extern float sqrtf(float x) throw(); 
# 8678
extern float ceilf(float x) throw(); 
# 8750
extern float floorf(float x) throw(); 
# 8809
extern float fmodf(float x, float y) throw(); 
# 8823
}
# 28 "/usr/include/math.h" 3
extern "C" {
# 28 "/usr/include/x86_64-linux-gnu/bits/mathdef.h" 3
typedef float float_t; 
# 29
typedef double double_t; 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 56
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 58
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 60
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 63
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 65
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 67
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 72
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 74
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 76
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 81
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 88
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 90
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 92
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 100
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 103
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 106
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 109
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 112
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 115
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 122
extern double pow10(double __x) throw(); extern double __pow10(double __x) throw(); 
# 128
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 131
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 134
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 141
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 144
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 153
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 156
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 162
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 169
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 178
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 181
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 184
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 187
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 192
extern int __isinf(double __value) throw() __attribute((const)); 
# 195
extern int __finite(double __value) throw() __attribute((const)); 
# 208
extern int finite(double __value) throw() __attribute((const)); 
# 211
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 215
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 221
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 228
extern double nan(const char * __tagb) throw() __attribute((const)); extern double __nan(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnan(double __value) throw() __attribute((const)); 
# 247
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 248
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 249
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 250
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 251
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 252
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 259
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 260
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 261
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 268
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 274
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 281
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 289
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 292
extern double nextafter(double __x, double __y) throw() __attribute((const)); extern double __nextafter(double __x, double __y) throw() __attribute((const)); 
# 294
extern double nexttoward(double __x, long double __y) throw() __attribute((const)); extern double __nexttoward(double __x, long double __y) throw() __attribute((const)); 
# 298
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 302
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 306
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 311
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 315
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 319
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 323
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 328
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 335
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 337
__extension__ extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 341
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 343
__extension__ extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 347
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 350
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 353
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 357
extern int __fpclassify(double __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbit(double __value) throw()
# 362
 __attribute((const)); 
# 366
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 375
extern int __issignaling(double __value) throw()
# 376
 __attribute((const)); 
# 383
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 56
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 58
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 60
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 63
extern float cosf(float __x) throw(); 
# 65
extern float sinf(float __x) throw(); 
# 67
extern float tanf(float __x) throw(); 
# 72
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 74
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 76
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 81
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 88
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 90
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 92
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 100
extern float expf(float __x) throw(); 
# 103
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 106
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 109
extern float logf(float __x) throw(); 
# 112
extern float log10f(float __x) throw(); 
# 115
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern float exp10f(float __x) throw(); 
# 122
extern float pow10f(float __x) throw(); extern float __pow10f(float __x) throw(); 
# 128
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 131
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 134
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 141
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 144
extern float log2f(float __x) throw(); 
# 153
extern float powf(float __x, float __y) throw(); 
# 156
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 162
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 169
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 178
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 181
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 184
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 187
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 192
extern int __isinff(float __value) throw() __attribute((const)); 
# 195
extern int __finitef(float __value) throw() __attribute((const)); 
# 204
extern int isinff(float __value) throw() __attribute((const)); 
# 208
extern int finitef(float __value) throw() __attribute((const)); 
# 211
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 215
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 221
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 228
extern float nanf(const char * __tagb) throw() __attribute((const)); extern float __nanf(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnanf(float __value) throw() __attribute((const)); 
# 241
extern int isnanf(float __value) throw() __attribute((const)); 
# 247
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 248
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 249
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 250
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 251
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 252
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 259
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 260
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 261
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 268
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 274
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 281
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 289
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 292
extern float nextafterf(float __x, float __y) throw() __attribute((const)); extern float __nextafterf(float __x, float __y) throw() __attribute((const)); 
# 294
extern float nexttowardf(float __x, long double __y) throw() __attribute((const)); extern float __nexttowardf(float __x, long double __y) throw() __attribute((const)); 
# 298
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 302
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 306
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 311
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 315
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 319
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 323
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 328
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 335
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 337
__extension__ extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 341
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 343
__extension__ extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 347
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 350
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 353
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 357
extern int __fpclassifyf(float __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbitf(float __value) throw()
# 362
 __attribute((const)); 
# 366
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 375
extern int __issignalingf(float __value) throw()
# 376
 __attribute((const)); 
# 383
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 56
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 58
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 60
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 63
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 65
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 67
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 72
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 74
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 76
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 81
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 88
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 90
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 92
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 100
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 103
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 106
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 109
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 112
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 115
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 122
extern long double pow10l(long double __x) throw(); extern long double __pow10l(long double __x) throw(); 
# 128
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 131
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 134
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 141
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 144
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 153
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 156
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 162
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 169
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 178
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 181
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 184
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 187
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 192
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 195
extern int __finitel(long double __value) throw() __attribute((const)); 
# 204
extern int isinfl(long double __value) throw() __attribute((const)); 
# 208
extern int finitel(long double __value) throw() __attribute((const)); 
# 211
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 215
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 221
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 228
extern long double nanl(const char * __tagb) throw() __attribute((const)); extern long double __nanl(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 241
extern int isnanl(long double __value) throw() __attribute((const)); 
# 247
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 248
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 249
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 250
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 251
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 252
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 259
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 260
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 261
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 268
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 274
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 281
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 289
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 292
extern long double nextafterl(long double __x, long double __y) throw() __attribute((const)); extern long double __nextafterl(long double __x, long double __y) throw() __attribute((const)); 
# 294
extern long double nexttowardl(long double __x, long double __y) throw() __attribute((const)); extern long double __nexttowardl(long double __x, long double __y) throw() __attribute((const)); 
# 298
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 302
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 306
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 311
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 315
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 319
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 323
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 328
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 335
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 337
__extension__ extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 341
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 343
__extension__ extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 347
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 350
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 353
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 357
extern int __fpclassifyl(long double __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbitl(long double __value) throw()
# 362
 __attribute((const)); 
# 366
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 375
extern int __issignalingl(long double __value) throw()
# 376
 __attribute((const)); 
# 383
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 168 "/usr/include/math.h" 3
extern int signgam; 
# 210
enum { 
# 211
FP_NAN, 
# 214
FP_INFINITE, 
# 217
FP_ZERO, 
# 220
FP_SUBNORMAL, 
# 223
FP_NORMAL
# 226
}; 
# 354
typedef 
# 348
enum { 
# 349
_IEEE_ = (-1), 
# 350
_SVID_ = 0, 
# 351
_XOPEN_, 
# 352
_POSIX_, 
# 353
_ISOC_
# 354
} _LIB_VERSION_TYPE; 
# 359
extern _LIB_VERSION_TYPE _LIB_VERSION; 
# 370
struct __exception { 
# 375
int type; 
# 376
char *name; 
# 377
double arg1; 
# 378
double arg2; 
# 379
double retval; 
# 380
}; 
# 383
extern int matherr(__exception * __exc) throw(); 
# 534
}
# 34 "/usr/include/stdlib.h" 3
extern "C" {
# 55 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
typedef 
# 51
enum { 
# 52
P_ALL, 
# 53
P_PID, 
# 54
P_PGID
# 55
} idtype_t; 
# 45 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3
static inline unsigned __bswap_32(unsigned __bsx) 
# 46
{ 
# 47
return __builtin_bswap32(__bsx); 
# 48
} 
# 109
static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 110
{ 
# 111
return __builtin_bswap64(__bsx); 
# 112
} 
# 66 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
union wait { 
# 68
int w_status; 
# 70
struct { 
# 72
unsigned __w_termsig:7; 
# 73
unsigned __w_coredump:1; 
# 74
unsigned __w_retcode:8; 
# 75
unsigned:16; 
# 83
} __wait_terminated; 
# 85
struct { 
# 87
unsigned __w_stopval:8; 
# 88
unsigned __w_stopsig:8; 
# 89
unsigned:16; 
# 96
} __wait_stopped; 
# 97
}; 
# 101 "/usr/include/stdlib.h" 3
typedef 
# 98
struct { 
# 99
int quot; 
# 100
int rem; 
# 101
} div_t; 
# 109
typedef 
# 106
struct { 
# 107
long quot; 
# 108
long rem; 
# 109
} ldiv_t; 
# 121
__extension__ typedef 
# 118
struct { 
# 119
long long quot; 
# 120
long long rem; 
# 121
} lldiv_t; 
# 139
extern size_t __ctype_get_mb_cur_max() throw(); 
# 144
extern double atof(const char * __nptr) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 147
extern int atoi(const char * __nptr) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 150
extern long atol(const char * __nptr) throw()
# 151
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 157
__extension__ extern long long atoll(const char * __nptr) throw()
# 158
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 164
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 166
 __attribute((__nonnull__(1))); 
# 172
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 173
 __attribute((__nonnull__(1))); 
# 175
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 177
 __attribute((__nonnull__(1))); 
# 183
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 185
 __attribute((__nonnull__(1))); 
# 187
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 195
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 197
 __attribute((__nonnull__(1))); 
# 200
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 209
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 211
 __attribute((__nonnull__(1))); 
# 214
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 216
 __attribute((__nonnull__(1))); 
# 239
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 241
 __attribute((__nonnull__(1, 4))); 
# 243
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 246
 __attribute((__nonnull__(1, 4))); 
# 249
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 252
 __attribute((__nonnull__(1, 4))); 
# 255
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 258
 __attribute((__nonnull__(1, 4))); 
# 260
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 262
 __attribute((__nonnull__(1, 3))); 
# 264
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 266
 __attribute((__nonnull__(1, 3))); 
# 268
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 271
 __attribute((__nonnull__(1, 3))); 
# 305
extern char *l64a(long __n) throw(); 
# 308
extern long a64l(const char * __s) throw()
# 309
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 44
typedef __loff_t loff_t; 
# 48
typedef __ino_t ino_t; 
# 55
typedef __ino64_t ino64_t; 
# 60
typedef __dev_t dev_t; 
# 65
typedef __gid_t gid_t; 
# 70
typedef __mode_t mode_t; 
# 75
typedef __nlink_t nlink_t; 
# 80
typedef __uid_t uid_t; 
# 86
typedef __off_t off_t; 
# 93
typedef __off64_t off64_t; 
# 104
typedef __id_t id_t; 
# 109
typedef __ssize_t ssize_t; 
# 115
typedef __daddr_t daddr_t; 
# 116
typedef __caddr_t caddr_t; 
# 122
typedef __key_t key_t; 
# 136
typedef __useconds_t useconds_t; 
# 140
typedef __suseconds_t suseconds_t; 
# 150
typedef unsigned long ulong; 
# 151
typedef unsigned short ushort; 
# 152
typedef unsigned uint; 
# 194
typedef signed char int8_t __attribute((__mode__(__QI__))); 
# 195
typedef short int16_t __attribute((__mode__(__HI__))); 
# 196
typedef int int32_t __attribute((__mode__(__SI__))); 
# 197
typedef long int64_t __attribute((__mode__(__DI__))); 
# 200
typedef unsigned char u_int8_t __attribute((__mode__(__QI__))); 
# 201
typedef unsigned short u_int16_t __attribute((__mode__(__HI__))); 
# 202
typedef unsigned u_int32_t __attribute((__mode__(__SI__))); 
# 203
typedef unsigned long u_int64_t __attribute((__mode__(__DI__))); 
# 205
typedef long register_t __attribute((__mode__(__word__))); 
# 22 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
typedef int __sig_atomic_t; 
# 30
typedef 
# 28
struct { 
# 29
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 30
} __sigset_t; 
# 37 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef __sigset_t sigset_t; 
# 54
typedef long __fd_mask; 
# 75
typedef 
# 65
struct { 
# 69
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 75
} fd_set; 
# 82
typedef __fd_mask fd_mask; 
# 96
extern "C" {
# 106
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 118
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 131
}
# 24 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
extern "C" {
# 27
__extension__ extern unsigned gnu_dev_major(unsigned long long __dev) throw()
# 28
 __attribute((const)); 
# 30
__extension__ extern unsigned gnu_dev_minor(unsigned long long __dev) throw()
# 31
 __attribute((const)); 
# 33
__extension__ extern unsigned long long gnu_dev_makedev(unsigned __major, unsigned __minor) throw()
# 35
 __attribute((const)); 
# 58
}
# 228 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 235
typedef __blkcnt_t blkcnt_t; 
# 239
typedef __fsblkcnt_t fsblkcnt_t; 
# 243
typedef __fsfilcnt_t fsfilcnt_t; 
# 262
typedef __blkcnt64_t blkcnt64_t; 
# 263
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 264
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 60 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 63
union pthread_attr_t { 
# 65
char __size[56]; 
# 66
long __align; 
# 67
}; 
# 69
typedef pthread_attr_t pthread_attr_t; 
# 79
typedef 
# 75
struct __pthread_internal_list { 
# 77
__pthread_internal_list *__prev; 
# 78
__pthread_internal_list *__next; 
# 79
} __pthread_list_t; 
# 128
typedef 
# 91
union { 
# 92
struct __pthread_mutex_s { 
# 94
int __lock; 
# 95
unsigned __count; 
# 96
int __owner; 
# 98
unsigned __nusers; 
# 102
int __kind; 
# 104
short __spins; 
# 105
short __elision; 
# 106
__pthread_list_t __list; 
# 125
} __data; 
# 126
char __size[40]; 
# 127
long __align; 
# 128
} pthread_mutex_t; 
# 134
typedef 
# 131
union { 
# 132
char __size[4]; 
# 133
int __align; 
# 134
} pthread_mutexattr_t; 
# 154
typedef 
# 140
union { 
# 142
struct { 
# 143
int __lock; 
# 144
unsigned __futex; 
# 145
__extension__ unsigned long long __total_seq; 
# 146
__extension__ unsigned long long __wakeup_seq; 
# 147
__extension__ unsigned long long __woken_seq; 
# 148
void *__mutex; 
# 149
unsigned __nwaiters; 
# 150
unsigned __broadcast_seq; 
# 151
} __data; 
# 152
char __size[48]; 
# 153
__extension__ long long __align; 
# 154
} pthread_cond_t; 
# 160
typedef 
# 157
union { 
# 158
char __size[4]; 
# 159
int __align; 
# 160
} pthread_condattr_t; 
# 164
typedef unsigned pthread_key_t; 
# 168
typedef int pthread_once_t; 
# 222
typedef 
# 175
union { 
# 178
struct { 
# 179
int __lock; 
# 180
unsigned __nr_readers; 
# 181
unsigned __readers_wakeup; 
# 182
unsigned __writer_wakeup; 
# 183
unsigned __nr_readers_queued; 
# 184
unsigned __nr_writers_queued; 
# 185
int __writer; 
# 186
int __shared; 
# 187
signed char __rwelision; 
# 192
unsigned char __pad1[7]; 
# 195
unsigned long __pad2; 
# 198
unsigned __flags; 
# 200
} __data; 
# 220
char __size[56]; 
# 221
long __align; 
# 222
} pthread_rwlock_t; 
# 228
typedef 
# 225
union { 
# 226
char __size[8]; 
# 227
long __align; 
# 228
} pthread_rwlockattr_t; 
# 234
typedef volatile int pthread_spinlock_t; 
# 243
typedef 
# 240
union { 
# 241
char __size[32]; 
# 242
long __align; 
# 243
} pthread_barrier_t; 
# 249
typedef 
# 246
union { 
# 247
char __size[4]; 
# 248
int __align; 
# 249
} pthread_barrierattr_t; 
# 273 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
}
# 321 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 324
extern void srandom(unsigned __seed) throw(); 
# 330
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 331
 __attribute((__nonnull__(2))); 
# 335
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 343
struct random_data { 
# 345
int32_t *fptr; 
# 346
int32_t *rptr; 
# 347
int32_t *state; 
# 348
int rand_type; 
# 349
int rand_deg; 
# 350
int rand_sep; 
# 351
int32_t *end_ptr; 
# 352
}; 
# 354
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 355
 __attribute((__nonnull__(1, 2))); 
# 357
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 358
 __attribute((__nonnull__(2))); 
# 360
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 363
 __attribute((__nonnull__(2, 4))); 
# 365
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 367
 __attribute((__nonnull__(1, 2))); 
# 374
extern int rand() throw(); 
# 376
extern void srand(unsigned __seed) throw(); 
# 381
extern int rand_r(unsigned * __seed) throw(); 
# 389
extern double drand48() throw(); 
# 390
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 393
extern long lrand48() throw(); 
# 394
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 395
 __attribute((__nonnull__(1))); 
# 398
extern long mrand48() throw(); 
# 399
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 400
 __attribute((__nonnull__(1))); 
# 403
extern void srand48(long __seedval) throw(); 
# 404
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 405
 __attribute((__nonnull__(1))); 
# 406
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 412
struct drand48_data { 
# 414
unsigned short __x[3]; 
# 415
unsigned short __old_x[3]; 
# 416
unsigned short __c; 
# 417
unsigned short __init; 
# 418
__extension__ unsigned long long __a; 
# 420
}; 
# 423
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 424
 __attribute((__nonnull__(1, 2))); 
# 425
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 427
 __attribute((__nonnull__(1, 2))); 
# 430
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 432
 __attribute((__nonnull__(1, 2))); 
# 433
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 436
 __attribute((__nonnull__(1, 2))); 
# 439
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 441
 __attribute((__nonnull__(1, 2))); 
# 442
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 445
 __attribute((__nonnull__(1, 2))); 
# 448
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 449
 __attribute((__nonnull__(2))); 
# 451
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 452
 __attribute((__nonnull__(1, 2))); 
# 454
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 456
 __attribute((__nonnull__(1, 2))); 
# 466
extern void *malloc(size_t __size) throw() __attribute((__malloc__)); 
# 468
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 469
 __attribute((__malloc__)); 
# 480
extern void *realloc(void * __ptr, size_t __size) throw()
# 481
 __attribute((__warn_unused_result__)); 
# 483
extern void free(void * __ptr) throw(); 
# 488
extern void cfree(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 498 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__)); 
# 503
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 504
 __attribute((__nonnull__(1))); 
# 509
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 510
 __attribute((__malloc__)) __attribute((__alloc_size__(2))); 
# 515
extern void abort() throw() __attribute((__noreturn__)); 
# 519
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 524
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 525
 __attribute((__nonnull__(1))); 
# 535
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 536
 __attribute((__nonnull__(1))); 
# 543
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 549
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 557
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 564
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 570
extern char *secure_getenv(const char * __name) throw()
# 571
 __attribute((__nonnull__(1))); 
# 578
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 584
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 585
 __attribute((__nonnull__(2))); 
# 588
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 595
extern int clearenv() throw(); 
# 606
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 619
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 629
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 641
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 651
extern int mkstemps64(char * __template, int __suffixlen)
# 652
 __attribute((__nonnull__(1))); 
# 662
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 673
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 683
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 693
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 694
 __attribute((__nonnull__(1))); 
# 705
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 706
 __attribute((__nonnull__(1))); 
# 716
extern int system(const char * __command); 
# 723
extern char *canonicalize_file_name(const char * __name) throw()
# 724
 __attribute((__nonnull__(1))); 
# 733
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 741
typedef int (*__compar_fn_t)(const void *, const void *); 
# 744
typedef __compar_fn_t comparison_fn_t; 
# 748
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 754
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 756
 __attribute((__nonnull__(1, 2, 5))); 
# 764
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 765
 __attribute((__nonnull__(1, 4))); 
# 767
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 769
 __attribute((__nonnull__(1, 4))); 
# 774
extern int abs(int __x) throw() __attribute((const)); 
# 775
extern long labs(long __x) throw() __attribute((const)); 
# 779
__extension__ extern long long llabs(long long __x) throw()
# 780
 __attribute((const)); 
# 788
extern div_t div(int __numer, int __denom) throw()
# 789
 __attribute((const)); 
# 790
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 791
 __attribute((const)); 
# 796
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 798
 __attribute((const)); 
# 811
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 812
 __attribute((__nonnull__(3, 4))); 
# 817
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 818
 __attribute((__nonnull__(3, 4))); 
# 823
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 824
 __attribute((__nonnull__(3))); 
# 829
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 831
 __attribute((__nonnull__(3, 4))); 
# 832
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 834
 __attribute((__nonnull__(3, 4))); 
# 835
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 836
 __attribute((__nonnull__(3))); 
# 841
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 843
 __attribute((__nonnull__(3, 4, 5))); 
# 844
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 846
 __attribute((__nonnull__(3, 4, 5))); 
# 848
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 851
 __attribute((__nonnull__(3, 4, 5))); 
# 852
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 855
 __attribute((__nonnull__(3, 4, 5))); 
# 862
extern int mblen(const char * __s, size_t __n) throw(); 
# 865
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 869
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 873
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 876
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 887
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 898
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 901
 __attribute((__nonnull__(1, 2, 3))); 
# 907
extern void setkey(const char * __key) throw() __attribute((__nonnull__(1))); 
# 915
extern int posix_openpt(int __oflag); 
# 923
extern int grantpt(int __fd) throw(); 
# 927
extern int unlockpt(int __fd) throw(); 
# 932
extern char *ptsname(int __fd) throw(); 
# 939
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 940
 __attribute((__nonnull__(2))); 
# 943
extern int getpt(); 
# 950
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 951
 __attribute((__nonnull__(1))); 
# 967
}
# 184 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/c++config.h" 3
namespace std { 
# 186
typedef unsigned long size_t; 
# 187
typedef long ptrdiff_t; 
# 190
typedef __decltype((nullptr)) nullptr_t; 
# 192
}
# 68 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 72
template< class _Iterator, class _Container> class __normal_iterator; 
# 76
}
# 78
namespace std __attribute((__visibility__("default"))) { 
# 82
struct __true_type { }; 
# 83
struct __false_type { }; 
# 85
template< bool > 
# 86
struct __truth_type { 
# 87
typedef __false_type __type; }; 
# 90
template<> struct __truth_type< true>  { 
# 91
typedef __true_type __type; }; 
# 95
template< class _Sp, class _Tp> 
# 96
struct __traitor { 
# 98
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 99
typedef typename __truth_type< __value> ::__type __type; 
# 100
}; 
# 103
template< class , class > 
# 104
struct __are_same { 
# 106
enum { __value}; 
# 107
typedef __false_type __type; 
# 108
}; 
# 110
template< class _Tp> 
# 111
struct __are_same< _Tp, _Tp>  { 
# 113
enum { __value = 1}; 
# 114
typedef __true_type __type; 
# 115
}; 
# 118
template< class _Tp> 
# 119
struct __is_void { 
# 121
enum { __value}; 
# 122
typedef __false_type __type; 
# 123
}; 
# 126
template<> struct __is_void< void>  { 
# 128
enum { __value = 1}; 
# 129
typedef __true_type __type; 
# 130
}; 
# 135
template< class _Tp> 
# 136
struct __is_integer { 
# 138
enum { __value}; 
# 139
typedef __false_type __type; 
# 140
}; 
# 146
template<> struct __is_integer< bool>  { 
# 148
enum { __value = 1}; 
# 149
typedef __true_type __type; 
# 150
}; 
# 153
template<> struct __is_integer< char>  { 
# 155
enum { __value = 1}; 
# 156
typedef __true_type __type; 
# 157
}; 
# 160
template<> struct __is_integer< signed char>  { 
# 162
enum { __value = 1}; 
# 163
typedef __true_type __type; 
# 164
}; 
# 167
template<> struct __is_integer< unsigned char>  { 
# 169
enum { __value = 1}; 
# 170
typedef __true_type __type; 
# 171
}; 
# 175
template<> struct __is_integer< wchar_t>  { 
# 177
enum { __value = 1}; 
# 178
typedef __true_type __type; 
# 179
}; 
# 184
template<> struct __is_integer< char16_t>  { 
# 186
enum { __value = 1}; 
# 187
typedef __true_type __type; 
# 188
}; 
# 191
template<> struct __is_integer< char32_t>  { 
# 193
enum { __value = 1}; 
# 194
typedef __true_type __type; 
# 195
}; 
# 199
template<> struct __is_integer< short>  { 
# 201
enum { __value = 1}; 
# 202
typedef __true_type __type; 
# 203
}; 
# 206
template<> struct __is_integer< unsigned short>  { 
# 208
enum { __value = 1}; 
# 209
typedef __true_type __type; 
# 210
}; 
# 213
template<> struct __is_integer< int>  { 
# 215
enum { __value = 1}; 
# 216
typedef __true_type __type; 
# 217
}; 
# 220
template<> struct __is_integer< unsigned>  { 
# 222
enum { __value = 1}; 
# 223
typedef __true_type __type; 
# 224
}; 
# 227
template<> struct __is_integer< long>  { 
# 229
enum { __value = 1}; 
# 230
typedef __true_type __type; 
# 231
}; 
# 234
template<> struct __is_integer< unsigned long>  { 
# 236
enum { __value = 1}; 
# 237
typedef __true_type __type; 
# 238
}; 
# 241
template<> struct __is_integer< long long>  { 
# 243
enum { __value = 1}; 
# 244
typedef __true_type __type; 
# 245
}; 
# 248
template<> struct __is_integer< unsigned long long>  { 
# 250
enum { __value = 1}; 
# 251
typedef __true_type __type; 
# 252
}; 
# 257
template< class _Tp> 
# 258
struct __is_floating { 
# 260
enum { __value}; 
# 261
typedef __false_type __type; 
# 262
}; 
# 266
template<> struct __is_floating< float>  { 
# 268
enum { __value = 1}; 
# 269
typedef __true_type __type; 
# 270
}; 
# 273
template<> struct __is_floating< double>  { 
# 275
enum { __value = 1}; 
# 276
typedef __true_type __type; 
# 277
}; 
# 280
template<> struct __is_floating< long double>  { 
# 282
enum { __value = 1}; 
# 283
typedef __true_type __type; 
# 284
}; 
# 289
template< class _Tp> 
# 290
struct __is_pointer { 
# 292
enum { __value}; 
# 293
typedef __false_type __type; 
# 294
}; 
# 296
template< class _Tp> 
# 297
struct __is_pointer< _Tp *>  { 
# 299
enum { __value = 1}; 
# 300
typedef __true_type __type; 
# 301
}; 
# 306
template< class _Tp> 
# 307
struct __is_normal_iterator { 
# 309
enum { __value}; 
# 310
typedef __false_type __type; 
# 311
}; 
# 313
template< class _Iterator, class _Container> 
# 314
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> >  { 
# 317
enum { __value = 1}; 
# 318
typedef __true_type __type; 
# 319
}; 
# 324
template< class _Tp> 
# 325
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 327
}; 
# 332
template< class _Tp> 
# 333
struct __is_fundamental : public __traitor< __is_void< _Tp> , __is_arithmetic< _Tp> >  { 
# 335
}; 
# 340
template< class _Tp> 
# 341
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 343
}; 
# 348
template< class _Tp> 
# 349
struct __is_char { 
# 351
enum { __value}; 
# 352
typedef __false_type __type; 
# 353
}; 
# 356
template<> struct __is_char< char>  { 
# 358
enum { __value = 1}; 
# 359
typedef __true_type __type; 
# 360
}; 
# 364
template<> struct __is_char< wchar_t>  { 
# 366
enum { __value = 1}; 
# 367
typedef __true_type __type; 
# 368
}; 
# 371
template< class _Tp> 
# 372
struct __is_byte { 
# 374
enum { __value}; 
# 375
typedef __false_type __type; 
# 376
}; 
# 379
template<> struct __is_byte< char>  { 
# 381
enum { __value = 1}; 
# 382
typedef __true_type __type; 
# 383
}; 
# 386
template<> struct __is_byte< signed char>  { 
# 388
enum { __value = 1}; 
# 389
typedef __true_type __type; 
# 390
}; 
# 393
template<> struct __is_byte< unsigned char>  { 
# 395
enum { __value = 1}; 
# 396
typedef __true_type __type; 
# 397
}; 
# 402
template< class _Tp> 
# 403
struct __is_move_iterator { 
# 405
enum { __value}; 
# 406
typedef __false_type __type; 
# 407
}; 
# 410
template< class _Iterator> class move_iterator; 
# 413
template< class _Iterator> 
# 414
struct __is_move_iterator< move_iterator< _Iterator> >  { 
# 416
enum { __value = 1}; 
# 417
typedef __true_type __type; 
# 418
}; 
# 422
}
# 37 "/usr/include/c++/4.8/ext/type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 42
template< bool , class > 
# 43
struct __enable_if { 
# 44
}; 
# 46
template< class _Tp> 
# 47
struct __enable_if< true, _Tp>  { 
# 48
typedef _Tp __type; }; 
# 52
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 53
struct __conditional_type { 
# 54
typedef _Iftrue __type; }; 
# 56
template< class _Iftrue, class _Iffalse> 
# 57
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 58
typedef _Iffalse __type; }; 
# 62
template< class _Tp> 
# 63
struct __add_unsigned { 
# 66
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 69
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 70
}; 
# 73
template<> struct __add_unsigned< char>  { 
# 74
typedef unsigned char __type; }; 
# 77
template<> struct __add_unsigned< signed char>  { 
# 78
typedef unsigned char __type; }; 
# 81
template<> struct __add_unsigned< short>  { 
# 82
typedef unsigned short __type; }; 
# 85
template<> struct __add_unsigned< int>  { 
# 86
typedef unsigned __type; }; 
# 89
template<> struct __add_unsigned< long>  { 
# 90
typedef unsigned long __type; }; 
# 93
template<> struct __add_unsigned< long long>  { 
# 94
typedef unsigned long long __type; }; 
# 98
template<> struct __add_unsigned< bool> ; 
# 101
template<> struct __add_unsigned< wchar_t> ; 
# 105
template< class _Tp> 
# 106
struct __remove_unsigned { 
# 109
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 112
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 113
}; 
# 116
template<> struct __remove_unsigned< char>  { 
# 117
typedef signed char __type; }; 
# 120
template<> struct __remove_unsigned< unsigned char>  { 
# 121
typedef signed char __type; }; 
# 124
template<> struct __remove_unsigned< unsigned short>  { 
# 125
typedef short __type; }; 
# 128
template<> struct __remove_unsigned< unsigned>  { 
# 129
typedef int __type; }; 
# 132
template<> struct __remove_unsigned< unsigned long>  { 
# 133
typedef long __type; }; 
# 136
template<> struct __remove_unsigned< unsigned long long>  { 
# 137
typedef long long __type; }; 
# 141
template<> struct __remove_unsigned< bool> ; 
# 144
template<> struct __remove_unsigned< wchar_t> ; 
# 148
template< class _Type> inline bool 
# 150
__is_null_pointer(_Type *__ptr) 
# 151
{ return __ptr == 0; } 
# 153
template< class _Type> inline bool 
# 155
__is_null_pointer(_Type) 
# 156
{ return false; } 
# 160
template< class _Tp, bool  = std::__is_integer< _Tp> ::__value> 
# 161
struct __promote { 
# 162
typedef double __type; }; 
# 167
template< class _Tp> 
# 168
struct __promote< _Tp, false>  { 
# 169
}; 
# 172
template<> struct __promote< long double>  { 
# 173
typedef long double __type; }; 
# 176
template<> struct __promote< double>  { 
# 177
typedef double __type; }; 
# 180
template<> struct __promote< float>  { 
# 181
typedef float __type; }; 
# 183
template< class _Tp, class _Up, class 
# 184
_Tp2 = typename __promote< _Tp> ::__type, class 
# 185
_Up2 = typename __promote< _Up> ::__type> 
# 186
struct __promote_2 { 
# 188
typedef __typeof__(_Tp2() + _Up2()) __type; 
# 189
}; 
# 191
template< class _Tp, class _Up, class _Vp, class 
# 192
_Tp2 = typename __promote< _Tp> ::__type, class 
# 193
_Up2 = typename __promote< _Up> ::__type, class 
# 194
_Vp2 = typename __promote< _Vp> ::__type> 
# 195
struct __promote_3 { 
# 197
typedef __typeof__((_Tp2() + _Up2()) + _Vp2()) __type; 
# 198
}; 
# 200
template< class _Tp, class _Up, class _Vp, class _Wp, class 
# 201
_Tp2 = typename __promote< _Tp> ::__type, class 
# 202
_Up2 = typename __promote< _Up> ::__type, class 
# 203
_Vp2 = typename __promote< _Vp> ::__type, class 
# 204
_Wp2 = typename __promote< _Wp> ::__type> 
# 205
struct __promote_4 { 
# 207
typedef __typeof__(((_Tp2() + _Up2()) + _Vp2()) + _Wp2()) __type; 
# 208
}; 
# 211
}
# 75 "/usr/include/c++/4.8/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 81
constexpr double abs(double __x) 
# 82
{ return __builtin_fabs(__x); } 
# 87
constexpr float abs(float __x) 
# 88
{ return __builtin_fabsf(__x); } 
# 91
constexpr long double abs(long double __x) 
# 92
{ return __builtin_fabsl(__x); } 
# 95
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
abs(_Tp __x) 
# 100
{ return __builtin_fabs(__x); } 
# 102
using ::acos;
# 106
constexpr float acos(float __x) 
# 107
{ return __builtin_acosf(__x); } 
# 110
constexpr long double acos(long double __x) 
# 111
{ return __builtin_acosl(__x); } 
# 114
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
acos(_Tp __x) 
# 119
{ return __builtin_acos(__x); } 
# 121
using ::asin;
# 125
constexpr float asin(float __x) 
# 126
{ return __builtin_asinf(__x); } 
# 129
constexpr long double asin(long double __x) 
# 130
{ return __builtin_asinl(__x); } 
# 133
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
asin(_Tp __x) 
# 138
{ return __builtin_asin(__x); } 
# 140
using ::atan;
# 144
constexpr float atan(float __x) 
# 145
{ return __builtin_atanf(__x); } 
# 148
constexpr long double atan(long double __x) 
# 149
{ return __builtin_atanl(__x); } 
# 152
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 156
atan(_Tp __x) 
# 157
{ return __builtin_atan(__x); } 
# 159
using ::atan2;
# 163
constexpr float atan2(float __y, float __x) 
# 164
{ return __builtin_atan2f(__y, __x); } 
# 167
constexpr long double atan2(long double __y, long double __x) 
# 168
{ return __builtin_atan2l(__y, __x); } 
# 171
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 174
atan2(_Tp __y, _Up __x) 
# 175
{ 
# 176
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 177
return atan2((__type)__y, (__type)__x); 
# 178
} 
# 180
using ::ceil;
# 184
constexpr float ceil(float __x) 
# 185
{ return __builtin_ceilf(__x); } 
# 188
constexpr long double ceil(long double __x) 
# 189
{ return __builtin_ceill(__x); } 
# 192
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
ceil(_Tp __x) 
# 197
{ return __builtin_ceil(__x); } 
# 199
using ::cos;
# 203
constexpr float cos(float __x) 
# 204
{ return __builtin_cosf(__x); } 
# 207
constexpr long double cos(long double __x) 
# 208
{ return __builtin_cosl(__x); } 
# 211
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cos(_Tp __x) 
# 216
{ return __builtin_cos(__x); } 
# 218
using ::cosh;
# 222
constexpr float cosh(float __x) 
# 223
{ return __builtin_coshf(__x); } 
# 226
constexpr long double cosh(long double __x) 
# 227
{ return __builtin_coshl(__x); } 
# 230
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
cosh(_Tp __x) 
# 235
{ return __builtin_cosh(__x); } 
# 237
using ::exp;
# 241
constexpr float exp(float __x) 
# 242
{ return __builtin_expf(__x); } 
# 245
constexpr long double exp(long double __x) 
# 246
{ return __builtin_expl(__x); } 
# 249
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
exp(_Tp __x) 
# 254
{ return __builtin_exp(__x); } 
# 256
using ::fabs;
# 260
constexpr float fabs(float __x) 
# 261
{ return __builtin_fabsf(__x); } 
# 264
constexpr long double fabs(long double __x) 
# 265
{ return __builtin_fabsl(__x); } 
# 268
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
fabs(_Tp __x) 
# 273
{ return __builtin_fabs(__x); } 
# 275
using ::floor;
# 279
constexpr float floor(float __x) 
# 280
{ return __builtin_floorf(__x); } 
# 283
constexpr long double floor(long double __x) 
# 284
{ return __builtin_floorl(__x); } 
# 287
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 291
floor(_Tp __x) 
# 292
{ return __builtin_floor(__x); } 
# 294
using ::fmod;
# 298
constexpr float fmod(float __x, float __y) 
# 299
{ return __builtin_fmodf(__x, __y); } 
# 302
constexpr long double fmod(long double __x, long double __y) 
# 303
{ return __builtin_fmodl(__x, __y); } 
# 306
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 309
fmod(_Tp __x, _Up __y) 
# 310
{ 
# 311
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 312
return fmod((__type)__x, (__type)__y); 
# 313
} 
# 315
using ::frexp;
# 319
inline float frexp(float __x, int *__exp) 
# 320
{ return __builtin_frexpf(__x, __exp); } 
# 323
inline long double frexp(long double __x, int *__exp) 
# 324
{ return __builtin_frexpl(__x, __exp); } 
# 327
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
frexp(_Tp __x, int *__exp) 
# 332
{ return __builtin_frexp(__x, __exp); } 
# 334
using ::ldexp;
# 338
constexpr float ldexp(float __x, int __exp) 
# 339
{ return __builtin_ldexpf(__x, __exp); } 
# 342
constexpr long double ldexp(long double __x, int __exp) 
# 343
{ return __builtin_ldexpl(__x, __exp); } 
# 346
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
ldexp(_Tp __x, int __exp) 
# 351
{ return __builtin_ldexp(__x, __exp); } 
# 353
using ::log;
# 357
constexpr float log(float __x) 
# 358
{ return __builtin_logf(__x); } 
# 361
constexpr long double log(long double __x) 
# 362
{ return __builtin_logl(__x); } 
# 365
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log(_Tp __x) 
# 370
{ return __builtin_log(__x); } 
# 372
using ::log10;
# 376
constexpr float log10(float __x) 
# 377
{ return __builtin_log10f(__x); } 
# 380
constexpr long double log10(long double __x) 
# 381
{ return __builtin_log10l(__x); } 
# 384
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 388
log10(_Tp __x) 
# 389
{ return __builtin_log10(__x); } 
# 391
using ::modf;
# 395
inline float modf(float __x, float *__iptr) 
# 396
{ return __builtin_modff(__x, __iptr); } 
# 399
inline long double modf(long double __x, long double *__iptr) 
# 400
{ return __builtin_modfl(__x, __iptr); } 
# 403
using ::pow;
# 407
constexpr float pow(float __x, float __y) 
# 408
{ return __builtin_powf(__x, __y); } 
# 411
constexpr long double pow(long double __x, long double __y) 
# 412
{ return __builtin_powl(__x, __y); } 
# 431
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 434
pow(_Tp __x, _Up __y) 
# 435
{ 
# 436
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 437
return pow((__type)__x, (__type)__y); 
# 438
} 
# 440
using ::sin;
# 444
constexpr float sin(float __x) 
# 445
{ return __builtin_sinf(__x); } 
# 448
constexpr long double sin(long double __x) 
# 449
{ return __builtin_sinl(__x); } 
# 452
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sin(_Tp __x) 
# 457
{ return __builtin_sin(__x); } 
# 459
using ::sinh;
# 463
constexpr float sinh(float __x) 
# 464
{ return __builtin_sinhf(__x); } 
# 467
constexpr long double sinh(long double __x) 
# 468
{ return __builtin_sinhl(__x); } 
# 471
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sinh(_Tp __x) 
# 476
{ return __builtin_sinh(__x); } 
# 478
using ::sqrt;
# 482
constexpr float sqrt(float __x) 
# 483
{ return __builtin_sqrtf(__x); } 
# 486
constexpr long double sqrt(long double __x) 
# 487
{ return __builtin_sqrtl(__x); } 
# 490
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
sqrt(_Tp __x) 
# 495
{ return __builtin_sqrt(__x); } 
# 497
using ::tan;
# 501
constexpr float tan(float __x) 
# 502
{ return __builtin_tanf(__x); } 
# 505
constexpr long double tan(long double __x) 
# 506
{ return __builtin_tanl(__x); } 
# 509
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tan(_Tp __x) 
# 514
{ return __builtin_tan(__x); } 
# 516
using ::tanh;
# 520
constexpr float tanh(float __x) 
# 521
{ return __builtin_tanhf(__x); } 
# 524
constexpr long double tanh(long double __x) 
# 525
{ return __builtin_tanhl(__x); } 
# 528
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 532
tanh(_Tp __x) 
# 533
{ return __builtin_tanh(__x); } 
# 536
}
# 555
namespace std __attribute((__visibility__("default"))) { 
# 561
constexpr int fpclassify(float __x) 
# 562
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 563
} 
# 566
constexpr int fpclassify(double __x) 
# 567
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 568
} 
# 571
constexpr int fpclassify(long double __x) 
# 572
{ return __builtin_fpclassify(0, 1, 4, 3, 2, __x); 
# 573
} 
# 575
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 578
fpclassify(_Tp __x) 
# 579
{ return (__x != 0) ? 4 : 2; } 
# 582
constexpr bool isfinite(float __x) 
# 583
{ return __builtin_isfinite(__x); } 
# 586
constexpr bool isfinite(double __x) 
# 587
{ return __builtin_isfinite(__x); } 
# 590
constexpr bool isfinite(long double __x) 
# 591
{ return __builtin_isfinite(__x); } 
# 593
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 596
isfinite(_Tp __x) 
# 597
{ return true; } 
# 600
constexpr bool isinf(float __x) 
# 601
{ return __builtin_isinf(__x); } 
# 604
constexpr bool isinf(double __x) 
# 605
{ return __builtin_isinf(__x); } 
# 608
constexpr bool isinf(long double __x) 
# 609
{ return __builtin_isinf(__x); } 
# 611
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 614
isinf(_Tp __x) 
# 615
{ return false; } 
# 618
constexpr bool isnan(float __x) 
# 619
{ return __builtin_isnan(__x); } 
# 622
constexpr bool isnan(double __x) 
# 623
{ return __builtin_isnan(__x); } 
# 626
constexpr bool isnan(long double __x) 
# 627
{ return __builtin_isnan(__x); } 
# 629
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 632
isnan(_Tp __x) 
# 633
{ return false; } 
# 636
constexpr bool isnormal(float __x) 
# 637
{ return __builtin_isnormal(__x); } 
# 640
constexpr bool isnormal(double __x) 
# 641
{ return __builtin_isnormal(__x); } 
# 644
constexpr bool isnormal(long double __x) 
# 645
{ return __builtin_isnormal(__x); } 
# 647
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 650
isnormal(_Tp __x) 
# 651
{ return (__x != 0) ? true : false; } 
# 654
constexpr bool signbit(float __x) 
# 655
{ return __builtin_signbit(__x); } 
# 658
constexpr bool signbit(double __x) 
# 659
{ return __builtin_signbit(__x); } 
# 662
constexpr bool signbit(long double __x) 
# 663
{ return __builtin_signbit(__x); } 
# 665
template< class _Tp> constexpr typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, bool> ::__type 
# 668
signbit(_Tp __x) 
# 669
{ return (__x < 0) ? true : false; } 
# 672
constexpr bool isgreater(float __x, float __y) 
# 673
{ return __builtin_isgreater(__x, __y); } 
# 676
constexpr bool isgreater(double __x, double __y) 
# 677
{ return __builtin_isgreater(__x, __y); } 
# 680
constexpr bool isgreater(long double __x, long double __y) 
# 681
{ return __builtin_isgreater(__x, __y); } 
# 683
template< class _Tp, class _Up> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 687
isgreater(_Tp __x, _Up __y) 
# 688
{ 
# 689
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 690
return __builtin_isgreater((__type)__x, (__type)__y); 
# 691
} 
# 694
constexpr bool isgreaterequal(float __x, float __y) 
# 695
{ return __builtin_isgreaterequal(__x, __y); } 
# 698
constexpr bool isgreaterequal(double __x, double __y) 
# 699
{ return __builtin_isgreaterequal(__x, __y); } 
# 702
constexpr bool isgreaterequal(long double __x, long double __y) 
# 703
{ return __builtin_isgreaterequal(__x, __y); } 
# 705
template< class _Tp, class _Up> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 709
isgreaterequal(_Tp __x, _Up __y) 
# 710
{ 
# 711
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 712
return __builtin_isgreaterequal((__type)__x, (__type)__y); 
# 713
} 
# 716
constexpr bool isless(float __x, float __y) 
# 717
{ return __builtin_isless(__x, __y); } 
# 720
constexpr bool isless(double __x, double __y) 
# 721
{ return __builtin_isless(__x, __y); } 
# 724
constexpr bool isless(long double __x, long double __y) 
# 725
{ return __builtin_isless(__x, __y); } 
# 727
template< class _Tp, class _Up> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 731
isless(_Tp __x, _Up __y) 
# 732
{ 
# 733
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 734
return __builtin_isless((__type)__x, (__type)__y); 
# 735
} 
# 738
constexpr bool islessequal(float __x, float __y) 
# 739
{ return __builtin_islessequal(__x, __y); } 
# 742
constexpr bool islessequal(double __x, double __y) 
# 743
{ return __builtin_islessequal(__x, __y); } 
# 746
constexpr bool islessequal(long double __x, long double __y) 
# 747
{ return __builtin_islessequal(__x, __y); } 
# 749
template< class _Tp, class _Up> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 753
islessequal(_Tp __x, _Up __y) 
# 754
{ 
# 755
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 756
return __builtin_islessequal((__type)__x, (__type)__y); 
# 757
} 
# 760
constexpr bool islessgreater(float __x, float __y) 
# 761
{ return __builtin_islessgreater(__x, __y); } 
# 764
constexpr bool islessgreater(double __x, double __y) 
# 765
{ return __builtin_islessgreater(__x, __y); } 
# 768
constexpr bool islessgreater(long double __x, long double __y) 
# 769
{ return __builtin_islessgreater(__x, __y); } 
# 771
template< class _Tp, class _Up> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 775
islessgreater(_Tp __x, _Up __y) 
# 776
{ 
# 777
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 778
return __builtin_islessgreater((__type)__x, (__type)__y); 
# 779
} 
# 782
constexpr bool isunordered(float __x, float __y) 
# 783
{ return __builtin_isunordered(__x, __y); } 
# 786
constexpr bool isunordered(double __x, double __y) 
# 787
{ return __builtin_isunordered(__x, __y); } 
# 790
constexpr bool isunordered(long double __x, long double __y) 
# 791
{ return __builtin_isunordered(__x, __y); } 
# 793
template< class _Tp, class _Up> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value && __is_arithmetic< _Up> ::__value, bool> ::__type 
# 797
isunordered(_Tp __x, _Up __y) 
# 798
{ 
# 799
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 800
return __builtin_isunordered((__type)__x, (__type)__y); 
# 801
} 
# 917
}
# 1032
namespace std __attribute((__visibility__("default"))) { 
# 1037
using ::double_t;
# 1038
using ::float_t;
# 1041
using ::acosh;
# 1042
using ::acoshf;
# 1043
using ::acoshl;
# 1045
using ::asinh;
# 1046
using ::asinhf;
# 1047
using ::asinhl;
# 1049
using ::atanh;
# 1050
using ::atanhf;
# 1051
using ::atanhl;
# 1053
using ::cbrt;
# 1054
using ::cbrtf;
# 1055
using ::cbrtl;
# 1057
using ::copysign;
# 1058
using ::copysignf;
# 1059
using ::copysignl;
# 1061
using ::erf;
# 1062
using ::erff;
# 1063
using ::erfl;
# 1065
using ::erfc;
# 1066
using ::erfcf;
# 1067
using ::erfcl;
# 1069
using ::exp2;
# 1070
using ::exp2f;
# 1071
using ::exp2l;
# 1073
using ::expm1;
# 1074
using ::expm1f;
# 1075
using ::expm1l;
# 1077
using ::fdim;
# 1078
using ::fdimf;
# 1079
using ::fdiml;
# 1081
using ::fma;
# 1082
using ::fmaf;
# 1083
using ::fmal;
# 1085
using ::fmax;
# 1086
using ::fmaxf;
# 1087
using ::fmaxl;
# 1089
using ::fmin;
# 1090
using ::fminf;
# 1091
using ::fminl;
# 1093
using ::hypot;
# 1094
using ::hypotf;
# 1095
using ::hypotl;
# 1097
using ::ilogb;
# 1098
using ::ilogbf;
# 1099
using ::ilogbl;
# 1101
using ::lgamma;
# 1102
using ::lgammaf;
# 1103
using ::lgammal;
# 1105
using ::llrint;
# 1106
using ::llrintf;
# 1107
using ::llrintl;
# 1109
using ::llround;
# 1110
using ::llroundf;
# 1111
using ::llroundl;
# 1113
using ::log1p;
# 1114
using ::log1pf;
# 1115
using ::log1pl;
# 1117
using ::log2;
# 1118
using ::log2f;
# 1119
using ::log2l;
# 1121
using ::logb;
# 1122
using ::logbf;
# 1123
using ::logbl;
# 1125
using ::lrint;
# 1126
using ::lrintf;
# 1127
using ::lrintl;
# 1129
using ::lround;
# 1130
using ::lroundf;
# 1131
using ::lroundl;
# 1133
using ::nan;
# 1134
using ::nanf;
# 1135
using ::nanl;
# 1137
using ::nearbyint;
# 1138
using ::nearbyintf;
# 1139
using ::nearbyintl;
# 1141
using ::nextafter;
# 1142
using ::nextafterf;
# 1143
using ::nextafterl;
# 1145
using ::nexttoward;
# 1146
using ::nexttowardf;
# 1147
using ::nexttowardl;
# 1149
using ::remainder;
# 1150
using ::remainderf;
# 1151
using ::remainderl;
# 1153
using ::remquo;
# 1154
using ::remquof;
# 1155
using ::remquol;
# 1157
using ::rint;
# 1158
using ::rintf;
# 1159
using ::rintl;
# 1161
using ::round;
# 1162
using ::roundf;
# 1163
using ::roundl;
# 1165
using ::scalbln;
# 1166
using ::scalblnf;
# 1167
using ::scalblnl;
# 1169
using ::scalbn;
# 1170
using ::scalbnf;
# 1171
using ::scalbnl;
# 1173
using ::tgamma;
# 1174
using ::tgammaf;
# 1175
using ::tgammal;
# 1177
using ::trunc;
# 1178
using ::truncf;
# 1179
using ::truncl;
# 1183
constexpr float acosh(float __x) 
# 1184
{ return __builtin_acoshf(__x); } 
# 1187
constexpr long double acosh(long double __x) 
# 1188
{ return __builtin_acoshl(__x); } 
# 1190
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1193
acosh(_Tp __x) 
# 1194
{ return __builtin_acosh(__x); } 
# 1197
constexpr float asinh(float __x) 
# 1198
{ return __builtin_asinhf(__x); } 
# 1201
constexpr long double asinh(long double __x) 
# 1202
{ return __builtin_asinhl(__x); } 
# 1204
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1207
asinh(_Tp __x) 
# 1208
{ return __builtin_asinh(__x); } 
# 1211
constexpr float atanh(float __x) 
# 1212
{ return __builtin_atanhf(__x); } 
# 1215
constexpr long double atanh(long double __x) 
# 1216
{ return __builtin_atanhl(__x); } 
# 1218
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1221
atanh(_Tp __x) 
# 1222
{ return __builtin_atanh(__x); } 
# 1225
constexpr float cbrt(float __x) 
# 1226
{ return __builtin_cbrtf(__x); } 
# 1229
constexpr long double cbrt(long double __x) 
# 1230
{ return __builtin_cbrtl(__x); } 
# 1232
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1235
cbrt(_Tp __x) 
# 1236
{ return __builtin_cbrt(__x); } 
# 1239
constexpr float copysign(float __x, float __y) 
# 1240
{ return __builtin_copysignf(__x, __y); } 
# 1243
constexpr long double copysign(long double __x, long double __y) 
# 1244
{ return __builtin_copysignl(__x, __y); } 
# 1246
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1248
copysign(_Tp __x, _Up __y) 
# 1249
{ 
# 1250
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1251
return copysign((__type)__x, (__type)__y); 
# 1252
} 
# 1255
constexpr float erf(float __x) 
# 1256
{ return __builtin_erff(__x); } 
# 1259
constexpr long double erf(long double __x) 
# 1260
{ return __builtin_erfl(__x); } 
# 1262
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1265
erf(_Tp __x) 
# 1266
{ return __builtin_erf(__x); } 
# 1269
constexpr float erfc(float __x) 
# 1270
{ return __builtin_erfcf(__x); } 
# 1273
constexpr long double erfc(long double __x) 
# 1274
{ return __builtin_erfcl(__x); } 
# 1276
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1279
erfc(_Tp __x) 
# 1280
{ return __builtin_erfc(__x); } 
# 1283
constexpr float exp2(float __x) 
# 1284
{ return __builtin_exp2f(__x); } 
# 1287
constexpr long double exp2(long double __x) 
# 1288
{ return __builtin_exp2l(__x); } 
# 1290
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1293
exp2(_Tp __x) 
# 1294
{ return __builtin_exp2(__x); } 
# 1297
constexpr float expm1(float __x) 
# 1298
{ return __builtin_expm1f(__x); } 
# 1301
constexpr long double expm1(long double __x) 
# 1302
{ return __builtin_expm1l(__x); } 
# 1304
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1307
expm1(_Tp __x) 
# 1308
{ return __builtin_expm1(__x); } 
# 1311
constexpr float fdim(float __x, float __y) 
# 1312
{ return __builtin_fdimf(__x, __y); } 
# 1315
constexpr long double fdim(long double __x, long double __y) 
# 1316
{ return __builtin_fdiml(__x, __y); } 
# 1318
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1320
fdim(_Tp __x, _Up __y) 
# 1321
{ 
# 1322
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1323
return fdim((__type)__x, (__type)__y); 
# 1324
} 
# 1327
constexpr float fma(float __x, float __y, float __z) 
# 1328
{ return __builtin_fmaf(__x, __y, __z); } 
# 1331
constexpr long double fma(long double __x, long double __y, long double __z) 
# 1332
{ return __builtin_fmal(__x, __y, __z); } 
# 1334
template< class _Tp, class _Up, class _Vp> constexpr typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type 
# 1336
fma(_Tp __x, _Up __y, _Vp __z) 
# 1337
{ 
# 1338
typedef typename __gnu_cxx::__promote_3< _Tp, _Up, _Vp> ::__type __type; 
# 1339
return fma((__type)__x, (__type)__y, (__type)__z); 
# 1340
} 
# 1343
constexpr float fmax(float __x, float __y) 
# 1344
{ return __builtin_fmaxf(__x, __y); } 
# 1347
constexpr long double fmax(long double __x, long double __y) 
# 1348
{ return __builtin_fmaxl(__x, __y); } 
# 1350
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1352
fmax(_Tp __x, _Up __y) 
# 1353
{ 
# 1354
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1355
return fmax((__type)__x, (__type)__y); 
# 1356
} 
# 1359
constexpr float fmin(float __x, float __y) 
# 1360
{ return __builtin_fminf(__x, __y); } 
# 1363
constexpr long double fmin(long double __x, long double __y) 
# 1364
{ return __builtin_fminl(__x, __y); } 
# 1366
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1368
fmin(_Tp __x, _Up __y) 
# 1369
{ 
# 1370
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1371
return fmin((__type)__x, (__type)__y); 
# 1372
} 
# 1375
constexpr float hypot(float __x, float __y) 
# 1376
{ return __builtin_hypotf(__x, __y); } 
# 1379
constexpr long double hypot(long double __x, long double __y) 
# 1380
{ return __builtin_hypotl(__x, __y); } 
# 1382
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1384
hypot(_Tp __x, _Up __y) 
# 1385
{ 
# 1386
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1387
return hypot((__type)__x, (__type)__y); 
# 1388
} 
# 1391
constexpr int ilogb(float __x) 
# 1392
{ return __builtin_ilogbf(__x); } 
# 1395
constexpr int ilogb(long double __x) 
# 1396
{ return __builtin_ilogbl(__x); } 
# 1398
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, int> ::__type 
# 1402
ilogb(_Tp __x) 
# 1403
{ return __builtin_ilogb(__x); } 
# 1406
constexpr float lgamma(float __x) 
# 1407
{ return __builtin_lgammaf(__x); } 
# 1410
constexpr long double lgamma(long double __x) 
# 1411
{ return __builtin_lgammal(__x); } 
# 1413
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1416
lgamma(_Tp __x) 
# 1417
{ return __builtin_lgamma(__x); } 
# 1420
constexpr long long llrint(float __x) 
# 1421
{ return __builtin_llrintf(__x); } 
# 1424
constexpr long long llrint(long double __x) 
# 1425
{ return __builtin_llrintl(__x); } 
# 1427
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1430
llrint(_Tp __x) 
# 1431
{ return __builtin_llrint(__x); } 
# 1434
constexpr long long llround(float __x) 
# 1435
{ return __builtin_llroundf(__x); } 
# 1438
constexpr long long llround(long double __x) 
# 1439
{ return __builtin_llroundl(__x); } 
# 1441
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long long> ::__type 
# 1444
llround(_Tp __x) 
# 1445
{ return __builtin_llround(__x); } 
# 1448
constexpr float log1p(float __x) 
# 1449
{ return __builtin_log1pf(__x); } 
# 1452
constexpr long double log1p(long double __x) 
# 1453
{ return __builtin_log1pl(__x); } 
# 1455
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1458
log1p(_Tp __x) 
# 1459
{ return __builtin_log1p(__x); } 
# 1463
constexpr float log2(float __x) 
# 1464
{ return __builtin_log2f(__x); } 
# 1467
constexpr long double log2(long double __x) 
# 1468
{ return __builtin_log2l(__x); } 
# 1470
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1473
log2(_Tp __x) 
# 1474
{ return __builtin_log2(__x); } 
# 1477
constexpr float logb(float __x) 
# 1478
{ return __builtin_logbf(__x); } 
# 1481
constexpr long double logb(long double __x) 
# 1482
{ return __builtin_logbl(__x); } 
# 1484
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1487
logb(_Tp __x) 
# 1488
{ return __builtin_logb(__x); } 
# 1491
constexpr long lrint(float __x) 
# 1492
{ return __builtin_lrintf(__x); } 
# 1495
constexpr long lrint(long double __x) 
# 1496
{ return __builtin_lrintl(__x); } 
# 1498
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1501
lrint(_Tp __x) 
# 1502
{ return __builtin_lrint(__x); } 
# 1505
constexpr long lround(float __x) 
# 1506
{ return __builtin_lroundf(__x); } 
# 1509
constexpr long lround(long double __x) 
# 1510
{ return __builtin_lroundl(__x); } 
# 1512
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, long> ::__type 
# 1515
lround(_Tp __x) 
# 1516
{ return __builtin_lround(__x); } 
# 1519
constexpr float nearbyint(float __x) 
# 1520
{ return __builtin_nearbyintf(__x); } 
# 1523
constexpr long double nearbyint(long double __x) 
# 1524
{ return __builtin_nearbyintl(__x); } 
# 1526
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1529
nearbyint(_Tp __x) 
# 1530
{ return __builtin_nearbyint(__x); } 
# 1533
constexpr float nextafter(float __x, float __y) 
# 1534
{ return __builtin_nextafterf(__x, __y); } 
# 1537
constexpr long double nextafter(long double __x, long double __y) 
# 1538
{ return __builtin_nextafterl(__x, __y); } 
# 1540
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1542
nextafter(_Tp __x, _Up __y) 
# 1543
{ 
# 1544
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1545
return nextafter((__type)__x, (__type)__y); 
# 1546
} 
# 1549
constexpr float nexttoward(float __x, long double __y) 
# 1550
{ return __builtin_nexttowardf(__x, __y); } 
# 1553
constexpr long double nexttoward(long double __x, long double __y) 
# 1554
{ return __builtin_nexttowardl(__x, __y); } 
# 1556
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1559
nexttoward(_Tp __x, long double __y) 
# 1560
{ return __builtin_nexttoward(__x, __y); } 
# 1563
constexpr float remainder(float __x, float __y) 
# 1564
{ return __builtin_remainderf(__x, __y); } 
# 1567
constexpr long double remainder(long double __x, long double __y) 
# 1568
{ return __builtin_remainderl(__x, __y); } 
# 1570
template< class _Tp, class _Up> constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1572
remainder(_Tp __x, _Up __y) 
# 1573
{ 
# 1574
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1575
return remainder((__type)__x, (__type)__y); 
# 1576
} 
# 1579
inline float remquo(float __x, float __y, int *__pquo) 
# 1580
{ return __builtin_remquof(__x, __y, __pquo); } 
# 1583
inline long double remquo(long double __x, long double __y, int *__pquo) 
# 1584
{ return __builtin_remquol(__x, __y, __pquo); } 
# 1586
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 1588
remquo(_Tp __x, _Up __y, int *__pquo) 
# 1589
{ 
# 1590
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 1591
return remquo((__type)__x, (__type)__y, __pquo); 
# 1592
} 
# 1595
constexpr float rint(float __x) 
# 1596
{ return __builtin_rintf(__x); } 
# 1599
constexpr long double rint(long double __x) 
# 1600
{ return __builtin_rintl(__x); } 
# 1602
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1605
rint(_Tp __x) 
# 1606
{ return __builtin_rint(__x); } 
# 1609
constexpr float round(float __x) 
# 1610
{ return __builtin_roundf(__x); } 
# 1613
constexpr long double round(long double __x) 
# 1614
{ return __builtin_roundl(__x); } 
# 1616
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1619
round(_Tp __x) 
# 1620
{ return __builtin_round(__x); } 
# 1623
constexpr float scalbln(float __x, long __ex) 
# 1624
{ return __builtin_scalblnf(__x, __ex); } 
# 1627
constexpr long double scalbln(long double __x, long __ex) 
# 1628
{ return __builtin_scalblnl(__x, __ex); } 
# 1630
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1633
scalbln(_Tp __x, long __ex) 
# 1634
{ return __builtin_scalbln(__x, __ex); } 
# 1637
constexpr float scalbn(float __x, int __ex) 
# 1638
{ return __builtin_scalbnf(__x, __ex); } 
# 1641
constexpr long double scalbn(long double __x, int __ex) 
# 1642
{ return __builtin_scalbnl(__x, __ex); } 
# 1644
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1647
scalbn(_Tp __x, int __ex) 
# 1648
{ return __builtin_scalbn(__x, __ex); } 
# 1651
constexpr float tgamma(float __x) 
# 1652
{ return __builtin_tgammaf(__x); } 
# 1655
constexpr long double tgamma(long double __x) 
# 1656
{ return __builtin_tgammal(__x); } 
# 1658
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1661
tgamma(_Tp __x) 
# 1662
{ return __builtin_tgamma(__x); } 
# 1665
constexpr float trunc(float __x) 
# 1666
{ return __builtin_truncf(__x); } 
# 1669
constexpr long double trunc(long double __x) 
# 1670
{ return __builtin_truncl(__x); } 
# 1672
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 1675
trunc(_Tp __x) 
# 1676
{ return __builtin_trunc(__x); } 
# 1679
}
# 114 "/usr/include/c++/4.8/cstdlib" 3
namespace std __attribute((__visibility__("default"))) { 
# 118
using ::div_t;
# 119
using ::ldiv_t;
# 121
using ::abort;
# 122
using ::abs;
# 123
using ::atexit;
# 126
using ::at_quick_exit;
# 129
using ::atof;
# 130
using ::atoi;
# 131
using ::atol;
# 132
using ::bsearch;
# 133
using ::calloc;
# 134
using ::div;
# 135
using ::exit;
# 136
using ::free;
# 137
using ::getenv;
# 138
using ::labs;
# 139
using ::ldiv;
# 140
using ::malloc;
# 142
using ::mblen;
# 143
using ::mbstowcs;
# 144
using ::mbtowc;
# 146
using ::qsort;
# 149
using ::quick_exit;
# 152
using ::rand;
# 153
using ::realloc;
# 154
using ::srand;
# 155
using ::strtod;
# 156
using ::strtol;
# 157
using ::strtoul;
# 158
using ::system;
# 160
using ::wcstombs;
# 161
using ::wctomb;
# 166
inline long abs(long __i) { return __builtin_labs(__i); } 
# 169
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 174
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 183
}
# 196
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 201
using ::lldiv_t;
# 207
using ::_Exit;
# 211
using ::llabs;
# 214
inline lldiv_t div(long long __n, long long __d) 
# 215
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 217
using ::lldiv;
# 228
using ::atoll;
# 229
using ::strtoll;
# 230
using ::strtoull;
# 232
using ::strtof;
# 233
using ::strtold;
# 236
}
# 238
namespace std { 
# 241
using __gnu_cxx::lldiv_t;
# 243
using __gnu_cxx::_Exit;
# 245
using __gnu_cxx::llabs;
# 246
using __gnu_cxx::div;
# 247
using __gnu_cxx::lldiv;
# 249
using __gnu_cxx::atoll;
# 250
using __gnu_cxx::strtof;
# 251
using __gnu_cxx::strtoll;
# 252
using __gnu_cxx::strtoull;
# 253
using __gnu_cxx::strtold;
# 254
}
# 8925 "/usr/local/cuda-8.0/include/math_functions.h"
__attribute((always_inline)) inline int signbit(float x); 
# 8929
__attribute((always_inline)) inline int signbit(double x); 
# 8931
__attribute((always_inline)) inline int signbit(long double x); 
# 8933
__attribute((always_inline)) inline int isfinite(float x); 
# 8937
__attribute((always_inline)) inline int isfinite(double x); 
# 8939
__attribute((always_inline)) inline int isfinite(long double x); 
# 8941
__attribute((always_inline)) inline int isnan(float x); 
# 8945
__attribute((always_inline)) inline int isnan(double x) throw(); 
# 8947
__attribute((always_inline)) inline int isnan(long double x); 
# 8949
__attribute((always_inline)) inline int isinf(float x); 
# 8953
__attribute((always_inline)) inline int isinf(double x) throw(); 
# 8955
__attribute((always_inline)) inline int isinf(long double x); 
# 9002
namespace std { 
# 9004
template< class T> extern T __pow_helper(T, int); 
# 9005
template< class T> extern T __cmath_power(T, unsigned); 
# 9006
}
# 9008
using std::abs;
# 9009
using std::fabs;
# 9010
using std::ceil;
# 9011
using std::floor;
# 9012
using std::sqrt;
# 9013
using std::pow;
# 9014
using std::log;
# 9015
using std::log10;
# 9016
using std::fmod;
# 9017
using std::modf;
# 9018
using std::exp;
# 9019
using std::frexp;
# 9020
using std::ldexp;
# 9021
using std::asin;
# 9022
using std::sin;
# 9023
using std::sinh;
# 9024
using std::acos;
# 9025
using std::cos;
# 9026
using std::cosh;
# 9027
using std::atan;
# 9028
using std::atan2;
# 9029
using std::tan;
# 9030
using std::tanh;
# 9393
namespace std { 
# 9406
extern inline long long abs(long long); 
# 9412
extern inline long abs(long); 
# 9413
extern constexpr float abs(float); 
# 9414
extern constexpr double abs(double); 
# 9415
extern constexpr float fabs(float); 
# 9416
extern constexpr float ceil(float); 
# 9417
extern constexpr float floor(float); 
# 9418
extern constexpr float sqrt(float); 
# 9419
extern constexpr float pow(float, float); 
# 9424
template< class _Tp, class _Up> extern constexpr typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type pow(_Tp, _Up); 
# 9434
extern constexpr float log(float); 
# 9435
extern constexpr float log10(float); 
# 9436
extern constexpr float fmod(float, float); 
# 9437
extern inline float modf(float, float *); 
# 9438
extern constexpr float exp(float); 
# 9439
extern inline float frexp(float, int *); 
# 9440
extern constexpr float ldexp(float, int); 
# 9441
extern constexpr float asin(float); 
# 9442
extern constexpr float sin(float); 
# 9443
extern constexpr float sinh(float); 
# 9444
extern constexpr float acos(float); 
# 9445
extern constexpr float cos(float); 
# 9446
extern constexpr float cosh(float); 
# 9447
extern constexpr float atan(float); 
# 9448
extern constexpr float atan2(float, float); 
# 9449
extern constexpr float tan(float); 
# 9450
extern constexpr float tanh(float); 
# 9518
}
# 9609
static inline float logb(float a); 
# 9611
static inline int ilogb(float a); 
# 9613
static inline float scalbn(float a, int b); 
# 9615
static inline float scalbln(float a, long b); 
# 9617
static inline float exp2(float a); 
# 9619
static inline float expm1(float a); 
# 9621
static inline float log2(float a); 
# 9623
static inline float log1p(float a); 
# 9625
static inline float acosh(float a); 
# 9627
static inline float asinh(float a); 
# 9629
static inline float atanh(float a); 
# 9631
static inline float hypot(float a, float b); 
# 9633
static inline float norm3d(float a, float b, float c); 
# 9635
static inline float norm4d(float a, float b, float c, float d); 
# 9637
static inline float cbrt(float a); 
# 9639
static inline float erf(float a); 
# 9641
static inline float erfc(float a); 
# 9643
static inline float lgamma(float a); 
# 9645
static inline float tgamma(float a); 
# 9647
static inline float copysign(float a, float b); 
# 9649
static inline float nextafter(float a, float b); 
# 9651
static inline float remainder(float a, float b); 
# 9653
static inline float remquo(float a, float b, int * quo); 
# 9655
static inline float round(float a); 
# 9657
static inline long lround(float a); 
# 9659
static inline long long llround(float a); 
# 9661
static inline float trunc(float a); 
# 9663
static inline float rint(float a); 
# 9665
static inline long lrint(float a); 
# 9667
static inline long long llrint(float a); 
# 9669
static inline float nearbyint(float a); 
# 9671
static inline float fdim(float a, float b); 
# 9673
static inline float fma(float a, float b, float c); 
# 9675
static inline float fmax(float a, float b); 
# 9677
static inline float fmin(float a, float b); 
# 9718
static inline float exp10(float a); 
# 9720
static inline float rsqrt(float a); 
# 9722
static inline float rcbrt(float a); 
# 9724
static inline float sinpi(float a); 
# 9726
static inline float cospi(float a); 
# 9728
static inline void sincospi(float a, float * sptr, float * cptr); 
# 9730
static inline void sincos(float a, float * sptr, float * cptr); 
# 9732
static inline float j0(float a); 
# 9734
static inline float j1(float a); 
# 9736
static inline float jn(int n, float a); 
# 9738
static inline float y0(float a); 
# 9740
static inline float y1(float a); 
# 9742
static inline float yn(int n, float a); 
# 9744
static inline float cyl_bessel_i0(float a); 
# 9746
static inline float cyl_bessel_i1(float a); 
# 9748
static inline float erfinv(float a); 
# 9750
static inline float erfcinv(float a); 
# 9752
static inline float normcdfinv(float a); 
# 9754
static inline float normcdf(float a); 
# 9756
static inline float erfcx(float a); 
# 9758
static inline double copysign(double a, float b); 
# 9760
static inline float copysign(float a, double b); 
# 9762
static inline unsigned min(unsigned a, unsigned b); 
# 9764
static inline unsigned min(int a, unsigned b); 
# 9766
static inline unsigned min(unsigned a, int b); 
# 9768
static inline long min(long a, long b); 
# 9770
static inline unsigned long min(unsigned long a, unsigned long b); 
# 9772
static inline unsigned long min(long a, unsigned long b); 
# 9774
static inline unsigned long min(unsigned long a, long b); 
# 9776
static inline long long min(long long a, long long b); 
# 9778
static inline unsigned long long min(unsigned long long a, unsigned long long b); 
# 9780
static inline unsigned long long min(long long a, unsigned long long b); 
# 9782
static inline unsigned long long min(unsigned long long a, long long b); 
# 9784
static inline float min(float a, float b); 
# 9786
static inline double min(double a, double b); 
# 9788
static inline double min(float a, double b); 
# 9790
static inline double min(double a, float b); 
# 9792
static inline unsigned max(unsigned a, unsigned b); 
# 9794
static inline unsigned max(int a, unsigned b); 
# 9796
static inline unsigned max(unsigned a, int b); 
# 9798
static inline long max(long a, long b); 
# 9800
static inline unsigned long max(unsigned long a, unsigned long b); 
# 9802
static inline unsigned long max(long a, unsigned long b); 
# 9804
static inline unsigned long max(unsigned long a, long b); 
# 9806
static inline long long max(long long a, long long b); 
# 9808
static inline unsigned long long max(unsigned long long a, unsigned long long b); 
# 9810
static inline unsigned long long max(long long a, unsigned long long b); 
# 9812
static inline unsigned long long max(unsigned long long a, long long b); 
# 9814
static inline float max(float a, float b); 
# 9816
static inline double max(double a, double b); 
# 9818
static inline double max(float a, double b); 
# 9820
static inline double max(double a, float b); 
# 248 "/usr/local/cuda-8.0/include/math_functions.hpp"
__attribute((always_inline)) inline int signbit(float x) { return __signbitf(x); } 
# 252
__attribute((always_inline)) inline int signbit(double x) { return __signbit(x); } 
# 254
__attribute((always_inline)) inline int signbit(long double x) { return __signbitl(x); } 
# 265
__attribute((always_inline)) inline int isfinite(float x) { return __finitef(x); } 
# 280
__attribute((always_inline)) inline int isfinite(double x) { return __finite(x); } 
# 293
__attribute((always_inline)) inline int isfinite(long double x) { return __finitel(x); } 
# 296
__attribute((always_inline)) inline int isnan(float x) { return __isnanf(x); } 
# 300
__attribute((always_inline)) inline int isnan(double x) throw() { return __isnan(x); } 
# 302
__attribute((always_inline)) inline int isnan(long double x) { return __isnanl(x); } 
# 304
__attribute((always_inline)) inline int isinf(float x) { return __isinff(x); } 
# 308
__attribute((always_inline)) inline int isinf(double x) throw() { return __isinf(x); } 
# 310
__attribute((always_inline)) inline int isinf(long double x) { return __isinfl(x); } 
# 503
static inline float logb(float a) 
# 504
{ 
# 505
return logbf(a); 
# 506
} 
# 508
static inline int ilogb(float a) 
# 509
{ 
# 510
return ilogbf(a); 
# 511
} 
# 513
static inline float scalbn(float a, int b) 
# 514
{ 
# 515
return scalbnf(a, b); 
# 516
} 
# 518
static inline float scalbln(float a, long b) 
# 519
{ 
# 520
return scalblnf(a, b); 
# 521
} 
# 523
static inline float exp2(float a) 
# 524
{ 
# 525
return exp2f(a); 
# 526
} 
# 528
static inline float expm1(float a) 
# 529
{ 
# 530
return expm1f(a); 
# 531
} 
# 533
static inline float log2(float a) 
# 534
{ 
# 535
return log2f(a); 
# 536
} 
# 538
static inline float log1p(float a) 
# 539
{ 
# 540
return log1pf(a); 
# 541
} 
# 543
static inline float acosh(float a) 
# 544
{ 
# 545
return acoshf(a); 
# 546
} 
# 548
static inline float asinh(float a) 
# 549
{ 
# 550
return asinhf(a); 
# 551
} 
# 553
static inline float atanh(float a) 
# 554
{ 
# 555
return atanhf(a); 
# 556
} 
# 558
static inline float hypot(float a, float b) 
# 559
{ 
# 560
return hypotf(a, b); 
# 561
} 
# 563
static inline float norm3d(float a, float b, float c) 
# 564
{ 
# 565
return norm3df(a, b, c); 
# 566
} 
# 568
static inline float norm4d(float a, float b, float c, float d) 
# 569
{ 
# 570
return norm4df(a, b, c, d); 
# 571
} 
# 573
static inline float cbrt(float a) 
# 574
{ 
# 575
return cbrtf(a); 
# 576
} 
# 578
static inline float erf(float a) 
# 579
{ 
# 580
return erff(a); 
# 581
} 
# 583
static inline float erfc(float a) 
# 584
{ 
# 585
return erfcf(a); 
# 586
} 
# 588
static inline float lgamma(float a) 
# 589
{ 
# 590
return lgammaf(a); 
# 591
} 
# 593
static inline float tgamma(float a) 
# 594
{ 
# 595
return tgammaf(a); 
# 596
} 
# 598
static inline float copysign(float a, float b) 
# 599
{ 
# 600
return copysignf(a, b); 
# 601
} 
# 603
static inline float nextafter(float a, float b) 
# 604
{ 
# 605
return nextafterf(a, b); 
# 606
} 
# 608
static inline float remainder(float a, float b) 
# 609
{ 
# 610
return remainderf(a, b); 
# 611
} 
# 613
static inline float remquo(float a, float b, int *quo) 
# 614
{ 
# 615
return remquof(a, b, quo); 
# 616
} 
# 618
static inline float round(float a) 
# 619
{ 
# 620
return roundf(a); 
# 621
} 
# 623
static inline long lround(float a) 
# 624
{ 
# 625
return lroundf(a); 
# 626
} 
# 628
static inline long long llround(float a) 
# 629
{ 
# 630
return llroundf(a); 
# 631
} 
# 633
static inline float trunc(float a) 
# 634
{ 
# 635
return truncf(a); 
# 636
} 
# 638
static inline float rint(float a) 
# 639
{ 
# 640
return rintf(a); 
# 641
} 
# 643
static inline long lrint(float a) 
# 644
{ 
# 645
return lrintf(a); 
# 646
} 
# 648
static inline long long llrint(float a) 
# 649
{ 
# 650
return llrintf(a); 
# 651
} 
# 653
static inline float nearbyint(float a) 
# 654
{ 
# 655
return nearbyintf(a); 
# 656
} 
# 658
static inline float fdim(float a, float b) 
# 659
{ 
# 660
return fdimf(a, b); 
# 661
} 
# 663
static inline float fma(float a, float b, float c) 
# 664
{ 
# 665
return fmaf(a, b, c); 
# 666
} 
# 668
static inline float fmax(float a, float b) 
# 669
{ 
# 670
return fmaxf(a, b); 
# 671
} 
# 673
static inline float fmin(float a, float b) 
# 674
{ 
# 675
return fminf(a, b); 
# 676
} 
# 681
static inline float exp10(float a) 
# 682
{ 
# 683
return exp10f(a); 
# 684
} 
# 686
static inline float rsqrt(float a) 
# 687
{ 
# 688
return rsqrtf(a); 
# 689
} 
# 691
static inline float rcbrt(float a) 
# 692
{ 
# 693
return rcbrtf(a); 
# 694
} 
# 696
static inline float sinpi(float a) 
# 697
{ 
# 698
return sinpif(a); 
# 699
} 
# 701
static inline float cospi(float a) 
# 702
{ 
# 703
return cospif(a); 
# 704
} 
# 706
static inline void sincospi(float a, float *sptr, float *cptr) 
# 707
{ 
# 708
sincospif(a, sptr, cptr); 
# 709
} 
# 711
static inline void sincos(float a, float *sptr, float *cptr) 
# 712
{ 
# 713
sincosf(a, sptr, cptr); 
# 714
} 
# 716
static inline float j0(float a) 
# 717
{ 
# 718
return j0f(a); 
# 719
} 
# 721
static inline float j1(float a) 
# 722
{ 
# 723
return j1f(a); 
# 724
} 
# 726
static inline float jn(int n, float a) 
# 727
{ 
# 728
return jnf(n, a); 
# 729
} 
# 731
static inline float y0(float a) 
# 732
{ 
# 733
return y0f(a); 
# 734
} 
# 736
static inline float y1(float a) 
# 737
{ 
# 738
return y1f(a); 
# 739
} 
# 741
static inline float yn(int n, float a) 
# 742
{ 
# 743
return ynf(n, a); 
# 744
} 
# 746
static inline float cyl_bessel_i0(float a) 
# 747
{ 
# 748
return cyl_bessel_i0f(a); 
# 749
} 
# 751
static inline float cyl_bessel_i1(float a) 
# 752
{ 
# 753
return cyl_bessel_i1f(a); 
# 754
} 
# 756
static inline float erfinv(float a) 
# 757
{ 
# 758
return erfinvf(a); 
# 759
} 
# 761
static inline float erfcinv(float a) 
# 762
{ 
# 763
return erfcinvf(a); 
# 764
} 
# 766
static inline float normcdfinv(float a) 
# 767
{ 
# 768
return normcdfinvf(a); 
# 769
} 
# 771
static inline float normcdf(float a) 
# 772
{ 
# 773
return normcdff(a); 
# 774
} 
# 776
static inline float erfcx(float a) 
# 777
{ 
# 778
return erfcxf(a); 
# 779
} 
# 781
static inline double copysign(double a, float b) 
# 782
{ 
# 783
return copysign(a, (double)b); 
# 784
} 
# 786
static inline float copysign(float a, double b) 
# 787
{ 
# 788
return copysignf(a, (float)b); 
# 789
} 
# 791
static inline unsigned min(unsigned a, unsigned b) 
# 792
{ 
# 793
return umin(a, b); 
# 794
} 
# 796
static inline unsigned min(int a, unsigned b) 
# 797
{ 
# 798
return umin((unsigned)a, b); 
# 799
} 
# 801
static inline unsigned min(unsigned a, int b) 
# 802
{ 
# 803
return umin(a, (unsigned)b); 
# 804
} 
# 806
static inline long min(long a, long b) 
# 807
{ 
# 813
if (sizeof(long) == sizeof(int)) { 
# 817
return (long)min((int)a, (int)b); 
# 818
} else { 
# 819
return (long)llmin((long long)a, (long long)b); 
# 820
}  
# 821
} 
# 823
static inline unsigned long min(unsigned long a, unsigned long b) 
# 824
{ 
# 828
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 832
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 833
} else { 
# 834
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 835
}  
# 836
} 
# 838
static inline unsigned long min(long a, unsigned long b) 
# 839
{ 
# 843
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 847
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 848
} else { 
# 849
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 850
}  
# 851
} 
# 853
static inline unsigned long min(unsigned long a, long b) 
# 854
{ 
# 858
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 862
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 863
} else { 
# 864
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 865
}  
# 866
} 
# 868
static inline long long min(long long a, long long b) 
# 869
{ 
# 870
return llmin(a, b); 
# 871
} 
# 873
static inline unsigned long long min(unsigned long long a, unsigned long long b) 
# 874
{ 
# 875
return ullmin(a, b); 
# 876
} 
# 878
static inline unsigned long long min(long long a, unsigned long long b) 
# 879
{ 
# 880
return ullmin((unsigned long long)a, b); 
# 881
} 
# 883
static inline unsigned long long min(unsigned long long a, long long b) 
# 884
{ 
# 885
return ullmin(a, (unsigned long long)b); 
# 886
} 
# 888
static inline float min(float a, float b) 
# 889
{ 
# 890
return fminf(a, b); 
# 891
} 
# 893
static inline double min(double a, double b) 
# 894
{ 
# 895
return fmin(a, b); 
# 896
} 
# 898
static inline double min(float a, double b) 
# 899
{ 
# 900
return fmin((double)a, b); 
# 901
} 
# 903
static inline double min(double a, float b) 
# 904
{ 
# 905
return fmin(a, (double)b); 
# 906
} 
# 908
static inline unsigned max(unsigned a, unsigned b) 
# 909
{ 
# 910
return umax(a, b); 
# 911
} 
# 913
static inline unsigned max(int a, unsigned b) 
# 914
{ 
# 915
return umax((unsigned)a, b); 
# 916
} 
# 918
static inline unsigned max(unsigned a, int b) 
# 919
{ 
# 920
return umax(a, (unsigned)b); 
# 921
} 
# 923
static inline long max(long a, long b) 
# 924
{ 
# 929
if (sizeof(long) == sizeof(int)) { 
# 933
return (long)max((int)a, (int)b); 
# 934
} else { 
# 935
return (long)llmax((long long)a, (long long)b); 
# 936
}  
# 937
} 
# 939
static inline unsigned long max(unsigned long a, unsigned long b) 
# 940
{ 
# 944
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 948
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 949
} else { 
# 950
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 951
}  
# 952
} 
# 954
static inline unsigned long max(long a, unsigned long b) 
# 955
{ 
# 959
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 963
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 964
} else { 
# 965
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 966
}  
# 967
} 
# 969
static inline unsigned long max(unsigned long a, long b) 
# 970
{ 
# 974
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 978
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 979
} else { 
# 980
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 981
}  
# 982
} 
# 984
static inline long long max(long long a, long long b) 
# 985
{ 
# 986
return llmax(a, b); 
# 987
} 
# 989
static inline unsigned long long max(unsigned long long a, unsigned long long b) 
# 990
{ 
# 991
return ullmax(a, b); 
# 992
} 
# 994
static inline unsigned long long max(long long a, unsigned long long b) 
# 995
{ 
# 996
return ullmax((unsigned long long)a, b); 
# 997
} 
# 999
static inline unsigned long long max(unsigned long long a, long long b) 
# 1000
{ 
# 1001
return ullmax(a, (unsigned long long)b); 
# 1002
} 
# 1004
static inline float max(float a, float b) 
# 1005
{ 
# 1006
return fmaxf(a, b); 
# 1007
} 
# 1009
static inline double max(double a, double b) 
# 1010
{ 
# 1011
return fmax(a, b); 
# 1012
} 
# 1014
static inline double max(float a, double b) 
# 1015
{ 
# 1016
return fmax((double)a, b); 
# 1017
} 
# 1019
static inline double max(double a, float b) 
# 1020
{ 
# 1021
return fmax(a, (double)b); 
# 1022
} 
# 1033
extern "C" inline int min(int a, int b) 
# 1034
{ 
# 1035
return (a < b) ? a : b; 
# 1036
} 
# 1038
extern "C" inline unsigned umin(unsigned a, unsigned b) 
# 1039
{ 
# 1040
return (a < b) ? a : b; 
# 1041
} 
# 1043
extern "C" inline long long llmin(long long a, long long b) 
# 1044
{ 
# 1045
return (a < b) ? a : b; 
# 1046
} 
# 1048
extern "C" inline unsigned long long ullmin(unsigned long long a, unsigned long long 
# 1049
b) 
# 1050
{ 
# 1051
return (a < b) ? a : b; 
# 1052
} 
# 1054
extern "C" inline int max(int a, int b) 
# 1055
{ 
# 1056
return (a > b) ? a : b; 
# 1057
} 
# 1059
extern "C" inline unsigned umax(unsigned a, unsigned b) 
# 1060
{ 
# 1061
return (a > b) ? a : b; 
# 1062
} 
# 1064
extern "C" inline long long llmax(long long a, long long b) 
# 1065
{ 
# 1066
return (a > b) ? a : b; 
# 1067
} 
# 1069
extern "C" inline unsigned long long ullmax(unsigned long long a, unsigned long long 
# 1070
b) 
# 1071
{ 
# 1072
return (a > b) ? a : b; 
# 1073
} 
# 77 "/usr/local/cuda-8.0/include/cuda_surface_types.h"
template< class T, int dim = 1> 
# 78
struct surface : public surfaceReference { 
# 81
surface() 
# 82
{ 
# 83
(surfaceReference::channelDesc) = cudaCreateChannelDesc< T> (); 
# 84
} 
# 86
surface(cudaChannelFormatDesc desc) 
# 87
{ 
# 88
(surfaceReference::channelDesc) = desc; 
# 89
} 
# 91
}; 
# 93
template< int dim> 
# 94
struct surface< void, dim>  : public surfaceReference { 
# 97
surface() 
# 98
{ 
# 99
(surfaceReference::channelDesc) = cudaCreateChannelDesc< void> (); 
# 100
} 
# 102
}; 
# 77 "/usr/local/cuda-8.0/include/cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
# 78
struct texture : public textureReference { 
# 81
texture(int norm = 0, cudaTextureFilterMode 
# 82
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 83
aMode = cudaAddressModeClamp) 
# 84
{ 
# 85
(textureReference::normalized) = norm; 
# 86
(textureReference::filterMode) = fMode; 
# 87
((textureReference::addressMode)[0]) = aMode; 
# 88
((textureReference::addressMode)[1]) = aMode; 
# 89
((textureReference::addressMode)[2]) = aMode; 
# 90
(textureReference::channelDesc) = cudaCreateChannelDesc< T> (); 
# 91
(textureReference::sRGB) = 0; 
# 92
} 
# 94
texture(int norm, cudaTextureFilterMode 
# 95
fMode, cudaTextureAddressMode 
# 96
aMode, cudaChannelFormatDesc 
# 97
desc) 
# 98
{ 
# 99
(textureReference::normalized) = norm; 
# 100
(textureReference::filterMode) = fMode; 
# 101
((textureReference::addressMode)[0]) = aMode; 
# 102
((textureReference::addressMode)[1]) = aMode; 
# 103
((textureReference::addressMode)[2]) = aMode; 
# 104
(textureReference::channelDesc) = desc; 
# 105
(textureReference::sRGB) = 0; 
# 106
} 
# 108
}; 
# 90 "/usr/local/cuda-8.0/include/device_functions.h"
extern "C" {
# 3230
}
# 3238
__attribute__((unused)) static inline int mulhi(int a, int b); 
# 3240
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b); 
# 3242
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b); 
# 3244
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b); 
# 3246
__attribute__((unused)) static inline long long mul64hi(long long a, long long b); 
# 3248
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b); 
# 3250
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b); 
# 3252
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b); 
# 3254
__attribute__((unused)) static inline int float_as_int(float a); 
# 3256
__attribute__((unused)) static inline float int_as_float(int a); 
# 3258
__attribute__((unused)) static inline unsigned float_as_uint(float a); 
# 3260
__attribute__((unused)) static inline float uint_as_float(unsigned a); 
# 3262
__attribute__((unused)) static inline float saturate(float a); 
# 3264
__attribute__((unused)) static inline int mul24(int a, int b); 
# 3266
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b); 
# 3268
__attribute((deprecated("Please use __trap() instead."))) __attribute__((unused)) static inline void trap(); 
# 3271
__attribute((deprecated("Please use __brkpt() instead."))) __attribute__((unused)) static inline void brkpt(int c = 0); 
# 3273
__attribute((deprecated("Please use __syncthreads() instead."))) __attribute__((unused)) static inline void syncthreads(); 
# 3275
__attribute((deprecated("Please use __prof_trigger() instead."))) __attribute__((unused)) static inline void prof_trigger(int e); 
# 3277
__attribute((deprecated("Please use __threadfence() instead."))) __attribute__((unused)) static inline void threadfence(bool global = true); 
# 3279
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode = cudaRoundZero); 
# 3281
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero); 
# 3283
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest); 
# 3285
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 83 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline int mulhi(int a, int b) 
# 84
{int volatile ___ = 1;(void)a;(void)b;
# 86
::exit(___);}
#if 0
# 84
{ 
# 85
return __mulhi(a, b); 
# 86
} 
#endif
# 88 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b) 
# 89
{int volatile ___ = 1;(void)a;(void)b;
# 91
::exit(___);}
#if 0
# 89
{ 
# 90
return __umulhi(a, b); 
# 91
} 
#endif
# 93 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b) 
# 94
{int volatile ___ = 1;(void)a;(void)b;
# 96
::exit(___);}
#if 0
# 94
{ 
# 95
return __umulhi((unsigned)a, b); 
# 96
} 
#endif
# 98 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b) 
# 99
{int volatile ___ = 1;(void)a;(void)b;
# 101
::exit(___);}
#if 0
# 99
{ 
# 100
return __umulhi(a, (unsigned)b); 
# 101
} 
#endif
# 103 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline long long mul64hi(long long a, long long b) 
# 104
{int volatile ___ = 1;(void)a;(void)b;
# 106
::exit(___);}
#if 0
# 104
{ 
# 105
return __mul64hi(a, b); 
# 106
} 
#endif
# 108 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b) 
# 109
{int volatile ___ = 1;(void)a;(void)b;
# 111
::exit(___);}
#if 0
# 109
{ 
# 110
return __umul64hi(a, b); 
# 111
} 
#endif
# 113 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b) 
# 114
{int volatile ___ = 1;(void)a;(void)b;
# 116
::exit(___);}
#if 0
# 114
{ 
# 115
return __umul64hi((unsigned long long)a, b); 
# 116
} 
#endif
# 118 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b) 
# 119
{int volatile ___ = 1;(void)a;(void)b;
# 121
::exit(___);}
#if 0
# 119
{ 
# 120
return __umul64hi(a, (unsigned long long)b); 
# 121
} 
#endif
# 123 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline int float_as_int(float a) 
# 124
{int volatile ___ = 1;(void)a;
# 126
::exit(___);}
#if 0
# 124
{ 
# 125
return __float_as_int(a); 
# 126
} 
#endif
# 128 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline float int_as_float(int a) 
# 129
{int volatile ___ = 1;(void)a;
# 131
::exit(___);}
#if 0
# 129
{ 
# 130
return __int_as_float(a); 
# 131
} 
#endif
# 133 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned float_as_uint(float a) 
# 134
{int volatile ___ = 1;(void)a;
# 136
::exit(___);}
#if 0
# 134
{ 
# 135
return __float_as_uint(a); 
# 136
} 
#endif
# 138 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline float uint_as_float(unsigned a) 
# 139
{int volatile ___ = 1;(void)a;
# 141
::exit(___);}
#if 0
# 139
{ 
# 140
return __uint_as_float(a); 
# 141
} 
#endif
# 142 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline float saturate(float a) 
# 143
{int volatile ___ = 1;(void)a;
# 145
::exit(___);}
#if 0
# 143
{ 
# 144
return __saturatef(a); 
# 145
} 
#endif
# 147 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline int mul24(int a, int b) 
# 148
{int volatile ___ = 1;(void)a;(void)b;
# 150
::exit(___);}
#if 0
# 148
{ 
# 149
return __mul24(a, b); 
# 150
} 
#endif
# 152 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b) 
# 153
{int volatile ___ = 1;(void)a;(void)b;
# 155
::exit(___);}
#if 0
# 153
{ 
# 154
return __umul24(a, b); 
# 155
} 
#endif
# 157 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline void trap() 
# 158
{int volatile ___ = 1;
# 160
::exit(___);}
#if 0
# 158
{ 
# 159
__trap(); 
# 160
} 
#endif
# 163 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline void brkpt(int c) 
# 164
{int volatile ___ = 1;(void)c;
# 166
::exit(___);}
#if 0
# 164
{ 
# 165
__brkpt(c); 
# 166
} 
#endif
# 168 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline void syncthreads() 
# 169
{int volatile ___ = 1;
# 171
::exit(___);}
#if 0
# 169
{ 
# 170
__syncthreads(); 
# 171
} 
#endif
# 173 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline void prof_trigger(int e) 
# 174
{int volatile ___ = 1;(void)e;
# 191
::exit(___);}
#if 0
# 174
{ 
# 175
if (e == 0) { __prof_trigger(0); } else { 
# 176
if (e == 1) { __prof_trigger(1); } else { 
# 177
if (e == 2) { __prof_trigger(2); } else { 
# 178
if (e == 3) { __prof_trigger(3); } else { 
# 179
if (e == 4) { __prof_trigger(4); } else { 
# 180
if (e == 5) { __prof_trigger(5); } else { 
# 181
if (e == 6) { __prof_trigger(6); } else { 
# 182
if (e == 7) { __prof_trigger(7); } else { 
# 183
if (e == 8) { __prof_trigger(8); } else { 
# 184
if (e == 9) { __prof_trigger(9); } else { 
# 185
if (e == 10) { __prof_trigger(10); } else { 
# 186
if (e == 11) { __prof_trigger(11); } else { 
# 187
if (e == 12) { __prof_trigger(12); } else { 
# 188
if (e == 13) { __prof_trigger(13); } else { 
# 189
if (e == 14) { __prof_trigger(14); } else { 
# 190
if (e == 15) { __prof_trigger(15); }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  
# 191
} 
#endif
# 193 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline void threadfence(bool global) 
# 194
{int volatile ___ = 1;(void)global;
# 196
::exit(___);}
#if 0
# 194
{ 
# 195
global ? __threadfence() : __threadfence_block(); 
# 196
} 
#endif
# 198 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode) 
# 199
{int volatile ___ = 1;(void)a;(void)mode;
# 204
::exit(___);}
#if 0
# 199
{ 
# 200
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
# 204
} 
#endif
# 206 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode) 
# 207
{int volatile ___ = 1;(void)a;(void)mode;
# 212
::exit(___);}
#if 0
# 207
{ 
# 208
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
# 212
} 
#endif
# 214 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode) 
# 215
{int volatile ___ = 1;(void)a;(void)mode;
# 220
::exit(___);}
#if 0
# 215
{ 
# 216
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
# 220
} 
#endif
# 222 "/usr/local/cuda-8.0/include/device_functions.hpp"
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode) 
# 223
{int volatile ___ = 1;(void)a;(void)mode;
# 228
::exit(___);}
#if 0
# 223
{ 
# 224
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
# 228
} 
#endif
# 111 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 117
{ } 
#endif
# 119 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 119
{ } 
#endif
# 121 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 121
{ } 
#endif
# 123 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 123
{ } 
#endif
# 125 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 125
{ } 
#endif
# 127 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 127
{ } 
#endif
# 129 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 129
{ } 
#endif
# 131 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 131
{ } 
#endif
# 133 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 133
{ } 
#endif
# 135 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 135
{ } 
#endif
# 137 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 139
{ } 
#endif
# 141 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 141
{ } 
#endif
# 143 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 143
{ } 
#endif
# 145 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 145
{ } 
#endif
# 147 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 147
{ } 
#endif
# 149 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 151
{ } 
#endif
# 164 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
extern "C" {
# 175
}
# 185
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 187
{ } 
#endif
# 189 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 189
{ } 
#endif
# 191 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 191
{ } 
#endif
# 193 "/usr/local/cuda-8.0/include/device_atomic_functions.h"
__attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 193
{ } 
#endif
# 80 "/usr/local/cuda-8.0/include/device_double_functions.h"
extern "C" {
# 1134
}
# 1143
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1145
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1147
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1149
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1151
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1153
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1155
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1157
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1159
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1161
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1163
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1165
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1167
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 85 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 86
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 91
::exit(___);}
#if 0
# 86
{ 
# 87
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 91
} 
#endif
# 93 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 99
} 
#endif
# 101 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 107
} 
#endif
# 109 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 115
} 
#endif
# 117 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 123
} 
#endif
# 125 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 131
} 
#endif
# 133 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 139
} 
#endif
# 141 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 147
} 
#endif
# 149 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 155
} 
#endif
# 157 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 163
} 
#endif
# 165 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 168
::exit(___);}
#if 0
# 166
{ 
# 167
return (double)a; 
# 168
} 
#endif
# 170 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 171
{int volatile ___ = 1;(void)a;(void)mode;
# 173
::exit(___);}
#if 0
# 171
{ 
# 172
return (double)a; 
# 173
} 
#endif
# 175 "/usr/local/cuda-8.0/include/device_double_functions.hpp"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 176
{int volatile ___ = 1;(void)a;(void)mode;
# 178
::exit(___);}
#if 0
# 176
{ 
# 177
return (double)a; 
# 178
} 
#endif
# 94 "/usr/local/cuda-8.0/include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 94
{ } 
#endif
# 102 "/usr/local/cuda-8.0/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 102
{ } 
#endif
# 104 "/usr/local/cuda-8.0/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 104
{ } 
#endif
# 106 "/usr/local/cuda-8.0/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/usr/local/cuda-8.0/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/usr/local/cuda-8.0/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/usr/local/cuda-8.0/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/usr/local/cuda-8.0/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 308 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 308
{ } 
#endif
# 311 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 311
{ } 
#endif
# 314 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 314
{ } 
#endif
# 317 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 317
{ } 
#endif
# 320 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 320
{ } 
#endif
# 323 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 323
{ } 
#endif
# 326 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 326
{ } 
#endif
# 329 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 329
{ } 
#endif
# 332 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 332
{ } 
#endif
# 335 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 335
{ } 
#endif
# 338 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 338
{ } 
#endif
# 341 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 341
{ } 
#endif
# 344 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 344
{ } 
#endif
# 347 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 347
{ } 
#endif
# 350 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 350
{ } 
#endif
# 353 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 353
{ } 
#endif
# 356 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 356
{ } 
#endif
# 359 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 359
{ } 
#endif
# 362 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 362
{ } 
#endif
# 365 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 365
{ } 
#endif
# 368 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 368
{ } 
#endif
# 371 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 371
{ } 
#endif
# 374 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 374
{ } 
#endif
# 377 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 377
{ } 
#endif
# 380 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 380
{ } 
#endif
# 383 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 383
{ } 
#endif
# 386 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 386
{ } 
#endif
# 389 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 389
{ } 
#endif
# 392 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 392
{ } 
#endif
# 395 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 395
{ } 
#endif
# 398 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 398
{ } 
#endif
# 401 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 401
{ } 
#endif
# 404 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 404
{ } 
#endif
# 407 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 407
{ } 
#endif
# 410 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 410
{ } 
#endif
# 413 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 413
{ } 
#endif
# 416 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 416
{ } 
#endif
# 419 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 419
{ } 
#endif
# 422 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 422
{ } 
#endif
# 425 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 425
{ } 
#endif
# 428 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 428
{ } 
#endif
# 431 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 432
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 432
{ } 
#endif
# 435 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 436
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 436
{ } 
#endif
# 439 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 440
compare, unsigned long long 
# 441
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 441
{ } 
#endif
# 444 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 445
compare, unsigned long long 
# 446
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 446
{ } 
#endif
# 449 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 449
{ } 
#endif
# 452 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 452
{ } 
#endif
# 455 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 455
{ } 
#endif
# 458 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 458
{ } 
#endif
# 461 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 461
{ } 
#endif
# 464 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 464
{ } 
#endif
# 467 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 467
{ } 
#endif
# 470 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 470
{ } 
#endif
# 473 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 473
{ } 
#endif
# 476 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 476
{ } 
#endif
# 479 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 479
{ } 
#endif
# 482 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 482
{ } 
#endif
# 485 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 485
{ } 
#endif
# 488 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 488
{ } 
#endif
# 491 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 491
{ } 
#endif
# 494 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 494
{ } 
#endif
# 497 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 497
{ } 
#endif
# 500 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 500
{ } 
#endif
# 503 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 503
{ } 
#endif
# 506 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 506
{ } 
#endif
# 509 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 509
{ } 
#endif
# 512 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 512
{ } 
#endif
# 515 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 515
{ } 
#endif
# 518 "/usr/local/cuda-8.0/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 518
{ } 
#endif
# 79 "/usr/local/cuda-8.0/include/sm_20_intrinsics.h"
extern "C" {
# 1466
}
# 1475
__attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1475
{ } 
#endif
# 1477 "/usr/local/cuda-8.0/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1477
{ } 
#endif
# 1479 "/usr/local/cuda-8.0/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1479
{ } 
#endif
# 1481 "/usr/local/cuda-8.0/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1481
{ } 
#endif
# 1486 "/usr/local/cuda-8.0/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1486
{ } 
#endif
# 98 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 98
{ } 
#endif
# 100 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 103
{ } 
#endif
# 105 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 105
{ } 
#endif
# 107 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 117
{ } 
#endif
# 119 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 119
{ } 
#endif
# 122 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 128
{ } 
#endif
# 130 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 132 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 132
{ } 
#endif
# 134 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 134
{ } 
#endif
# 136 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 138 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 138
{ } 
#endif
# 140 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 142 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 148 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 150 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 152 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 152
{ } 
#endif
# 154 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 156 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 158 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 160 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 160
{ } 
#endif
# 162 "/usr/local/cuda-8.0/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 89 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 105 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 114
{ } 
#endif
# 115 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 120 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 120
{ } 
#endif
# 121 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 121
{ } 
#endif
# 125 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 125
{ } 
#endif
# 126 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 138 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 138
{ } 
#endif
# 139 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 141 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 153 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 157
{ } 
#endif
# 161 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 161
{ } 
#endif
# 162 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 164 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 174 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 174
{ } 
#endif
# 175 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 177 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 186 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 186
{ } 
#endif
# 187 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 189 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 192 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 192
{ } 
#endif
# 193 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 193
{ } 
#endif
# 197 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 197
{ } 
#endif
# 198 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 200 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 210 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 210
{ } 
#endif
# 211 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 213 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 222 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 222
{ } 
#endif
# 223 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 225 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 228 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 228
{ } 
#endif
# 229 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 229
{ } 
#endif
# 236 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 236
{ } 
#endif
# 237 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 237
{ } 
#endif
# 240 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 240
{ } 
#endif
# 241 "/usr/local/cuda-8.0/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 241
{ } 
#endif
# 91 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 94 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 97 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 100 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 101
{ } 
#endif
# 108 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/local/cuda-8.0/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 112
{ } 
#endif
# 100 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 101
__attribute((always_inline)) __attribute__((unused)) inline void surf1Dread(T *res, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 102
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
# 111
::exit(___);}
#if 0
# 102
{ 
# 111
} 
#endif
# 113 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 114
__attribute((always_inline)) __attribute__((unused)) inline T surf1Dread(surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 115
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
# 123
::exit(___);}
#if 0
# 115
{ 
# 123
} 
#endif
# 125 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 126
__attribute((always_inline)) __attribute__((unused)) inline void surf1Dread(T *res, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 127
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
# 131
::exit(___);}
#if 0
# 127
{ 
# 131
} 
#endif
# 260 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 262
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 271
::exit(___);}
#if 0
# 262
{ 
# 271
} 
#endif
# 273 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 274
__attribute((always_inline)) __attribute__((unused)) inline T surf2Dread(surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 275
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
# 283
::exit(___);}
#if 0
# 275
{ 
# 283
} 
#endif
# 285 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 286
__attribute((always_inline)) __attribute__((unused)) inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 287
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
# 291
::exit(___);}
#if 0
# 287
{ 
# 291
} 
#endif
# 422 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 423
__attribute((always_inline)) __attribute__((unused)) inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 424
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 433
::exit(___);}
#if 0
# 424
{ 
# 433
} 
#endif
# 435 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 436
__attribute((always_inline)) __attribute__((unused)) inline T surf3Dread(surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 437
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 445
::exit(___);}
#if 0
# 437
{ 
# 445
} 
#endif
# 447 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 448
__attribute((always_inline)) __attribute__((unused)) inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 449
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 453
::exit(___);}
#if 0
# 449
{ 
# 453
} 
#endif
# 582 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 583
__attribute((always_inline)) __attribute__((unused)) inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 584
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 593
::exit(___);}
#if 0
# 584
{ 
# 593
} 
#endif
# 595 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 596
__attribute((always_inline)) __attribute__((unused)) inline T surf1DLayeredread(surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 597
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
# 605
::exit(___);}
#if 0
# 597
{ 
# 605
} 
#endif
# 607 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 608
__attribute((always_inline)) __attribute__((unused)) inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 609
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
# 613
::exit(___);}
#if 0
# 609
{ 
# 613
} 
#endif
# 768 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 769
__attribute((always_inline)) __attribute__((unused)) inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 770
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 779
::exit(___);}
#if 0
# 770
{ 
# 779
} 
#endif
# 781 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 782
__attribute((always_inline)) __attribute__((unused)) inline T surf2DLayeredread(surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 783
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 791
::exit(___);}
#if 0
# 783
{ 
# 791
} 
#endif
# 793 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 794
__attribute((always_inline)) __attribute__((unused)) inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 795
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 799
::exit(___);}
#if 0
# 795
{ 
# 799
} 
#endif
# 919 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 920
__attribute((always_inline)) __attribute__((unused)) inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 921
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 930
::exit(___);}
#if 0
# 921
{ 
# 930
} 
#endif
# 932 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 933
__attribute((always_inline)) __attribute__((unused)) inline T surfCubemapread(surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 934
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 942
::exit(___);}
#if 0
# 934
{ 
# 942
} 
#endif
# 944 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 945
__attribute((always_inline)) __attribute__((unused)) inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 946
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 950
::exit(___);}
#if 0
# 946
{ 
# 950
} 
#endif
# 1070 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1071
__attribute((always_inline)) __attribute__((unused)) inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1072
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 1081
::exit(___);}
#if 0
# 1072
{ 
# 1081
} 
#endif
# 1083 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1084
__attribute((always_inline)) __attribute__((unused)) inline T surfCubemapLayeredread(surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1085
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 1093
::exit(___);}
#if 0
# 1085
{ 
# 1093
} 
#endif
# 1095 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1096
__attribute((always_inline)) __attribute__((unused)) inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1097
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 1101
::exit(___);}
#if 0
# 1097
{ 
# 1101
} 
#endif
# 1232 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1233
__attribute((always_inline)) __attribute__((unused)) inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1234
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
# 1254
::exit(___);}
#if 0
# 1234
{ 
# 1254
} 
#endif
# 1256 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1257
__attribute((always_inline)) __attribute__((unused)) inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1258
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
# 1262
::exit(___);}
#if 0
# 1258
{ 
# 1262
} 
#endif
# 1377 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1378
__attribute((always_inline)) __attribute__((unused)) inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1379
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 1399
::exit(___);}
#if 0
# 1379
{ 
# 1399
} 
#endif
# 1401 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1402
__attribute((always_inline)) __attribute__((unused)) inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1403
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
# 1407
::exit(___);}
#if 0
# 1403
{ 
# 1407
} 
#endif
# 1520 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1521
__attribute((always_inline)) __attribute__((unused)) inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1522
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 1542
::exit(___);}
#if 0
# 1522
{ 
# 1542
} 
#endif
# 1544 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1545
__attribute((always_inline)) __attribute__((unused)) inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1546
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 1550
::exit(___);}
#if 0
# 1546
{ 
# 1550
} 
#endif
# 1666 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1667
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1668
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 1688
::exit(___);}
#if 0
# 1668
{ 
# 1688
} 
#endif
# 1690 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1691
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1692
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
# 1696
::exit(___);}
#if 0
# 1692
{ 
# 1696
} 
#endif
# 1822 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1823
__attribute((always_inline)) __attribute__((unused)) inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1824
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 1844
::exit(___);}
#if 0
# 1824
{ 
# 1844
} 
#endif
# 1846 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1847
__attribute((always_inline)) __attribute__((unused)) inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1848
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 1852
::exit(___);}
#if 0
# 1848
{ 
# 1852
} 
#endif
# 1958 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1959
__attribute((always_inline)) __attribute__((unused)) inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1960
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 1980
::exit(___);}
#if 0
# 1960
{ 
# 1980
} 
#endif
# 1982 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 1983
__attribute((always_inline)) __attribute__((unused)) inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 1984
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 1988
::exit(___);}
#if 0
# 1984
{ 
# 1988
} 
#endif
# 2093 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 2094
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 2095
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 2115
::exit(___);}
#if 0
# 2095
{ 
# 2115
} 
#endif
# 2117 "/usr/local/cuda-8.0/include/surface_functions.h"
template< class T> 
# 2118
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 2119
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 2123
::exit(___);}
#if 0
# 2119
{ 
# 2123
} 
#endif
# 70 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 71
tex1Dfetch(texture< T, 1, cudaReadModeElementType> , int) {int volatile ___ = 1;::exit(___);}
#if 0
# 71
{ } 
#endif
# 73 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> 
# 74
struct __nv_tex_rmnf_ret { }; 
# 76
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
# 77
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
# 78
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
# 79
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
# 80
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
# 81
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
# 82
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
# 83
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
# 84
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
# 85
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
# 86
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
# 87
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
# 88
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
# 89
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
# 90
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 91
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
# 92
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 94
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 95
tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat> , int) {int volatile ___ = 1;::exit(___);}
#if 0
# 95
{ } 
#endif
# 215 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 216
tex1D(texture< T, 1, cudaReadModeElementType> , float) {int volatile ___ = 1;::exit(___);}
#if 0
# 216
{ } 
#endif
# 218 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 219
tex1D(texture< T, 1, cudaReadModeNormalizedFloat> , float) {int volatile ___ = 1;::exit(___);}
#if 0
# 219
{ } 
#endif
# 345 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 346
tex2D(texture< T, 2, cudaReadModeElementType> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 346
{ } 
#endif
# 348 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 349
tex2D(texture< T, 2, cudaReadModeNormalizedFloat> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 349
{ } 
#endif
# 475 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 476
tex1DLayered(texture< T, 241, cudaReadModeElementType> , float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 476
{ } 
#endif
# 478 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 479
tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat> , float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 479
{ } 
#endif
# 603 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 604
tex2DLayered(texture< T, 242, cudaReadModeElementType> , float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 604
{ } 
#endif
# 606 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 607
tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat> , float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 607
{ } 
#endif
# 735 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 736
tex3D(texture< T, 3, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 736
{ } 
#endif
# 738 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 739
tex3D(texture< T, 3, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 739
{ } 
#endif
# 864 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 865
texCubemap(texture< T, 12, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 865
{ } 
#endif
# 867 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 868
texCubemap(texture< T, 12, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 868
{ } 
#endif
# 992 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 993
texCubemapLayered(texture< T, 252, cudaReadModeElementType> , float, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 993
{ } 
#endif
# 995 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 996
texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat> , float, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 996
{ } 
#endif
# 1121 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> 
# 1122
struct __nv_tex2dgather_ret { }; 
# 1123
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
# 1124
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
# 1125
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
# 1126
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
# 1127
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
# 1128
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
# 1129
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
# 1130
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
# 1131
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
# 1132
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
# 1133
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
# 1135
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
# 1136
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
# 1137
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
# 1138
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
# 1139
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
# 1140
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
# 1141
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
# 1142
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
# 1143
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
# 1144
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
# 1146
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
# 1147
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
# 1148
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
# 1149
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
# 1150
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
# 1151
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
# 1152
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
# 1153
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
# 1154
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
# 1155
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
# 1157
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
# 1158
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
# 1159
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
# 1160
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
# 1161
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
# 1163
template< class T> __attribute__((unused)) static typename __nv_tex2dgather_ret< T> ::type 
# 1164
tex2Dgather(texture< T, 2, cudaReadModeElementType> , float, float, int = 0) {int volatile ___ = 1;::exit(___);}
#if 0
# 1164
{ } 
#endif
# 1166 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static float4 
# 1167
tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat> , float, float, int = 0) {int volatile ___ = 1;::exit(___);}
#if 0
# 1167
{ } 
#endif
# 1232 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 1233
tex1DLod(texture< T, 1, cudaReadModeElementType> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1233
{ } 
#endif
# 1235 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 1236
tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat> , float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1236
{ } 
#endif
# 1360 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 1361
tex2DLod(texture< T, 2, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1361
{ } 
#endif
# 1363 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 1364
tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1364
{ } 
#endif
# 1484 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 1485
tex1DLayeredLod(texture< T, 241, cudaReadModeElementType> , float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1485
{ } 
#endif
# 1487 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 1488
tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat> , float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1488
{ } 
#endif
# 1612 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 1613
tex2DLayeredLod(texture< T, 242, cudaReadModeElementType> , float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1613
{ } 
#endif
# 1615 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 1616
tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat> , float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1616
{ } 
#endif
# 1740 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 1741
tex3DLod(texture< T, 3, cudaReadModeElementType> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1741
{ } 
#endif
# 1743 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 1744
tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1744
{ } 
#endif
# 1868 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 1869
texCubemapLod(texture< T, 12, cudaReadModeElementType> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1869
{ } 
#endif
# 1871 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 1872
texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat> , float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1872
{ } 
#endif
# 1996 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 1997
texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType> , float, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1997
{ } 
#endif
# 1999 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 2000
texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat> , float, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 2000
{ } 
#endif
# 2124 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 2125
tex1DGrad(texture< T, 1, cudaReadModeElementType> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 2125
{ } 
#endif
# 2127 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 2128
tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat> , float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 2128
{ } 
#endif
# 2252 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 2253
tex2DGrad(texture< T, 2, cudaReadModeElementType> , float, float, float2, float2) {int volatile ___ = 1;::exit(___);}
#if 0
# 2253
{ } 
#endif
# 2255 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 2256
tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat> , float, float, float2, float2) {int volatile ___ = 1;::exit(___);}
#if 0
# 2256
{ } 
#endif
# 2380 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 2381
tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType> , float, int, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 2381
{ } 
#endif
# 2383 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 2384
tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat> , float, int, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 2384
{ } 
#endif
# 2509 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 2510
tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType> , float, float, int, float2, float2) {int volatile ___ = 1;::exit(___);}
#if 0
# 2510
{ } 
#endif
# 2512 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 2513
tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat> , float, float, int, float2, float2) {int volatile ___ = 1;::exit(___);}
#if 0
# 2513
{ } 
#endif
# 2637 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static T 
# 2638
tex3DGrad(texture< T, 3, cudaReadModeElementType> , float, float, float, float4, float4) {int volatile ___ = 1;::exit(___);}
#if 0
# 2638
{ } 
#endif
# 2640 "/usr/local/cuda-8.0/include/texture_fetch_functions.h"
template< class T> __attribute__((unused)) static typename __nv_tex_rmnf_ret< T> ::type 
# 2641
tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat> , float, float, float, float4, float4) {int volatile ___ = 1;::exit(___);}
#if 0
# 2641
{ } 
#endif
# 67 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 68
tex1Dfetch(T *, cudaTextureObject_t, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 68
{ } 
#endif
# 121 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 122
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 123
{int volatile ___ = 1;(void)texObject;(void)x;
# 127
::exit(___);}
#if 0
# 123
{ 
# 124
T ret; 
# 125
tex1Dfetch(&ret, texObject, x); 
# 126
return ret; 
# 127
} 
#endif
# 135 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 136
tex1D(T *, cudaTextureObject_t, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 136
{ } 
#endif
# 190 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 191
tex1D(cudaTextureObject_t texObject, float x) 
# 192
{int volatile ___ = 1;(void)texObject;(void)x;
# 196
::exit(___);}
#if 0
# 192
{ 
# 193
T ret; 
# 194
tex1D(&ret, texObject, x); 
# 195
return ret; 
# 196
} 
#endif
# 205 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 206
tex2D(T *, cudaTextureObject_t, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 206
{ } 
#endif
# 258 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 259
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 260
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 264
::exit(___);}
#if 0
# 260
{ 
# 261
T ret; 
# 262
tex2D(&ret, texObject, x, y); 
# 263
return ret; 
# 264
} 
#endif
# 272 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 273
tex3D(T *, cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 273
{ } 
#endif
# 325 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 326
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 327
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 331
::exit(___);}
#if 0
# 327
{ 
# 328
T ret; 
# 329
tex3D(&ret, texObject, x, y, z); 
# 330
return ret; 
# 331
} 
#endif
# 340 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 341
tex1DLayered(T *, cudaTextureObject_t, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 341
{ } 
#endif
# 393 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 394
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 395
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 399
::exit(___);}
#if 0
# 395
{ 
# 396
T ret; 
# 397
tex1DLayered(&ret, texObject, x, layer); 
# 398
return ret; 
# 399
} 
#endif
# 408 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 409
tex2DLayered(T *, cudaTextureObject_t, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 409
{ } 
#endif
# 461 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 462
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 463
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 467
::exit(___);}
#if 0
# 463
{ 
# 464
T ret; 
# 465
tex2DLayered(&ret, texObject, x, y, layer); 
# 466
return ret; 
# 467
} 
#endif
# 476 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 477
texCubemap(T *, cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 477
{ } 
#endif
# 529 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 530
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 531
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 535
::exit(___);}
#if 0
# 531
{ 
# 532
T ret; 
# 533
texCubemap(&ret, texObject, x, y, z); 
# 534
return ret; 
# 535
} 
#endif
# 544 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 545
texCubemapLayered(T *, cudaTextureObject_t, float, float, float, int) {int volatile ___ = 1;::exit(___);}
#if 0
# 545
{ } 
#endif
# 598 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 599
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 600
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 604
::exit(___);}
#if 0
# 600
{ 
# 601
T ret; 
# 602
texCubemapLayered(&ret, texObject, x, y, z, layer); 
# 603
return ret; 
# 604
} 
#endif
# 613 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 614
tex2Dgather(T *, cudaTextureObject_t, float, float, int = 0) {int volatile ___ = 1;::exit(___);}
#if 0
# 614
{ } 
#endif
# 660 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 661
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 662
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 666
::exit(___);}
#if 0
# 662
{ 
# 663
T ret; 
# 664
tex2Dgather(&ret, to, x, y, comp); 
# 665
return ret; 
# 666
} 
#endif
# 675 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 676
tex1DLod(T *, cudaTextureObject_t, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 676
{ } 
#endif
# 728 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 729
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 730
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 734
::exit(___);}
#if 0
# 730
{ 
# 731
T ret; 
# 732
tex1DLod(&ret, texObject, x, level); 
# 733
return ret; 
# 734
} 
#endif
# 743 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 744
tex2DLod(T *, cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 744
{ } 
#endif
# 797 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 798
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 799
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 803
::exit(___);}
#if 0
# 799
{ 
# 800
T ret; 
# 801
tex2DLod(&ret, texObject, x, y, level); 
# 802
return ret; 
# 803
} 
#endif
# 812 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 813
tex3DLod(T *, cudaTextureObject_t, float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 813
{ } 
#endif
# 865 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 866
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 867
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 871
::exit(___);}
#if 0
# 867
{ 
# 868
T ret; 
# 869
tex3DLod(&ret, texObject, x, y, z, level); 
# 870
return ret; 
# 871
} 
#endif
# 879 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 880
tex1DLayeredLod(T *, cudaTextureObject_t, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 880
{ } 
#endif
# 932 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 933
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 934
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 938
::exit(___);}
#if 0
# 934
{ 
# 935
T ret; 
# 936
tex1DLayeredLod(&ret, texObject, x, layer, level); 
# 937
return ret; 
# 938
} 
#endif
# 947 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 948
tex2DLayeredLod(T *, cudaTextureObject_t, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 948
{ } 
#endif
# 1000 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1001
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 1002
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 1006
::exit(___);}
#if 0
# 1002
{ 
# 1003
T ret; 
# 1004
tex2DLayeredLod(&ret, texObject, x, y, layer, level); 
# 1005
return ret; 
# 1006
} 
#endif
# 1014 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 1015
texCubemapLod(T *, cudaTextureObject_t, float, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1015
{ } 
#endif
# 1067 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1068
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 1069
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 1073
::exit(___);}
#if 0
# 1069
{ 
# 1070
T ret; 
# 1071
texCubemapLod(&ret, texObject, x, y, z, level); 
# 1072
return ret; 
# 1073
} 
#endif
# 1081 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 1082
texCubemapLayeredLod(T *, cudaTextureObject_t, float, float, float, int, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1082
{ } 
#endif
# 1134 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1135
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 1136
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 1140
::exit(___);}
#if 0
# 1136
{ 
# 1137
T ret; 
# 1138
texCubemapLayeredLod(&ret, texObject, x, y, z, layer, level); 
# 1139
return ret; 
# 1140
} 
#endif
# 1148 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 1149
tex1DGrad(T *, cudaTextureObject_t, float, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1149
{ } 
#endif
# 1202 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1203
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 1204
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 1208
::exit(___);}
#if 0
# 1204
{ 
# 1205
T ret; 
# 1206
tex1DGrad(&ret, texObject, x, dPdx, dPdy); 
# 1207
return ret; 
# 1208
} 
#endif
# 1216 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 1217
tex2DGrad(T *, cudaTextureObject_t, float, float, float2, float2) {int volatile ___ = 1;::exit(___);}
#if 0
# 1217
{ } 
#endif
# 1269 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1270
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 1271
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 1275
::exit(___);}
#if 0
# 1271
{ 
# 1272
T ret; 
# 1273
tex2DGrad(&ret, texObject, x, y, dPdx, dPdy); 
# 1274
return ret; 
# 1275
} 
#endif
# 1283 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 1284
tex3DGrad(T *, cudaTextureObject_t, float, float, float, float4, float4) {int volatile ___ = 1;::exit(___);}
#if 0
# 1284
{ } 
#endif
# 1336 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1337
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 1338
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 1342
::exit(___);}
#if 0
# 1338
{ 
# 1339
T ret; 
# 1340
tex3DGrad(&ret, texObject, x, y, z, dPdx, dPdy); 
# 1341
return ret; 
# 1342
} 
#endif
# 1350 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 1351
tex1DLayeredGrad(T *, cudaTextureObject_t, float, int, float, float) {int volatile ___ = 1;::exit(___);}
#if 0
# 1351
{ } 
#endif
# 1404 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1405
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 1406
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 1410
::exit(___);}
#if 0
# 1406
{ 
# 1407
T ret; 
# 1408
tex1DLayeredGrad(&ret, texObject, x, layer, dPdx, dPdy); 
# 1409
return ret; 
# 1410
} 
#endif
# 1418 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 1419
tex2DLayeredGrad(T *, cudaTextureObject_t, float, float, int, float2, float2) {int volatile ___ = 1;::exit(___);}
#if 0
# 1419
{ } 
#endif
# 1471 "/usr/local/cuda-8.0/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 1472
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 1473
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 1477
::exit(___);}
#if 0
# 1473
{ 
# 1474
T ret; 
# 1475
tex2DLayeredGrad(&ret, texObject, x, y, layer, dPdx, dPdy); 
# 1476
return ret; 
# 1477
} 
#endif
# 68 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 69
surf1Dread(T *, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 69
{ } 
#endif
# 111 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 112
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 113
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 119
::exit(___);}
#if 0
# 113
{ 
# 119
} 
#endif
# 128 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 129
surf2Dread(T *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 129
{ } 
#endif
# 172 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 173
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 174
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 180
::exit(___);}
#if 0
# 174
{ 
# 180
} 
#endif
# 189 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 190
surf3Dread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 190
{ } 
#endif
# 231 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 232
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 233
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 239
::exit(___);}
#if 0
# 233
{ 
# 239
} 
#endif
# 247 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 248
surf1DLayeredread(T *, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 248
{ } 
#endif
# 290 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 291
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 292
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 298
::exit(___);}
#if 0
# 292
{ 
# 298
} 
#endif
# 306 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 307
surf2DLayeredread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 307
{ } 
#endif
# 348 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 349
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 350
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 356
::exit(___);}
#if 0
# 350
{ 
# 356
} 
#endif
# 364 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 365
surfCubemapread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 365
{ } 
#endif
# 406 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 407
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 408
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 414
::exit(___);}
#if 0
# 408
{ 
# 414
} 
#endif
# 422 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 423
surfCubemapLayeredread(T *, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 423
{ } 
#endif
# 464 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 465
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 466
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 472
::exit(___);}
#if 0
# 466
{ 
# 472
} 
#endif
# 480 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 481
surf1Dwrite(T, cudaSurfaceObject_t, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 481
{ } 
#endif
# 528 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 529
surf2Dwrite(T, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 529
{ } 
#endif
# 576 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 577
surf3Dwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 577
{ } 
#endif
# 626 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 627
surf1DLayeredwrite(T, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 627
{ } 
#endif
# 675 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 676
surf2DLayeredwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 676
{ } 
#endif
# 723 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 724
surfCubemapwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 724
{ } 
#endif
# 771 "/usr/local/cuda-8.0/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static void 
# 772
surfCubemapLayeredwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;::exit(___);}
#if 0
# 772
{ } 
#endif
# 68 "/usr/local/cuda-8.0/include/device_launch_parameters.h"
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 183 "/usr/local/cuda-8.0/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 184
cudaLaunchKernel(const T *
# 185
func, dim3 
# 186
gridDim, dim3 
# 187
blockDim, void **
# 188
args, size_t 
# 189
sharedMem = 0, cudaStream_t 
# 190
stream = 0) 
# 192
{ 
# 193
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 194
} 
# 221
template< class T> static inline cudaError_t 
# 222
cudaSetupArgument(T 
# 223
arg, size_t 
# 224
offset) 
# 226
{ 
# 227
return ::cudaSetupArgument((const void *)(&arg), sizeof(T), offset); 
# 228
} 
# 260
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 261
event, unsigned 
# 262
flags) 
# 264
{ 
# 265
return ::cudaEventCreateWithFlags(event, flags); 
# 266
} 
# 323
static inline cudaError_t cudaMallocHost(void **
# 324
ptr, size_t 
# 325
size, unsigned 
# 326
flags) 
# 328
{ 
# 329
return ::cudaHostAlloc(ptr, size, flags); 
# 330
} 
# 332
template< class T> static inline cudaError_t 
# 333
cudaHostAlloc(T **
# 334
ptr, size_t 
# 335
size, unsigned 
# 336
flags) 
# 338
{ 
# 339
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 340
} 
# 342
template< class T> static inline cudaError_t 
# 343
cudaHostGetDevicePointer(T **
# 344
pDevice, void *
# 345
pHost, unsigned 
# 346
flags) 
# 348
{ 
# 349
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 350
} 
# 449
template< class T> static inline cudaError_t 
# 450
cudaMallocManaged(T **
# 451
devPtr, size_t 
# 452
size, unsigned 
# 453
flags = 1) 
# 455
{ 
# 456
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 457
} 
# 528
template< class T> static inline cudaError_t 
# 529
cudaStreamAttachMemAsync(cudaStream_t 
# 530
stream, T *
# 531
devPtr, size_t 
# 532
length = 0, unsigned 
# 533
flags = 4) 
# 535
{ 
# 536
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 537
} 
# 539
template< class T> inline cudaError_t 
# 540
cudaMalloc(T **
# 541
devPtr, size_t 
# 542
size) 
# 544
{ 
# 545
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 546
} 
# 548
template< class T> static inline cudaError_t 
# 549
cudaMallocHost(T **
# 550
ptr, size_t 
# 551
size, unsigned 
# 552
flags = 0) 
# 554
{ 
# 555
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 556
} 
# 558
template< class T> static inline cudaError_t 
# 559
cudaMallocPitch(T **
# 560
devPtr, size_t *
# 561
pitch, size_t 
# 562
width, size_t 
# 563
height) 
# 565
{ 
# 566
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 567
} 
# 604
template< class T> static inline cudaError_t 
# 605
cudaMemcpyToSymbol(const T &
# 606
symbol, const void *
# 607
src, size_t 
# 608
count, size_t 
# 609
offset = 0, cudaMemcpyKind 
# 610
kind = cudaMemcpyHostToDevice) 
# 612
{ 
# 613
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 614
} 
# 656
template< class T> static inline cudaError_t 
# 657
cudaMemcpyToSymbolAsync(const T &
# 658
symbol, const void *
# 659
src, size_t 
# 660
count, size_t 
# 661
offset = 0, cudaMemcpyKind 
# 662
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 663
stream = 0) 
# 665
{ 
# 666
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 667
} 
# 702
template< class T> static inline cudaError_t 
# 703
cudaMemcpyFromSymbol(void *
# 704
dst, const T &
# 705
symbol, size_t 
# 706
count, size_t 
# 707
offset = 0, cudaMemcpyKind 
# 708
kind = cudaMemcpyDeviceToHost) 
# 710
{ 
# 711
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 712
} 
# 754
template< class T> static inline cudaError_t 
# 755
cudaMemcpyFromSymbolAsync(void *
# 756
dst, const T &
# 757
symbol, size_t 
# 758
count, size_t 
# 759
offset = 0, cudaMemcpyKind 
# 760
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 761
stream = 0) 
# 763
{ 
# 764
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 765
} 
# 787
template< class T> static inline cudaError_t 
# 788
cudaGetSymbolAddress(void **
# 789
devPtr, const T &
# 790
symbol) 
# 792
{ 
# 793
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 794
} 
# 816
template< class T> static inline cudaError_t 
# 817
cudaGetSymbolSize(size_t *
# 818
size, const T &
# 819
symbol) 
# 821
{ 
# 822
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 823
} 
# 859
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 860
cudaBindTexture(size_t *
# 861
offset, const texture< T, dim, readMode>  &
# 862
tex, const void *
# 863
devPtr, const cudaChannelFormatDesc &
# 864
desc, size_t 
# 865
size = ((2147483647) * 2U) + 1U) 
# 867
{ 
# 868
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
# 869
} 
# 904
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 905
cudaBindTexture(size_t *
# 906
offset, const texture< T, dim, readMode>  &
# 907
tex, const void *
# 908
devPtr, size_t 
# 909
size = ((2147483647) * 2U) + 1U) 
# 911
{ 
# 912
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 913
} 
# 960
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 961
cudaBindTexture2D(size_t *
# 962
offset, const texture< T, dim, readMode>  &
# 963
tex, const void *
# 964
devPtr, const cudaChannelFormatDesc &
# 965
desc, size_t 
# 966
width, size_t 
# 967
height, size_t 
# 968
pitch) 
# 970
{ 
# 971
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
# 972
} 
# 1018
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1019
cudaBindTexture2D(size_t *
# 1020
offset, const texture< T, dim, readMode>  &
# 1021
tex, const void *
# 1022
devPtr, size_t 
# 1023
width, size_t 
# 1024
height, size_t 
# 1025
pitch) 
# 1027
{ 
# 1028
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
# 1029
} 
# 1060
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1061
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1062
tex, cudaArray_const_t 
# 1063
array, const cudaChannelFormatDesc &
# 1064
desc) 
# 1066
{ 
# 1067
return ::cudaBindTextureToArray(&tex, array, &desc); 
# 1068
} 
# 1098
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1099
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1100
tex, cudaArray_const_t 
# 1101
array) 
# 1103
{ 
# 1104
cudaChannelFormatDesc desc; 
# 1105
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1107
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
# 1108
} 
# 1139
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1140
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1141
tex, cudaMipmappedArray_const_t 
# 1142
mipmappedArray, const cudaChannelFormatDesc &
# 1143
desc) 
# 1145
{ 
# 1146
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
# 1147
} 
# 1177
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1178
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1179
tex, cudaMipmappedArray_const_t 
# 1180
mipmappedArray) 
# 1182
{ 
# 1183
cudaChannelFormatDesc desc; 
# 1184
cudaArray_t levelArray; 
# 1185
cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
# 1187
if (err != (cudaSuccess)) { 
# 1188
return err; 
# 1189
}  
# 1190
err = ::cudaGetChannelDesc(&desc, levelArray); 
# 1192
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
# 1193
} 
# 1216
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1217
cudaUnbindTexture(const texture< T, dim, readMode>  &
# 1218
tex) 
# 1220
{ 
# 1221
return ::cudaUnbindTexture(&tex); 
# 1222
} 
# 1250
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1251
cudaGetTextureAlignmentOffset(size_t *
# 1252
offset, const texture< T, dim, readMode>  &
# 1253
tex) 
# 1255
{ 
# 1256
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
# 1257
} 
# 1302
template< class T> static inline cudaError_t 
# 1303
cudaFuncSetCacheConfig(T *
# 1304
func, cudaFuncCache 
# 1305
cacheConfig) 
# 1307
{ 
# 1308
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1309
} 
# 1311
template< class T> static inline cudaError_t 
# 1312
cudaFuncSetSharedMemConfig(T *
# 1313
func, cudaSharedMemConfig 
# 1314
config) 
# 1316
{ 
# 1317
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1318
} 
# 1347
template< class T> inline cudaError_t 
# 1348
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1349
numBlocks, T 
# 1350
func, int 
# 1351
blockSize, size_t 
# 1352
dynamicSMemSize) 
# 1353
{ 
# 1354
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1355
} 
# 1398
template< class T> inline cudaError_t 
# 1399
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1400
numBlocks, T 
# 1401
func, int 
# 1402
blockSize, size_t 
# 1403
dynamicSMemSize, unsigned 
# 1404
flags) 
# 1405
{ 
# 1406
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1407
} 
# 1412
class __cudaOccupancyB2DHelper { 
# 1413
size_t n; 
# 1415
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1416
size_t operator()(int) 
# 1417
{ 
# 1418
return n; 
# 1419
} 
# 1420
}; 
# 1467
template< class UnaryFunction, class T> static inline cudaError_t 
# 1468
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 1469
minGridSize, int *
# 1470
blockSize, T 
# 1471
func, UnaryFunction 
# 1472
blockSizeToDynamicSMemSize, int 
# 1473
blockSizeLimit = 0, unsigned 
# 1474
flags = 0) 
# 1475
{ 
# 1476
cudaError_t status; 
# 1479
int device; 
# 1480
cudaFuncAttributes attr; 
# 1483
int maxThreadsPerMultiProcessor; 
# 1484
int warpSize; 
# 1485
int devMaxThreadsPerBlock; 
# 1486
int multiProcessorCount; 
# 1487
int funcMaxThreadsPerBlock; 
# 1488
int occupancyLimit; 
# 1489
int granularity; 
# 1492
int maxBlockSize = 0; 
# 1493
int numBlocks = 0; 
# 1494
int maxOccupancy = 0; 
# 1497
int blockSizeToTryAligned; 
# 1498
int blockSizeToTry; 
# 1499
int blockSizeLimitAligned; 
# 1500
int occupancyInBlocks; 
# 1501
int occupancyInThreads; 
# 1502
size_t dynamicSMemSize; 
# 1508
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 1509
return cudaErrorInvalidValue; 
# 1510
}  
# 1516
status = ::cudaGetDevice(&device); 
# 1517
if (status != (cudaSuccess)) { 
# 1518
return status; 
# 1519
}  
# 1521
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 1525
if (status != (cudaSuccess)) { 
# 1526
return status; 
# 1527
}  
# 1529
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 1533
if (status != (cudaSuccess)) { 
# 1534
return status; 
# 1535
}  
# 1537
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 1541
if (status != (cudaSuccess)) { 
# 1542
return status; 
# 1543
}  
# 1545
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 1549
if (status != (cudaSuccess)) { 
# 1550
return status; 
# 1551
}  
# 1553
status = cudaFuncGetAttributes(&attr, func); 
# 1554
if (status != (cudaSuccess)) { 
# 1555
return status; 
# 1556
}  
# 1558
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 1564
occupancyLimit = maxThreadsPerMultiProcessor; 
# 1565
granularity = warpSize; 
# 1567
if (blockSizeLimit == 0) { 
# 1568
blockSizeLimit = devMaxThreadsPerBlock; 
# 1569
}  
# 1571
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 1572
blockSizeLimit = devMaxThreadsPerBlock; 
# 1573
}  
# 1575
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 1576
blockSizeLimit = funcMaxThreadsPerBlock; 
# 1577
}  
# 1579
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 1581
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 1585
if (blockSizeLimit < blockSizeToTryAligned) { 
# 1586
blockSizeToTry = blockSizeLimit; 
# 1587
} else { 
# 1588
blockSizeToTry = blockSizeToTryAligned; 
# 1589
}  
# 1591
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 1593
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 1600
if (status != (cudaSuccess)) { 
# 1601
return status; 
# 1602
}  
# 1604
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 1606
if (occupancyInThreads > maxOccupancy) { 
# 1607
maxBlockSize = blockSizeToTry; 
# 1608
numBlocks = occupancyInBlocks; 
# 1609
maxOccupancy = occupancyInThreads; 
# 1610
}  
# 1614
if (occupancyLimit == maxOccupancy) { 
# 1615
break; 
# 1616
}  
# 1617
}  
# 1625
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 1626
(*blockSize) = maxBlockSize; 
# 1628
return status; 
# 1629
} 
# 1662
template< class UnaryFunction, class T> static inline cudaError_t 
# 1663
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 1664
minGridSize, int *
# 1665
blockSize, T 
# 1666
func, UnaryFunction 
# 1667
blockSizeToDynamicSMemSize, int 
# 1668
blockSizeLimit = 0) 
# 1669
{ 
# 1670
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 1671
} 
# 1707
template< class T> static inline cudaError_t 
# 1708
cudaOccupancyMaxPotentialBlockSize(int *
# 1709
minGridSize, int *
# 1710
blockSize, T 
# 1711
func, size_t 
# 1712
dynamicSMemSize = 0, int 
# 1713
blockSizeLimit = 0) 
# 1714
{ 
# 1715
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 1716
} 
# 1766
template< class T> static inline cudaError_t 
# 1767
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 1768
minGridSize, int *
# 1769
blockSize, T 
# 1770
func, size_t 
# 1771
dynamicSMemSize = 0, int 
# 1772
blockSizeLimit = 0, unsigned 
# 1773
flags = 0) 
# 1774
{ 
# 1775
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 1776
} 
# 1814
template< class T> static inline cudaError_t 
# 1815
cudaLaunch(T *
# 1816
func) 
# 1818
{ 
# 1819
return ::cudaLaunch((const void *)func); 
# 1820
} 
# 1851
template< class T> inline cudaError_t 
# 1852
cudaFuncGetAttributes(cudaFuncAttributes *
# 1853
attr, T *
# 1854
entry) 
# 1856
{ 
# 1857
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 1858
} 
# 1880
template< class T, int dim> static inline cudaError_t 
# 1881
cudaBindSurfaceToArray(const surface< T, dim>  &
# 1882
surf, cudaArray_const_t 
# 1883
array, const cudaChannelFormatDesc &
# 1884
desc) 
# 1886
{ 
# 1887
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
# 1888
} 
# 1909
template< class T, int dim> static inline cudaError_t 
# 1910
cudaBindSurfaceToArray(const surface< T, dim>  &
# 1911
surf, cudaArray_const_t 
# 1912
array) 
# 1914
{ 
# 1915
cudaChannelFormatDesc desc; 
# 1916
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1918
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
# 1919
} 
# 1930
#pragma GCC diagnostic pop
# 50 "/usr/include/c++/4.8/bits/memoryfwd.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 63
template< class > class allocator; 
# 67
template<> class allocator< void> ; 
# 70
template< class , class > struct uses_allocator; 
# 76
}
# 42 "/usr/include/c++/4.8/bits/stringfwd.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 52
template< class _CharT> struct char_traits; 
# 55
template< class _CharT, class _Traits = char_traits< _CharT> , class 
# 56
_Alloc = allocator< _CharT> > class basic_string; 
# 59
template<> struct char_traits< char> ; 
# 62
typedef basic_string< char>  string; 
# 65
template<> struct char_traits< wchar_t> ; 
# 68
typedef basic_string< wchar_t>  wstring; 
# 74
template<> struct char_traits< char16_t> ; 
# 75
template<> struct char_traits< char32_t> ; 
# 78
typedef basic_string< char16_t>  u16string; 
# 81
typedef basic_string< char32_t>  u32string; 
# 87
}
# 44 "/usr/include/stdio.h" 3
struct _IO_FILE; 
# 48
typedef _IO_FILE FILE; 
# 64
typedef _IO_FILE __FILE; 
# 40 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stdarg.h" 3
typedef __builtin_va_list __gnuc_va_list; 
# 353 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3
typedef unsigned wint_t; 
# 94 "/usr/include/wchar.h" 3
typedef 
# 83
struct { 
# 84
int __count; 
# 86
union { 
# 88
unsigned __wch; 
# 92
char __wchb[4]; 
# 93
} __value; 
# 94
} __mbstate_t; 
# 106
typedef __mbstate_t mbstate_t; 
# 132
extern "C" {
# 137
struct tm; 
# 147
extern wchar_t *wcscpy(wchar_t *__restrict__ __dest, const wchar_t *__restrict__ __src) throw()
# 149
 __attribute((__nonnull__(1, 2))); 
# 152
extern wchar_t *wcsncpy(wchar_t *__restrict__ __dest, const wchar_t *__restrict__ __src, size_t __n) throw()
# 154
 __attribute((__nonnull__(1, 2))); 
# 157
extern wchar_t *wcscat(wchar_t *__restrict__ __dest, const wchar_t *__restrict__ __src) throw()
# 159
 __attribute((__nonnull__(1, 2))); 
# 161
extern wchar_t *wcsncat(wchar_t *__restrict__ __dest, const wchar_t *__restrict__ __src, size_t __n) throw()
# 163
 __attribute((__nonnull__(1, 2))); 
# 166
extern int wcscmp(const wchar_t * __s1, const wchar_t * __s2) throw()
# 167
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 169
extern int wcsncmp(const wchar_t * __s1, const wchar_t * __s2, size_t __n) throw()
# 170
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 175
extern int wcscasecmp(const wchar_t * __s1, const wchar_t * __s2) throw(); 
# 178
extern int wcsncasecmp(const wchar_t * __s1, const wchar_t * __s2, size_t __n) throw(); 
# 185
extern int wcscasecmp_l(const wchar_t * __s1, const wchar_t * __s2, __locale_t __loc) throw(); 
# 188
extern int wcsncasecmp_l(const wchar_t * __s1, const wchar_t * __s2, size_t __n, __locale_t __loc) throw(); 
# 195
extern int wcscoll(const wchar_t * __s1, const wchar_t * __s2) throw(); 
# 199
extern size_t wcsxfrm(wchar_t *__restrict__ __s1, const wchar_t *__restrict__ __s2, size_t __n) throw(); 
# 209
extern int wcscoll_l(const wchar_t * __s1, const wchar_t * __s2, __locale_t __loc) throw(); 
# 215
extern size_t wcsxfrm_l(wchar_t * __s1, const wchar_t * __s2, size_t __n, __locale_t __loc) throw(); 
# 219
extern wchar_t *wcsdup(const wchar_t * __s) throw() __attribute((__malloc__)); 
# 225
extern "C++" wchar_t *wcschr(wchar_t * __wcs, wchar_t __wc) throw() __asm__("wcschr")
# 226
 __attribute((__pure__)); 
# 227
extern "C++" const wchar_t *wcschr(const wchar_t * __wcs, wchar_t __wc) throw() __asm__("wcschr")
# 228
 __attribute((__pure__)); 
# 235
extern "C++" wchar_t *wcsrchr(wchar_t * __wcs, wchar_t __wc) throw() __asm__("wcsrchr")
# 236
 __attribute((__pure__)); 
# 237
extern "C++" const wchar_t *wcsrchr(const wchar_t * __wcs, wchar_t __wc) throw() __asm__("wcsrchr")
# 238
 __attribute((__pure__)); 
# 248
extern wchar_t *wcschrnul(const wchar_t * __s, wchar_t __wc) throw()
# 249
 __attribute((__pure__)); 
# 255
extern size_t wcscspn(const wchar_t * __wcs, const wchar_t * __reject) throw()
# 256
 __attribute((__pure__)); 
# 259
extern size_t wcsspn(const wchar_t * __wcs, const wchar_t * __accept) throw()
# 260
 __attribute((__pure__)); 
# 263
extern "C++" wchar_t *wcspbrk(wchar_t * __wcs, const wchar_t * __accept) throw() __asm__("wcspbrk")
# 264
 __attribute((__pure__)); 
# 265
extern "C++" const wchar_t *wcspbrk(const wchar_t * __wcs, const wchar_t * __accept) throw() __asm__("wcspbrk")
# 267
 __attribute((__pure__)); 
# 274
extern "C++" wchar_t *wcsstr(wchar_t * __haystack, const wchar_t * __needle) throw() __asm__("wcsstr")
# 275
 __attribute((__pure__)); 
# 276
extern "C++" const wchar_t *wcsstr(const wchar_t * __haystack, const wchar_t * __needle) throw() __asm__("wcsstr")
# 278
 __attribute((__pure__)); 
# 285
extern wchar_t *wcstok(wchar_t *__restrict__ __s, const wchar_t *__restrict__ __delim, wchar_t **__restrict__ __ptr) throw(); 
# 290
extern size_t wcslen(const wchar_t * __s) throw() __attribute((__pure__)); 
# 296
extern "C++" wchar_t *wcswcs(wchar_t * __haystack, const wchar_t * __needle) throw() __asm__("wcswcs")
# 297
 __attribute((__pure__)); 
# 298
extern "C++" const wchar_t *wcswcs(const wchar_t * __haystack, const wchar_t * __needle) throw() __asm__("wcswcs")
# 300
 __attribute((__pure__)); 
# 309
extern size_t wcsnlen(const wchar_t * __s, size_t __maxlen) throw()
# 310
 __attribute((__pure__)); 
# 317
extern "C++" wchar_t *wmemchr(wchar_t * __s, wchar_t __c, size_t __n) throw() __asm__("wmemchr")
# 318
 __attribute((__pure__)); 
# 319
extern "C++" const wchar_t *wmemchr(const wchar_t * __s, wchar_t __c, size_t __n) throw() __asm__("wmemchr")
# 321
 __attribute((__pure__)); 
# 328
extern int wmemcmp(const wchar_t * __s1, const wchar_t * __s2, size_t __n) throw()
# 329
 __attribute((__pure__)); 
# 332
extern wchar_t *wmemcpy(wchar_t *__restrict__ __s1, const wchar_t *__restrict__ __s2, size_t __n) throw(); 
# 337
extern wchar_t *wmemmove(wchar_t * __s1, const wchar_t * __s2, size_t __n) throw(); 
# 341
extern wchar_t *wmemset(wchar_t * __s, wchar_t __c, size_t __n) throw(); 
# 347
extern wchar_t *wmempcpy(wchar_t *__restrict__ __s1, const wchar_t *__restrict__ __s2, size_t __n) throw(); 
# 356
extern wint_t btowc(int __c) throw(); 
# 360
extern int wctob(wint_t __c) throw(); 
# 364
extern int mbsinit(const mbstate_t * __ps) throw() __attribute((__pure__)); 
# 368
extern size_t mbrtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n, mbstate_t *__restrict__ __p) throw(); 
# 373
extern size_t wcrtomb(char *__restrict__ __s, wchar_t __wc, mbstate_t *__restrict__ __ps) throw(); 
# 377
extern size_t __mbrlen(const char *__restrict__ __s, size_t __n, mbstate_t *__restrict__ __ps) throw(); 
# 379
extern size_t mbrlen(const char *__restrict__ __s, size_t __n, mbstate_t *__restrict__ __ps) throw(); 
# 411
extern size_t mbsrtowcs(wchar_t *__restrict__ __dst, const char **__restrict__ __src, size_t __len, mbstate_t *__restrict__ __ps) throw(); 
# 417
extern size_t wcsrtombs(char *__restrict__ __dst, const wchar_t **__restrict__ __src, size_t __len, mbstate_t *__restrict__ __ps) throw(); 
# 426
extern size_t mbsnrtowcs(wchar_t *__restrict__ __dst, const char **__restrict__ __src, size_t __nmc, size_t __len, mbstate_t *__restrict__ __ps) throw(); 
# 432
extern size_t wcsnrtombs(char *__restrict__ __dst, const wchar_t **__restrict__ __src, size_t __nwc, size_t __len, mbstate_t *__restrict__ __ps) throw(); 
# 442
extern int wcwidth(wchar_t __c) throw(); 
# 446
extern int wcswidth(const wchar_t * __s, size_t __n) throw(); 
# 453
extern double wcstod(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr) throw(); 
# 460
extern float wcstof(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr) throw(); 
# 462
extern long double wcstold(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr) throw(); 
# 471
extern long wcstol(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base) throw(); 
# 476
extern unsigned long wcstoul(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base) throw(); 
# 486
__extension__ extern long long wcstoll(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base) throw(); 
# 493
__extension__ extern unsigned long long wcstoull(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base) throw(); 
# 503
__extension__ extern long long wcstoq(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base) throw(); 
# 510
__extension__ extern unsigned long long wcstouq(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base) throw(); 
# 533
extern long wcstol_l(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base, __locale_t __loc) throw(); 
# 537
extern unsigned long wcstoul_l(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base, __locale_t __loc) throw(); 
# 542
__extension__ extern long long wcstoll_l(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base, __locale_t __loc) throw(); 
# 547
__extension__ extern unsigned long long wcstoull_l(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, int __base, __locale_t __loc) throw(); 
# 552
extern double wcstod_l(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, __locale_t __loc) throw(); 
# 556
extern float wcstof_l(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, __locale_t __loc) throw(); 
# 560
extern long double wcstold_l(const wchar_t *__restrict__ __nptr, wchar_t **__restrict__ __endptr, __locale_t __loc) throw(); 
# 569
extern wchar_t *wcpcpy(wchar_t *__restrict__ __dest, const wchar_t *__restrict__ __src) throw(); 
# 574
extern wchar_t *wcpncpy(wchar_t *__restrict__ __dest, const wchar_t *__restrict__ __src, size_t __n) throw(); 
# 583
extern __FILE *open_wmemstream(wchar_t ** __bufloc, size_t * __sizeloc) throw(); 
# 590
extern int fwide(__FILE * __fp, int __mode) throw(); 
# 597
extern int fwprintf(__FILE *__restrict__ __stream, const wchar_t *__restrict__ __format, ...); 
# 604
extern int wprintf(const wchar_t *__restrict__ __format, ...); 
# 607
extern int swprintf(wchar_t *__restrict__ __s, size_t __n, const wchar_t *__restrict__ __format, ...) throw(); 
# 615
extern int vfwprintf(__FILE *__restrict__ __s, const wchar_t *__restrict__ __format, __gnuc_va_list __arg); 
# 623
extern int vwprintf(const wchar_t *__restrict__ __format, __gnuc_va_list __arg); 
# 628
extern int vswprintf(wchar_t *__restrict__ __s, size_t __n, const wchar_t *__restrict__ __format, __gnuc_va_list __arg) throw(); 
# 638
extern int fwscanf(__FILE *__restrict__ __stream, const wchar_t *__restrict__ __format, ...); 
# 645
extern int wscanf(const wchar_t *__restrict__ __format, ...); 
# 648
extern int swscanf(const wchar_t *__restrict__ __s, const wchar_t *__restrict__ __format, ...) throw(); 
# 692
extern int vfwscanf(__FILE *__restrict__ __s, const wchar_t *__restrict__ __format, __gnuc_va_list __arg); 
# 700
extern int vwscanf(const wchar_t *__restrict__ __format, __gnuc_va_list __arg); 
# 704
extern int vswscanf(const wchar_t *__restrict__ __s, const wchar_t *__restrict__ __format, __gnuc_va_list __arg) throw(); 
# 748
extern wint_t fgetwc(__FILE * __stream); 
# 749
extern wint_t getwc(__FILE * __stream); 
# 755
extern wint_t getwchar(); 
# 762
extern wint_t fputwc(wchar_t __wc, __FILE * __stream); 
# 763
extern wint_t putwc(wchar_t __wc, __FILE * __stream); 
# 769
extern wint_t putwchar(wchar_t __wc); 
# 777
extern wchar_t *fgetws(wchar_t *__restrict__ __ws, int __n, __FILE *__restrict__ __stream); 
# 784
extern int fputws(const wchar_t *__restrict__ __ws, __FILE *__restrict__ __stream); 
# 792
extern wint_t ungetwc(wint_t __wc, __FILE * __stream); 
# 804
extern wint_t getwc_unlocked(__FILE * __stream); 
# 805
extern wint_t getwchar_unlocked(); 
# 813
extern wint_t fgetwc_unlocked(__FILE * __stream); 
# 821
extern wint_t fputwc_unlocked(wchar_t __wc, __FILE * __stream); 
# 830
extern wint_t putwc_unlocked(wchar_t __wc, __FILE * __stream); 
# 831
extern wint_t putwchar_unlocked(wchar_t __wc); 
# 840
extern wchar_t *fgetws_unlocked(wchar_t *__restrict__ __ws, int __n, __FILE *__restrict__ __stream); 
# 849
extern int fputws_unlocked(const wchar_t *__restrict__ __ws, __FILE *__restrict__ __stream); 
# 858
extern size_t wcsftime(wchar_t *__restrict__ __s, size_t __maxsize, const wchar_t *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 868
extern size_t wcsftime_l(wchar_t *__restrict__ __s, size_t __maxsize, const wchar_t *__restrict__ __format, const tm *__restrict__ __tp, __locale_t __loc) throw(); 
# 894
}
# 62 "/usr/include/c++/4.8/cwchar" 3
namespace std { 
# 64
using ::mbstate_t;
# 65
}
# 135
namespace std __attribute((__visibility__("default"))) { 
# 139
using ::wint_t;
# 141
using ::btowc;
# 142
using ::fgetwc;
# 143
using ::fgetws;
# 144
using ::fputwc;
# 145
using ::fputws;
# 146
using ::fwide;
# 147
using ::fwprintf;
# 148
using ::fwscanf;
# 149
using ::getwc;
# 150
using ::getwchar;
# 151
using ::mbrlen;
# 152
using ::mbrtowc;
# 153
using ::mbsinit;
# 154
using ::mbsrtowcs;
# 155
using ::putwc;
# 156
using ::putwchar;
# 158
using ::swprintf;
# 160
using ::swscanf;
# 161
using ::ungetwc;
# 162
using ::vfwprintf;
# 164
using ::vfwscanf;
# 167
using ::vswprintf;
# 170
using ::vswscanf;
# 172
using ::vwprintf;
# 174
using ::vwscanf;
# 176
using ::wcrtomb;
# 177
using ::wcscat;
# 178
using ::wcscmp;
# 179
using ::wcscoll;
# 180
using ::wcscpy;
# 181
using ::wcscspn;
# 182
using ::wcsftime;
# 183
using ::wcslen;
# 184
using ::wcsncat;
# 185
using ::wcsncmp;
# 186
using ::wcsncpy;
# 187
using ::wcsrtombs;
# 188
using ::wcsspn;
# 189
using ::wcstod;
# 191
using ::wcstof;
# 193
using ::wcstok;
# 194
using ::wcstol;
# 195
using ::wcstoul;
# 196
using ::wcsxfrm;
# 197
using ::wctob;
# 198
using ::wmemcmp;
# 199
using ::wmemcpy;
# 200
using ::wmemmove;
# 201
using ::wmemset;
# 202
using ::wprintf;
# 203
using ::wscanf;
# 204
using ::wcschr;
# 205
using ::wcspbrk;
# 206
using ::wcsrchr;
# 207
using ::wcsstr;
# 208
using ::wmemchr;
# 233
}
# 241
namespace __gnu_cxx { 
# 248
using ::wcstold;
# 257
using ::wcstoll;
# 258
using ::wcstoull;
# 260
}
# 262
namespace std { 
# 264
using __gnu_cxx::wcstold;
# 265
using __gnu_cxx::wcstoll;
# 266
using __gnu_cxx::wcstoull;
# 267
}
# 277
namespace std { 
# 297
}
# 68 "/usr/include/c++/4.8/bits/postypes.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 88
typedef long streamoff; 
# 98
typedef ptrdiff_t streamsize; 
# 111
template< class _StateT> 
# 112
class fpos { 
# 115
streamoff _M_off; 
# 116
_StateT _M_state; 
# 123
public: fpos() : _M_off((0)), _M_state() 
# 124
{ } 
# 133
fpos(streamoff __off) : _M_off(__off), _M_state() 
# 134
{ } 
# 137
operator streamoff() const { return _M_off; } 
# 141
void state(_StateT __st) 
# 142
{ (_M_state) = __st; } 
# 146
_StateT state() const 
# 147
{ return _M_state; } 
# 154
fpos &operator+=(streamoff __off) 
# 155
{ 
# 156
(_M_off) += __off; 
# 157
return *this; 
# 158
} 
# 165
fpos &operator-=(streamoff __off) 
# 166
{ 
# 167
(_M_off) -= __off; 
# 168
return *this; 
# 169
} 
# 178
fpos operator+(streamoff __off) const 
# 179
{ 
# 180
fpos __pos(*this); 
# 181
__pos += __off; 
# 182
return __pos; 
# 183
} 
# 192
fpos operator-(streamoff __off) const 
# 193
{ 
# 194
fpos __pos(*this); 
# 195
__pos -= __off; 
# 196
return __pos; 
# 197
} 
# 205
streamoff operator-(const fpos &__other) const 
# 206
{ return (_M_off) - (__other._M_off); } 
# 207
}; 
# 214
template< class _StateT> inline bool 
# 216
operator==(const fpos< _StateT>  &__lhs, const fpos< _StateT>  &__rhs) 
# 217
{ return ((streamoff)__lhs) == ((streamoff)__rhs); } 
# 219
template< class _StateT> inline bool 
# 221
operator!=(const fpos< _StateT>  &__lhs, const fpos< _StateT>  &__rhs) 
# 222
{ return ((streamoff)__lhs) != ((streamoff)__rhs); } 
# 228
typedef fpos< __mbstate_t>  streampos; 
# 230
typedef fpos< __mbstate_t>  wstreampos; 
# 234
typedef fpos< __mbstate_t>  u16streampos; 
# 236
typedef fpos< __mbstate_t>  u32streampos; 
# 240
}
# 42 "/usr/include/c++/4.8/iosfwd" 3
namespace std __attribute((__visibility__("default"))) { 
# 74
class ios_base; 
# 76
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_ios; 
# 79
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_streambuf; 
# 82
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_istream; 
# 85
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_ostream; 
# 88
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_iostream; 
# 91
template< class _CharT, class _Traits = char_traits< _CharT> , class 
# 92
_Alloc = allocator< _CharT> > class basic_stringbuf; 
# 95
template< class _CharT, class _Traits = char_traits< _CharT> , class 
# 96
_Alloc = allocator< _CharT> > class basic_istringstream; 
# 99
template< class _CharT, class _Traits = char_traits< _CharT> , class 
# 100
_Alloc = allocator< _CharT> > class basic_ostringstream; 
# 103
template< class _CharT, class _Traits = char_traits< _CharT> , class 
# 104
_Alloc = allocator< _CharT> > class basic_stringstream; 
# 107
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_filebuf; 
# 110
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_ifstream; 
# 113
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_ofstream; 
# 116
template< class _CharT, class _Traits = char_traits< _CharT> > class basic_fstream; 
# 119
template< class _CharT, class _Traits = char_traits< _CharT> > class istreambuf_iterator; 
# 122
template< class _CharT, class _Traits = char_traits< _CharT> > class ostreambuf_iterator; 
# 127
typedef basic_ios< char>  ios; 
# 130
typedef basic_streambuf< char>  streambuf; 
# 133
typedef basic_istream< char>  istream; 
# 136
typedef basic_ostream< char>  ostream; 
# 139
typedef basic_iostream< char>  iostream; 
# 142
typedef basic_stringbuf< char>  stringbuf; 
# 145
typedef basic_istringstream< char>  istringstream; 
# 148
typedef basic_ostringstream< char>  ostringstream; 
# 151
typedef basic_stringstream< char>  stringstream; 
# 154
typedef basic_filebuf< char>  filebuf; 
# 157
typedef basic_ifstream< char>  ifstream; 
# 160
typedef basic_ofstream< char>  ofstream; 
# 163
typedef basic_fstream< char>  fstream; 
# 167
typedef basic_ios< wchar_t>  wios; 
# 170
typedef basic_streambuf< wchar_t>  wstreambuf; 
# 173
typedef basic_istream< wchar_t>  wistream; 
# 176
typedef basic_ostream< wchar_t>  wostream; 
# 179
typedef basic_iostream< wchar_t>  wiostream; 
# 182
typedef basic_stringbuf< wchar_t>  wstringbuf; 
# 185
typedef basic_istringstream< wchar_t>  wistringstream; 
# 188
typedef basic_ostringstream< wchar_t>  wostringstream; 
# 191
typedef basic_stringstream< wchar_t>  wstringstream; 
# 194
typedef basic_filebuf< wchar_t>  wfilebuf; 
# 197
typedef basic_ifstream< wchar_t>  wifstream; 
# 200
typedef basic_ofstream< wchar_t>  wofstream; 
# 203
typedef basic_fstream< wchar_t>  wfstream; 
# 208
}
# 35 "/usr/include/c++/4.8/exception" 3
#pragma GCC visibility push ( default )
# 40
extern "C++" {
# 42
namespace std { 
# 60
class exception { 
# 63
public: exception() noexcept { } 
# 64
virtual ~exception() noexcept; 
# 68
virtual const char *what() const noexcept; 
# 69
}; 
# 73
class bad_exception : public exception { 
# 76
public: bad_exception() noexcept { } 
# 80
virtual ~bad_exception() noexcept; 
# 83
virtual const char *what() const noexcept; 
# 84
}; 
# 87
typedef void (*terminate_handler)(void); 
# 90
typedef void (*unexpected_handler)(void); 
# 93
terminate_handler set_terminate(terminate_handler) noexcept; 
# 97
void terminate() noexcept __attribute((__noreturn__)); 
# 100
unexpected_handler set_unexpected(unexpected_handler) noexcept; 
# 104
void unexpected() __attribute((__noreturn__)); 
# 117
bool uncaught_exception() noexcept __attribute((__pure__)); 
# 120
}
# 122
namespace __gnu_cxx { 
# 142
void __verbose_terminate_handler(); 
# 145
}
# 147
}
# 149
#pragma GCC visibility pop
# 34 "/usr/include/c++/4.8/bits/exception_ptr.h" 3
#pragma GCC visibility push ( default )
# 43
extern "C++" {
# 45
namespace std { 
# 47
class type_info; 
# 53
namespace __exception_ptr { 
# 55
class exception_ptr; 
# 56
}
# 58
using __exception_ptr::exception_ptr;
# 64
__exception_ptr::exception_ptr current_exception() noexcept; 
# 67
void rethrow_exception(__exception_ptr::exception_ptr) __attribute((__noreturn__)); 
# 69
namespace __exception_ptr { 
# 75
class exception_ptr { 
# 77
void *_M_exception_object; 
# 79
explicit exception_ptr(void * __e) noexcept; 
# 81
void _M_addref() noexcept; 
# 82
void _M_release() noexcept; 
# 84
void *_M_get() const noexcept __attribute((__pure__)); 
# 86
friend exception_ptr std::current_exception() noexcept; 
# 87
friend void std::rethrow_exception(exception_ptr); 
# 90
public: exception_ptr() noexcept; 
# 92
exception_ptr(const exception_ptr &) noexcept; 
# 95
exception_ptr(nullptr_t) noexcept : _M_exception_object((0)) 
# 97
{ } 
# 99
exception_ptr(exception_ptr &&__o) noexcept : _M_exception_object(__o._M_exception_object) 
# 101
{ (__o._M_exception_object) = (0); } 
# 112
exception_ptr &operator=(const exception_ptr &) noexcept; 
# 116
exception_ptr &operator=(exception_ptr &&__o) noexcept 
# 117
{ 
# 118
((exception_ptr)(static_cast< exception_ptr &&>(__o))).swap(*this); 
# 119
return *this; 
# 120
} 
# 123
~exception_ptr() noexcept; 
# 126
void swap(exception_ptr &) noexcept; 
# 138
explicit operator bool() const 
# 139
{ return _M_exception_object; } 
# 143
friend bool operator==(const exception_ptr &, const exception_ptr &) noexcept
# 144
 __attribute((__pure__)); 
# 147
const type_info *__cxa_exception_type() const noexcept
# 148
 __attribute((__pure__)); 
# 149
}; 
# 152
bool operator==(const exception_ptr &, const exception_ptr &) noexcept
# 153
 __attribute((__pure__)); 
# 156
bool operator!=(const exception_ptr &, const exception_ptr &) noexcept
# 157
 __attribute((__pure__)); 
# 160
inline void swap(exception_ptr &__lhs, exception_ptr &__rhs) 
# 161
{ __lhs.swap(__rhs); } 
# 163
}
# 167
template< class _Ex> __exception_ptr::exception_ptr 
# 169
copy_exception(_Ex __ex) noexcept 
# 170
{ 
# 171
try 
# 172
{ 
# 174
throw __ex; 
# 176
} 
# 177
catch (...) 
# 178
{ 
# 179
return current_exception(); 
# 180
}  
# 181
} 
# 186
template< class _Ex> __exception_ptr::exception_ptr 
# 188
make_exception_ptr(_Ex __ex) noexcept 
# 189
{ return std::copy_exception< _Ex> (__ex); } 
# 192
}
# 194
}
# 196
#pragma GCC visibility pop
# 33 "/usr/include/c++/4.8/bits/nested_exception.h" 3
#pragma GCC visibility push ( default )
# 45
extern "C++" {
# 47
namespace std { 
# 55
class nested_exception { 
# 57
__exception_ptr::exception_ptr _M_ptr; 
# 60
public: nested_exception() noexcept : _M_ptr(current_exception()) { } 
# 62
nested_exception(const nested_exception &) = default;
# 64
nested_exception &operator=(const nested_exception &) = default;
# 66
virtual ~nested_exception() noexcept; 
# 69
void rethrow_nested() const 
# 70
{ rethrow_exception(_M_ptr); } 
# 73
__exception_ptr::exception_ptr nested_ptr() const 
# 74
{ return _M_ptr; } 
# 75
}; 
# 77
template< class _Except> 
# 78
struct _Nested_exception : public _Except, public nested_exception { 
# 80
explicit _Nested_exception(_Except &&__ex) : _Except(static_cast< _Except &&>(__ex)) 
# 82
{ } 
# 83
}; 
# 85
template< class _Ex> 
# 86
struct __get_nested_helper { 
# 89
static const nested_exception *_S_get(const _Ex &__ex) 
# 90
{ return dynamic_cast< const nested_exception *>(&__ex); } 
# 91
}; 
# 93
template< class _Ex> 
# 94
struct __get_nested_helper< _Ex *>  { 
# 97
static const nested_exception *_S_get(const _Ex *__ex) 
# 98
{ return dynamic_cast< const nested_exception *>(__ex); } 
# 99
}; 
# 101
template< class _Ex> inline const nested_exception *
# 103
__get_nested_exception(const _Ex &__ex) 
# 104
{ return __get_nested_helper< _Ex> ::_S_get(__ex); } 
# 106
template< class _Ex> inline void __throw_with_nested(_Ex &&, const nested_exception * = 0)
# 109
 __attribute((__noreturn__)); 
# 111
template< class _Ex> inline void __throw_with_nested(_Ex &&, ...)
# 113
 __attribute((__noreturn__)); 
# 118
template< class _Ex> inline void 
# 120
__throw_with_nested(_Ex &&__ex, const nested_exception *) 
# 121
{ throw __ex; } 
# 123
template< class _Ex> inline void 
# 125
__throw_with_nested(_Ex &&__ex, ...) 
# 126
{ throw ((_Nested_exception< _Ex> )(static_cast< _Ex &&>(__ex))); } 
# 128
template< class _Ex> inline void throw_with_nested(_Ex __ex)
# 130
 __attribute((__noreturn__)); 
# 134
template< class _Ex> inline void 
# 136
throw_with_nested(_Ex __ex) 
# 137
{ 
# 138
if (__get_nested_exception(__ex)) { 
# 139
throw __ex; }  
# 140
__throw_with_nested(static_cast< _Ex &&>(__ex), &__ex); 
# 141
} 
# 144
template< class _Ex> inline void 
# 146
rethrow_if_nested(const _Ex &__ex) 
# 147
{ 
# 148
if (const nested_exception *__nested = __get_nested_exception(__ex)) { 
# 149
__nested->rethrow_nested(); }  
# 150
} 
# 154
inline void rethrow_if_nested(const nested_exception &__ex) 
# 155
{ __ex.rethrow_nested(); } 
# 158
}
# 160
}
# 164
#pragma GCC visibility pop
# 42 "/usr/include/c++/4.8/bits/functexcept.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 48
void __throw_bad_exception() __attribute((__noreturn__)); 
# 52
void __throw_bad_alloc() __attribute((__noreturn__)); 
# 56
void __throw_bad_cast() __attribute((__noreturn__)); 
# 59
void __throw_bad_typeid() __attribute((__noreturn__)); 
# 63
void __throw_logic_error(const char *) __attribute((__noreturn__)); 
# 66
void __throw_domain_error(const char *) __attribute((__noreturn__)); 
# 69
void __throw_invalid_argument(const char *) __attribute((__noreturn__)); 
# 72
void __throw_length_error(const char *) __attribute((__noreturn__)); 
# 75
void __throw_out_of_range(const char *) __attribute((__noreturn__)); 
# 78
void __throw_runtime_error(const char *) __attribute((__noreturn__)); 
# 81
void __throw_range_error(const char *) __attribute((__noreturn__)); 
# 84
void __throw_overflow_error(const char *) __attribute((__noreturn__)); 
# 87
void __throw_underflow_error(const char *) __attribute((__noreturn__)); 
# 91
void __throw_ios_failure(const char *) __attribute((__noreturn__)); 
# 94
void __throw_system_error(int) __attribute((__noreturn__)); 
# 97
void __throw_future_error(int) __attribute((__noreturn__)); 
# 101
void __throw_bad_function_call() __attribute((__noreturn__)); 
# 104
}
# 37 "/usr/include/c++/4.8/ext/numeric_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 54
template< class _Value> 
# 55
struct __numeric_traits_integer { 
# 58
static const _Value __min = ((((_Value)(-1)) < 0) ? ((_Value)1) << ((sizeof(_Value) * (8)) - (((_Value)(-1)) < 0)) : ((_Value)0)); 
# 59
static const _Value __max = ((((_Value)(-1)) < 0) ? (((((_Value)1) << (((sizeof(_Value) * (8)) - (((_Value)(-1)) < 0)) - (1))) - 1) << 1) + 1 : (~((_Value)0))); 
# 63
static const bool __is_signed = (((_Value)(-1)) < 0); 
# 64
static const int __digits = ((sizeof(_Value) * (8)) - (((_Value)(-1)) < 0)); 
# 65
}; 
# 67
template< class _Value> const _Value 
# 68
__numeric_traits_integer< _Value> ::__min; 
# 70
template< class _Value> const _Value 
# 71
__numeric_traits_integer< _Value> ::__max; 
# 73
template< class _Value> const bool 
# 74
__numeric_traits_integer< _Value> ::__is_signed; 
# 76
template< class _Value> const int 
# 77
__numeric_traits_integer< _Value> ::__digits; 
# 99
template< class _Value> 
# 100
struct __numeric_traits_floating { 
# 103
static const int __max_digits10 = ((2) + ((((std::__are_same< _Value, float> ::__value) ? 24 : ((std::__are_same< _Value, double> ::__value) ? 53 : 64)) * 643L) / (2136))); 
# 106
static const bool __is_signed = true; 
# 107
static const int __digits10 = ((std::__are_same< _Value, float> ::__value) ? 6 : ((std::__are_same< _Value, double> ::__value) ? 15 : 18)); 
# 108
static const int __max_exponent10 = ((std::__are_same< _Value, float> ::__value) ? 38 : ((std::__are_same< _Value, double> ::__value) ? 308 : 4932)); 
# 109
}; 
# 111
template< class _Value> const int 
# 112
__numeric_traits_floating< _Value> ::__max_digits10; 
# 114
template< class _Value> const bool 
# 115
__numeric_traits_floating< _Value> ::__is_signed; 
# 117
template< class _Value> const int 
# 118
__numeric_traits_floating< _Value> ::__digits10; 
# 120
template< class _Value> const int 
# 121
__numeric_traits_floating< _Value> ::__max_exponent10; 
# 123
template< class _Value> 
# 124
struct __numeric_traits : public __conditional_type< std::__is_integer< _Value> ::__value, __numeric_traits_integer< _Value> , __numeric_traits_floating< _Value> > ::__type { 
# 128
}; 
# 131
}
# 36 "/usr/include/c++/4.8/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 45
template< class _Tp> inline _Tp *
# 47
__addressof(_Tp &__r) noexcept 
# 48
{ 
# 49
return reinterpret_cast< _Tp *>(&(const_cast< char &>(reinterpret_cast< const volatile char &>(__r)))); 
# 51
} 
# 54
}
# 40 "/usr/include/c++/4.8/type_traits" 3
namespace std __attribute((__visibility__("default"))) { 
# 56
template< class _Tp, _Tp __v> 
# 57
struct integral_constant { 
# 59
static constexpr _Tp value = (__v); 
# 60
typedef _Tp value_type; 
# 61
typedef integral_constant type; 
# 62
constexpr operator value_type() const { return value; } 
# 63
}; 
# 65
template< class _Tp, _Tp __v> constexpr _Tp 
# 66
integral_constant< _Tp, __v> ::value; 
# 69
typedef integral_constant< bool, true>  true_type; 
# 72
typedef integral_constant< bool, false>  false_type; 
# 76
template< bool , class , class > struct conditional; 
# 79
template< class ...> struct __or_; 
# 83
template<> struct __or_< >  : public false_type { 
# 85
}; 
# 87
template< class _B1> 
# 88
struct __or_< _B1>  : public _B1 { 
# 90
}; 
# 92
template< class _B1, class _B2> 
# 93
struct __or_< _B1, _B2>  : public conditional< _B1::value, _B1, _B2> ::type { 
# 95
}; 
# 97
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 98
struct __or_< _B1, _B2, _B3, _Bn...>  : public conditional< _B1::value, _B1, __or_< _B2, _B3, _Bn...> > ::type { 
# 100
}; 
# 102
template< class ...> struct __and_; 
# 106
template<> struct __and_< >  : public true_type { 
# 108
}; 
# 110
template< class _B1> 
# 111
struct __and_< _B1>  : public _B1 { 
# 113
}; 
# 115
template< class _B1, class _B2> 
# 116
struct __and_< _B1, _B2>  : public conditional< _B1::value, _B2, _B1> ::type { 
# 118
}; 
# 120
template< class _B1, class _B2, class _B3, class ..._Bn> 
# 121
struct __and_< _B1, _B2, _B3, _Bn...>  : public conditional< _B1::value, __and_< _B2, _B3, _Bn...> , _B1> ::type { 
# 123
}; 
# 125
template< class _Pp> 
# 126
struct __not_ : public integral_constant< bool, !_Pp::value>  { 
# 128
}; 
# 130
struct __sfinae_types { 
# 132
typedef char __one; 
# 133
typedef struct { char __arr[2]; } __two; 
# 134
}; 
# 141
template< class _Tp> 
# 142
struct __success_type { 
# 143
typedef _Tp type; }; 
# 145
struct __failure_type { 
# 146
}; 
# 150
template< class > struct remove_cv; 
# 153
template< class > 
# 154
struct __is_void_helper : public false_type { 
# 155
}; 
# 158
template<> struct __is_void_helper< void>  : public true_type { 
# 159
}; 
# 162
template< class _Tp> 
# 163
struct is_void : public integral_constant< bool, __is_void_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 166
}; 
# 168
template< class > 
# 169
struct __is_integral_helper : public false_type { 
# 170
}; 
# 173
template<> struct __is_integral_helper< bool>  : public true_type { 
# 174
}; 
# 177
template<> struct __is_integral_helper< char>  : public true_type { 
# 178
}; 
# 181
template<> struct __is_integral_helper< signed char>  : public true_type { 
# 182
}; 
# 185
template<> struct __is_integral_helper< unsigned char>  : public true_type { 
# 186
}; 
# 190
template<> struct __is_integral_helper< wchar_t>  : public true_type { 
# 191
}; 
# 195
template<> struct __is_integral_helper< char16_t>  : public true_type { 
# 196
}; 
# 199
template<> struct __is_integral_helper< char32_t>  : public true_type { 
# 200
}; 
# 203
template<> struct __is_integral_helper< short>  : public true_type { 
# 204
}; 
# 207
template<> struct __is_integral_helper< unsigned short>  : public true_type { 
# 208
}; 
# 211
template<> struct __is_integral_helper< int>  : public true_type { 
# 212
}; 
# 215
template<> struct __is_integral_helper< unsigned>  : public true_type { 
# 216
}; 
# 219
template<> struct __is_integral_helper< long>  : public true_type { 
# 220
}; 
# 223
template<> struct __is_integral_helper< unsigned long>  : public true_type { 
# 224
}; 
# 227
template<> struct __is_integral_helper< long long>  : public true_type { 
# 228
}; 
# 231
template<> struct __is_integral_helper< unsigned long long>  : public true_type { 
# 232
}; 
# 245
template< class _Tp> 
# 246
struct is_integral : public integral_constant< bool, __is_integral_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 249
}; 
# 251
template< class > 
# 252
struct __is_floating_point_helper : public false_type { 
# 253
}; 
# 256
template<> struct __is_floating_point_helper< float>  : public true_type { 
# 257
}; 
# 260
template<> struct __is_floating_point_helper< double>  : public true_type { 
# 261
}; 
# 264
template<> struct __is_floating_point_helper< long double>  : public true_type { 
# 265
}; 
# 274
template< class _Tp> 
# 275
struct is_floating_point : public integral_constant< bool, __is_floating_point_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 278
}; 
# 281
template< class > 
# 282
struct is_array : public false_type { 
# 283
}; 
# 285
template< class _Tp, size_t _Size> 
# 286
struct is_array< _Tp [_Size]>  : public true_type { 
# 287
}; 
# 289
template< class _Tp> 
# 290
struct is_array< _Tp []>  : public true_type { 
# 291
}; 
# 293
template< class > 
# 294
struct __is_pointer_helper : public false_type { 
# 295
}; 
# 297
template< class _Tp> 
# 298
struct __is_pointer_helper< _Tp *>  : public true_type { 
# 299
}; 
# 302
template< class _Tp> 
# 303
struct is_pointer : public integral_constant< bool, __is_pointer_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 306
}; 
# 309
template< class > 
# 310
struct is_lvalue_reference : public false_type { 
# 311
}; 
# 313
template< class _Tp> 
# 314
struct is_lvalue_reference< _Tp &>  : public true_type { 
# 315
}; 
# 318
template< class > 
# 319
struct is_rvalue_reference : public false_type { 
# 320
}; 
# 322
template< class _Tp> 
# 323
struct is_rvalue_reference< _Tp &&>  : public true_type { 
# 324
}; 
# 326
template< class > struct is_function; 
# 329
template< class > 
# 330
struct __is_member_object_pointer_helper : public false_type { 
# 331
}; 
# 333
template< class _Tp, class _Cp> 
# 334
struct __is_member_object_pointer_helper< _Tp (_Cp::*)>  : public integral_constant< bool, !is_function< _Tp> ::value>  { 
# 335
}; 
# 338
template< class _Tp> 
# 339
struct is_member_object_pointer : public integral_constant< bool, __is_member_object_pointer_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 342
}; 
# 344
template< class > 
# 345
struct __is_member_function_pointer_helper : public false_type { 
# 346
}; 
# 348
template< class _Tp, class _Cp> 
# 349
struct __is_member_function_pointer_helper< _Tp (_Cp::*)>  : public integral_constant< bool, is_function< _Tp> ::value>  { 
# 350
}; 
# 353
template< class _Tp> 
# 354
struct is_member_function_pointer : public integral_constant< bool, __is_member_function_pointer_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 357
}; 
# 360
template< class _Tp> 
# 361
struct is_enum : public integral_constant< bool, __is_enum(_Tp)>  { 
# 363
}; 
# 366
template< class _Tp> 
# 367
struct is_union : public integral_constant< bool, __is_union(_Tp)>  { 
# 369
}; 
# 372
template< class _Tp> 
# 373
struct is_class : public integral_constant< bool, __is_class(_Tp)>  { 
# 375
}; 
# 378
template< class > 
# 379
struct is_function : public false_type { 
# 380
}; 
# 382
template< class _Res, class ..._ArgTypes> 
# 383
struct is_function< _Res (_ArgTypes ...)>  : public true_type { 
# 384
}; 
# 386
template< class _Res, class ..._ArgTypes> 
# 387
struct is_function< _Res (_ArgTypes ..., ...)>  : public true_type { 
# 388
}; 
# 390
template< class _Res, class ..._ArgTypes> 
# 391
struct is_function< _Res (_ArgTypes ...) const>  : public true_type { 
# 392
}; 
# 394
template< class _Res, class ..._ArgTypes> 
# 395
struct is_function< _Res (_ArgTypes ..., ...) const>  : public true_type { 
# 396
}; 
# 398
template< class _Res, class ..._ArgTypes> 
# 399
struct is_function< _Res (_ArgTypes ...) volatile>  : public true_type { 
# 400
}; 
# 402
template< class _Res, class ..._ArgTypes> 
# 403
struct is_function< _Res (_ArgTypes ..., ...) volatile>  : public true_type { 
# 404
}; 
# 406
template< class _Res, class ..._ArgTypes> 
# 407
struct is_function< _Res (_ArgTypes ...) const volatile>  : public true_type { 
# 408
}; 
# 410
template< class _Res, class ..._ArgTypes> 
# 411
struct is_function< _Res (_ArgTypes ..., ...) const volatile>  : public true_type { 
# 412
}; 
# 414
template< class > 
# 415
struct __is_nullptr_t_helper : public false_type { 
# 416
}; 
# 419
template<> struct __is_nullptr_t_helper< nullptr_t>  : public true_type { 
# 420
}; 
# 423
template< class _Tp> 
# 424
struct __is_nullptr_t : public integral_constant< bool, __is_nullptr_t_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 427
}; 
# 432
template< class _Tp> 
# 433
struct is_reference : public __or_< is_lvalue_reference< _Tp> , is_rvalue_reference< _Tp> > ::type { 
# 436
}; 
# 439
template< class _Tp> 
# 440
struct is_arithmetic : public __or_< is_integral< _Tp> , is_floating_point< _Tp> > ::type { 
# 442
}; 
# 445
template< class _Tp> 
# 446
struct is_fundamental : public __or_< is_arithmetic< _Tp> , is_void< _Tp> , __is_nullptr_t< _Tp> > ::type { 
# 448
}; 
# 451
template< class _Tp> 
# 452
struct is_object : public __not_< __or_< is_function< _Tp> , is_reference< _Tp> , is_void< _Tp> > > ::type { 
# 455
}; 
# 457
template< class > struct is_member_pointer; 
# 461
template< class _Tp> 
# 462
struct is_scalar : public __or_< is_arithmetic< _Tp> , is_enum< _Tp> , is_pointer< _Tp> , is_member_pointer< _Tp> , __is_nullptr_t< _Tp> > ::type { 
# 465
}; 
# 468
template< class _Tp> 
# 469
struct is_compound : public integral_constant< bool, !is_fundamental< _Tp> ::value>  { 
# 470
}; 
# 472
template< class _Tp> 
# 473
struct __is_member_pointer_helper : public false_type { 
# 474
}; 
# 476
template< class _Tp, class _Cp> 
# 477
struct __is_member_pointer_helper< _Tp (_Cp::*)>  : public true_type { 
# 478
}; 
# 481
template< class _Tp> 
# 482
struct is_member_pointer : public integral_constant< bool, __is_member_pointer_helper< typename remove_cv< _Tp> ::type> ::value>  { 
# 485
}; 
# 490
template< class > 
# 491
struct is_const : public false_type { 
# 492
}; 
# 494
template< class _Tp> 
# 495
struct is_const< const _Tp>  : public true_type { 
# 496
}; 
# 499
template< class > 
# 500
struct is_volatile : public false_type { 
# 501
}; 
# 503
template< class _Tp> 
# 504
struct is_volatile< volatile _Tp>  : public true_type { 
# 505
}; 
# 508
template< class _Tp> 
# 509
struct is_trivial : public integral_constant< bool, __is_trivial(_Tp)>  { 
# 511
}; 
# 516
template< class _Tp> 
# 517
struct is_standard_layout : public integral_constant< bool, __is_standard_layout(_Tp)>  { 
# 519
}; 
# 523
template< class _Tp> 
# 524
struct is_pod : public integral_constant< bool, __is_pod(_Tp)>  { 
# 526
}; 
# 529
template< class _Tp> 
# 530
struct is_literal_type : public integral_constant< bool, __is_literal_type(_Tp)>  { 
# 532
}; 
# 535
template< class _Tp> 
# 536
struct is_empty : public integral_constant< bool, __is_empty(_Tp)>  { 
# 538
}; 
# 541
template< class _Tp> 
# 542
struct is_polymorphic : public integral_constant< bool, __is_polymorphic(_Tp)>  { 
# 544
}; 
# 547
template< class _Tp> 
# 548
struct is_abstract : public integral_constant< bool, __is_abstract(_Tp)>  { 
# 550
}; 
# 552
template< class _Tp, bool 
# 553
 = is_integral< _Tp> ::value, bool 
# 554
 = is_floating_point< _Tp> ::value> 
# 555
struct __is_signed_helper : public false_type { 
# 556
}; 
# 558
template< class _Tp> 
# 559
struct __is_signed_helper< _Tp, false, true>  : public true_type { 
# 560
}; 
# 562
template< class _Tp> 
# 563
struct __is_signed_helper< _Tp, true, false>  : public integral_constant< bool, (bool)((((_Tp)(-1)) < ((_Tp)0)))>  { 
# 565
}; 
# 568
template< class _Tp> 
# 569
struct is_signed : public integral_constant< bool, __is_signed_helper< _Tp> ::value>  { 
# 571
}; 
# 574
template< class _Tp> 
# 575
struct is_unsigned : public __and_< is_arithmetic< _Tp> , __not_< is_signed< _Tp> > > ::type { 
# 577
}; 
# 582
template< class > struct add_rvalue_reference; 
# 589
template< class _Tp> inline typename add_rvalue_reference< _Tp> ::type declval() noexcept; 
# 592
template< class , unsigned  = 0U> struct extent; 
# 595
template< class > struct remove_all_extents; 
# 598
template< class _Tp> 
# 599
struct __is_array_known_bounds : public integral_constant< bool, (extent< _Tp> ::value > 0)>  { 
# 601
}; 
# 603
template< class _Tp> 
# 604
struct __is_array_unknown_bounds : public __and_< is_array< _Tp> , __not_< extent< _Tp> > > ::type { 
# 606
}; 
# 613
struct __do_is_destructible_impl { 
# 615
template< class _Tp, class  = __decltype(((declval< _Tp &> ().~_Tp())))> static true_type __test(int); 
# 618
template< class > static false_type __test(...); 
# 620
}; 
# 622
template< class _Tp> 
# 623
struct __is_destructible_impl : public __do_is_destructible_impl { 
# 626
typedef __decltype((__test< _Tp> (0))) type; 
# 627
}; 
# 629
template< class _Tp, bool 
# 630
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 633
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_destructible_safe; 
# 636
template< class _Tp> 
# 637
struct __is_destructible_safe< _Tp, false, false>  : public __is_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 640
}; 
# 642
template< class _Tp> 
# 643
struct __is_destructible_safe< _Tp, true, false>  : public false_type { 
# 644
}; 
# 646
template< class _Tp> 
# 647
struct __is_destructible_safe< _Tp, false, true>  : public true_type { 
# 648
}; 
# 651
template< class _Tp> 
# 652
struct is_destructible : public integral_constant< bool, __is_destructible_safe< _Tp> ::value>  { 
# 654
}; 
# 660
struct __do_is_nt_destructible_impl { 
# 662
template< class _Tp> static integral_constant< bool, noexcept((declval< _Tp &> ().~_Tp()))>  __test(int); 
# 666
template< class > static false_type __test(...); 
# 668
}; 
# 670
template< class _Tp> 
# 671
struct __is_nt_destructible_impl : public __do_is_nt_destructible_impl { 
# 674
typedef __decltype((__test< _Tp> (0))) type; 
# 675
}; 
# 677
template< class _Tp, bool 
# 678
 = __or_< is_void< _Tp> , __is_array_unknown_bounds< _Tp> , is_function< _Tp> > ::value, bool 
# 681
 = __or_< is_reference< _Tp> , is_scalar< _Tp> > ::value> struct __is_nt_destructible_safe; 
# 684
template< class _Tp> 
# 685
struct __is_nt_destructible_safe< _Tp, false, false>  : public __is_nt_destructible_impl< typename remove_all_extents< _Tp> ::type> ::type { 
# 688
}; 
# 690
template< class _Tp> 
# 691
struct __is_nt_destructible_safe< _Tp, true, false>  : public false_type { 
# 692
}; 
# 694
template< class _Tp> 
# 695
struct __is_nt_destructible_safe< _Tp, false, true>  : public true_type { 
# 696
}; 
# 699
template< class _Tp> 
# 700
struct is_nothrow_destructible : public integral_constant< bool, __is_nt_destructible_safe< _Tp> ::value>  { 
# 702
}; 
# 704
struct __do_is_default_constructible_impl { 
# 706
template< class _Tp, class  = __decltype((_Tp()))> static true_type __test(int); 
# 709
template< class > static false_type __test(...); 
# 711
}; 
# 713
template< class _Tp> 
# 714
struct __is_default_constructible_impl : public __do_is_default_constructible_impl { 
# 717
typedef __decltype((__test< _Tp> (0))) type; 
# 718
}; 
# 720
template< class _Tp> 
# 721
struct __is_default_constructible_atom : public __and_< __not_< is_void< _Tp> > , __is_default_constructible_impl< _Tp> > ::type { 
# 724
}; 
# 726
template< class _Tp, bool  = is_array< _Tp> ::value> struct __is_default_constructible_safe; 
# 734
template< class _Tp> 
# 735
struct __is_default_constructible_safe< _Tp, true>  : public __and_< __is_array_known_bounds< _Tp> , __is_default_constructible_atom< typename remove_all_extents< _Tp> ::type> > ::type { 
# 739
}; 
# 741
template< class _Tp> 
# 742
struct __is_default_constructible_safe< _Tp, false>  : public __is_default_constructible_atom< _Tp> ::type { 
# 744
}; 
# 747
template< class _Tp> 
# 748
struct is_default_constructible : public integral_constant< bool, __is_default_constructible_safe< _Tp> ::value>  { 
# 751
}; 
# 765
struct __do_is_static_castable_impl { 
# 767
template< class _From, class _To, class 
# 768
 = __decltype((static_cast< _To>(declval< _From> ())))> static true_type 
# 767
__test(int); 
# 771
template< class , class > static false_type __test(...); 
# 773
}; 
# 775
template< class _From, class _To> 
# 776
struct __is_static_castable_impl : public __do_is_static_castable_impl { 
# 779
typedef __decltype((__test< _From, _To> (0))) type; 
# 780
}; 
# 782
template< class _From, class _To> 
# 783
struct __is_static_castable_safe : public __is_static_castable_impl< _From, _To> ::type { 
# 785
}; 
# 788
template< class _From, class _To> 
# 789
struct __is_static_castable : public integral_constant< bool, __is_static_castable_safe< _From, _To> ::value>  { 
# 792
}; 
# 799
struct __do_is_direct_constructible_impl { 
# 801
template< class _Tp, class _Arg, class 
# 802
 = __decltype((::new (_Tp)(declval< _Arg> ())))> static true_type 
# 801
__test(int); 
# 805
template< class , class > static false_type __test(...); 
# 807
}; 
# 809
template< class _Tp, class _Arg> 
# 810
struct __is_direct_constructible_impl : public __do_is_direct_constructible_impl { 
# 813
typedef __decltype((__test< _Tp, _Arg> (0))) type; 
# 814
}; 
# 816
template< class _Tp, class _Arg> 
# 817
struct __is_direct_constructible_new_safe : public __and_< is_destructible< _Tp> , __is_direct_constructible_impl< _Tp, _Arg> > ::type { 
# 820
}; 
# 822
template< class , class > struct is_same; 
# 825
template< class , class > struct is_base_of; 
# 828
template< class > struct remove_reference; 
# 831
template< class _From, class _To, bool 
# 832
 = __not_< __or_< is_void< _From> , is_function< _From> > > ::value> struct __is_base_to_derived_ref; 
# 838
template< class _From, class _To> 
# 839
struct __is_base_to_derived_ref< _From, _To, true>  { 
# 842
typedef typename remove_cv< typename remove_reference< _From> ::type> ::type __src_t; 
# 844
typedef typename remove_cv< typename remove_reference< _To> ::type> ::type __dst_t; 
# 846
typedef __and_< __not_< is_same< typename remove_cv< typename remove_reference< _From> ::type> ::type, typename remove_cv< typename remove_reference< _To> ::type> ::type> > , is_base_of< typename remove_cv< typename remove_reference< _From> ::type> ::type, typename remove_cv< typename remove_reference< _To> ::type> ::type> >  type; 
# 847
static constexpr bool value = (type::value); 
# 848
}; 
# 850
template< class _From, class _To> 
# 851
struct __is_base_to_derived_ref< _From, _To, false>  : public false_type { 
# 853
}; 
# 855
template< class _From, class _To, bool 
# 856
 = __and_< is_lvalue_reference< _From> , is_rvalue_reference< _To> > ::value> struct __is_lvalue_to_rvalue_ref; 
# 862
template< class _From, class _To> 
# 863
struct __is_lvalue_to_rvalue_ref< _From, _To, true>  { 
# 866
typedef typename remove_cv< typename remove_reference< _From> ::type> ::type __src_t; 
# 868
typedef typename remove_cv< typename remove_reference< _To> ::type> ::type __dst_t; 
# 871
typedef __and_< __not_< is_function< typename remove_cv< typename remove_reference< _From> ::type> ::type> > , __or_< is_same< typename remove_cv< typename remove_reference< _From> ::type> ::type, typename remove_cv< typename remove_reference< _To> ::type> ::type> , is_base_of< typename remove_cv< typename remove_reference< _To> ::type> ::type, typename remove_cv< typename remove_reference< _From> ::type> ::type> > >  type; 
# 872
static constexpr bool value = (type::value); 
# 873
}; 
# 875
template< class _From, class _To> 
# 876
struct __is_lvalue_to_rvalue_ref< _From, _To, false>  : public false_type { 
# 878
}; 
# 886
template< class _Tp, class _Arg> 
# 887
struct __is_direct_constructible_ref_cast : public __and_< __is_static_castable< _Arg, _Tp> , __not_< __or_< __is_base_to_derived_ref< _Arg, _Tp> , __is_lvalue_to_rvalue_ref< _Arg, _Tp> > > > ::type { 
# 892
}; 
# 894
template< class _Tp, class _Arg> 
# 895
struct __is_direct_constructible_new : public conditional< is_reference< _Tp> ::value, __is_direct_constructible_ref_cast< _Tp, _Arg> , __is_direct_constructible_new_safe< _Tp, _Arg> > ::type { 
# 900
}; 
# 902
template< class _Tp, class _Arg> 
# 903
struct __is_direct_constructible : public integral_constant< bool, __is_direct_constructible_new< _Tp, _Arg> ::value>  { 
# 906
}; 
# 913
struct __do_is_nary_constructible_impl { 
# 915
template< class _Tp, class ..._Args, class 
# 916
 = __decltype((_Tp(declval< _Args> ()...)))> static true_type 
# 915
__test(int); 
# 919
template< class , class ...> static false_type __test(...); 
# 921
}; 
# 923
template< class _Tp, class ..._Args> 
# 924
struct __is_nary_constructible_impl : public __do_is_nary_constructible_impl { 
# 927
typedef __decltype((__test< _Tp, _Args...> (0))) type; 
# 928
}; 
# 930
template< class _Tp, class ..._Args> 
# 931
struct __is_nary_constructible : public __is_nary_constructible_impl< _Tp, _Args...> ::type { 
# 934
static_assert((sizeof...(_Args) > (1)), "Only useful for > 1 arguments");
# 936
}; 
# 938
template< class _Tp, class ..._Args> 
# 939
struct __is_constructible_impl : public __is_nary_constructible< _Tp, _Args...>  { 
# 941
}; 
# 943
template< class _Tp, class _Arg> 
# 944
struct __is_constructible_impl< _Tp, _Arg>  : public __is_direct_constructible< _Tp, _Arg>  { 
# 946
}; 
# 948
template< class _Tp> 
# 949
struct __is_constructible_impl< _Tp>  : public is_default_constructible< _Tp>  { 
# 951
}; 
# 954
template< class _Tp, class ..._Args> 
# 955
struct is_constructible : public integral_constant< bool, __is_constructible_impl< _Tp, _Args...> ::value>  { 
# 958
}; 
# 960
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_copy_constructible_impl; 
# 963
template< class _Tp> 
# 964
struct __is_copy_constructible_impl< _Tp, true>  : public false_type { 
# 965
}; 
# 967
template< class _Tp> 
# 968
struct __is_copy_constructible_impl< _Tp, false>  : public is_constructible< _Tp, const _Tp &>  { 
# 970
}; 
# 973
template< class _Tp> 
# 974
struct is_copy_constructible : public __is_copy_constructible_impl< _Tp>  { 
# 976
}; 
# 978
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_move_constructible_impl; 
# 981
template< class _Tp> 
# 982
struct __is_move_constructible_impl< _Tp, true>  : public false_type { 
# 983
}; 
# 985
template< class _Tp> 
# 986
struct __is_move_constructible_impl< _Tp, false>  : public is_constructible< _Tp, _Tp &&>  { 
# 988
}; 
# 991
template< class _Tp> 
# 992
struct is_move_constructible : public __is_move_constructible_impl< _Tp>  { 
# 994
}; 
# 996
template< class _Tp> 
# 997
struct __is_nt_default_constructible_atom : public integral_constant< bool, noexcept((_Tp()))>  { 
# 999
}; 
# 1001
template< class _Tp, bool  = is_array< _Tp> ::value> struct __is_nt_default_constructible_impl; 
# 1004
template< class _Tp> 
# 1005
struct __is_nt_default_constructible_impl< _Tp, true>  : public __and_< __is_array_known_bounds< _Tp> , __is_nt_default_constructible_atom< typename remove_all_extents< _Tp> ::type> > ::type { 
# 1009
}; 
# 1011
template< class _Tp> 
# 1012
struct __is_nt_default_constructible_impl< _Tp, false>  : public __is_nt_default_constructible_atom< _Tp>  { 
# 1014
}; 
# 1017
template< class _Tp> 
# 1018
struct is_nothrow_default_constructible : public __and_< is_default_constructible< _Tp> , __is_nt_default_constructible_impl< _Tp> > ::type { 
# 1021
}; 
# 1023
template< class _Tp, class ..._Args> 
# 1024
struct __is_nt_constructible_impl : public integral_constant< bool, noexcept((_Tp(declval< _Args> ()...)))>  { 
# 1026
}; 
# 1028
template< class _Tp, class _Arg> 
# 1029
struct __is_nt_constructible_impl< _Tp, _Arg>  : public integral_constant< bool, noexcept((static_cast< _Tp>(declval< _Arg> ())))>  { 
# 1032
}; 
# 1034
template< class _Tp> 
# 1035
struct __is_nt_constructible_impl< _Tp>  : public is_nothrow_default_constructible< _Tp>  { 
# 1037
}; 
# 1040
template< class _Tp, class ..._Args> 
# 1041
struct is_nothrow_constructible : public __and_< is_constructible< _Tp, _Args...> , __is_nt_constructible_impl< _Tp, _Args...> > ::type { 
# 1044
}; 
# 1046
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_nothrow_copy_constructible_impl; 
# 1049
template< class _Tp> 
# 1050
struct __is_nothrow_copy_constructible_impl< _Tp, true>  : public false_type { 
# 1051
}; 
# 1053
template< class _Tp> 
# 1054
struct __is_nothrow_copy_constructible_impl< _Tp, false>  : public is_nothrow_constructible< _Tp, const _Tp &>  { 
# 1056
}; 
# 1059
template< class _Tp> 
# 1060
struct is_nothrow_copy_constructible : public __is_nothrow_copy_constructible_impl< _Tp>  { 
# 1062
}; 
# 1064
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_nothrow_move_constructible_impl; 
# 1067
template< class _Tp> 
# 1068
struct __is_nothrow_move_constructible_impl< _Tp, true>  : public false_type { 
# 1069
}; 
# 1071
template< class _Tp> 
# 1072
struct __is_nothrow_move_constructible_impl< _Tp, false>  : public is_nothrow_constructible< _Tp, _Tp &&>  { 
# 1074
}; 
# 1077
template< class _Tp> 
# 1078
struct is_nothrow_move_constructible : public __is_nothrow_move_constructible_impl< _Tp>  { 
# 1080
}; 
# 1082
template< class _Tp, class _Up> 
# 1083
class __is_assignable_helper : public __sfinae_types { 
# 1086
template< class _Tp1, class _Up1> static __decltype(((declval< _Tp1> () = declval< _Up1> ()), (__one()))) __test(int); 
# 1090
template< class , class > static __two __test(...); 
# 1094
public: static constexpr bool value = (sizeof(__test< _Tp, _Up> (0)) == (1)); 
# 1095
}; 
# 1098
template< class _Tp, class _Up> 
# 1099
struct is_assignable : public integral_constant< bool, __is_assignable_helper< _Tp, _Up> ::value>  { 
# 1102
}; 
# 1104
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_copy_assignable_impl; 
# 1107
template< class _Tp> 
# 1108
struct __is_copy_assignable_impl< _Tp, true>  : public false_type { 
# 1109
}; 
# 1111
template< class _Tp> 
# 1112
struct __is_copy_assignable_impl< _Tp, false>  : public is_assignable< _Tp &, const _Tp &>  { 
# 1114
}; 
# 1117
template< class _Tp> 
# 1118
struct is_copy_assignable : public __is_copy_assignable_impl< _Tp>  { 
# 1120
}; 
# 1122
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_move_assignable_impl; 
# 1125
template< class _Tp> 
# 1126
struct __is_move_assignable_impl< _Tp, true>  : public false_type { 
# 1127
}; 
# 1129
template< class _Tp> 
# 1130
struct __is_move_assignable_impl< _Tp, false>  : public is_assignable< _Tp &, _Tp &&>  { 
# 1132
}; 
# 1135
template< class _Tp> 
# 1136
struct is_move_assignable : public __is_move_assignable_impl< _Tp>  { 
# 1138
}; 
# 1140
template< class _Tp, class _Up> 
# 1141
struct __is_nt_assignable_impl : public integral_constant< bool, noexcept((declval< _Tp> () = declval< _Up> ()))>  { 
# 1143
}; 
# 1146
template< class _Tp, class _Up> 
# 1147
struct is_nothrow_assignable : public __and_< is_assignable< _Tp, _Up> , __is_nt_assignable_impl< _Tp, _Up> > ::type { 
# 1150
}; 
# 1152
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_nt_copy_assignable_impl; 
# 1155
template< class _Tp> 
# 1156
struct __is_nt_copy_assignable_impl< _Tp, true>  : public false_type { 
# 1157
}; 
# 1159
template< class _Tp> 
# 1160
struct __is_nt_copy_assignable_impl< _Tp, false>  : public is_nothrow_assignable< _Tp &, const _Tp &>  { 
# 1162
}; 
# 1165
template< class _Tp> 
# 1166
struct is_nothrow_copy_assignable : public __is_nt_copy_assignable_impl< _Tp>  { 
# 1168
}; 
# 1170
template< class _Tp, bool  = is_void< _Tp> ::value> struct __is_nt_move_assignable_impl; 
# 1173
template< class _Tp> 
# 1174
struct __is_nt_move_assignable_impl< _Tp, true>  : public false_type { 
# 1175
}; 
# 1177
template< class _Tp> 
# 1178
struct __is_nt_move_assignable_impl< _Tp, false>  : public is_nothrow_assignable< _Tp &, _Tp &&>  { 
# 1180
}; 
# 1183
template< class _Tp> 
# 1184
struct is_nothrow_move_assignable : public __is_nt_move_assignable_impl< _Tp>  { 
# 1186
}; 
# 1203
template< class _Tp> 
# 1204
struct is_trivially_destructible : public __and_< is_destructible< _Tp> , integral_constant< bool, __has_trivial_destructor(_Tp)> > ::type { 
# 1207
}; 
# 1210
template< class _Tp> 
# 1211
struct has_trivial_default_constructor : public integral_constant< bool, __has_trivial_constructor(_Tp)>  { 
# 1213
}; 
# 1216
template< class _Tp> 
# 1217
struct has_trivial_copy_constructor : public integral_constant< bool, __has_trivial_copy(_Tp)>  { 
# 1219
}; 
# 1222
template< class _Tp> 
# 1223
struct has_trivial_copy_assign : public integral_constant< bool, __has_trivial_assign(_Tp)>  { 
# 1225
}; 
# 1228
template< class _Tp> 
# 1229
struct has_virtual_destructor : public integral_constant< bool, __has_virtual_destructor(_Tp)>  { 
# 1231
}; 
# 1237
template< class _Tp> 
# 1238
struct alignment_of : public integral_constant< unsigned long, __alignof__(_Tp)>  { 
# 1239
}; 
# 1242
template< class > 
# 1243
struct rank : public integral_constant< unsigned long, 0UL>  { 
# 1244
}; 
# 1246
template< class _Tp, size_t _Size> 
# 1247
struct rank< _Tp [_Size]>  : public integral_constant< unsigned long, 1 + rank< _Tp> ::value>  { 
# 1248
}; 
# 1250
template< class _Tp> 
# 1251
struct rank< _Tp []>  : public integral_constant< unsigned long, 1 + rank< _Tp> ::value>  { 
# 1252
}; 
# 1255
template< class , unsigned _Uint> 
# 1256
struct extent : public integral_constant< unsigned long, 0UL>  { 
# 1257
}; 
# 1259
template< class _Tp, unsigned _Uint, size_t _Size> 
# 1260
struct extent< _Tp [_Size], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? _Size : extent< _Tp, _Uint - (1)> ::value>  { 
# 1264
}; 
# 1266
template< class _Tp, unsigned _Uint> 
# 1267
struct extent< _Tp [], _Uint>  : public integral_constant< unsigned long, (_Uint == (0)) ? 0 : extent< _Tp, _Uint - (1)> ::value>  { 
# 1271
}; 
# 1277
template< class , class > 
# 1278
struct is_same : public false_type { 
# 1279
}; 
# 1281
template< class _Tp> 
# 1282
struct is_same< _Tp, _Tp>  : public true_type { 
# 1283
}; 
# 1286
template< class _Base, class _Derived> 
# 1287
struct is_base_of : public integral_constant< bool, __is_base_of(_Base, _Derived)>  { 
# 1289
}; 
# 1291
template< class _From, class _To, bool 
# 1292
 = __or_< is_void< _From> , is_function< _To> , is_array< _To> > ::value> 
# 1294
struct __is_convertible_helper { 
# 1295
static constexpr bool value = (is_void< _To> ::value); }; 
# 1297
template< class _From, class _To> 
# 1298
class __is_convertible_helper< _From, _To, false>  : public __sfinae_types { 
# 1301
template< class _To1> static void __test_aux(_To1); 
# 1304
template< class _From1, class _To1> static __decltype((__test_aux< _To1> (std::declval< _From1> ()), (__one()))) __test(int); 
# 1308
template< class , class > static __two __test(...); 
# 1312
public: static constexpr bool value = (sizeof(__test< _From, _To> (0)) == (1)); 
# 1313
}; 
# 1316
template< class _From, class _To> 
# 1317
struct is_convertible : public integral_constant< bool, __is_convertible_helper< _From, _To> ::value>  { 
# 1320
}; 
# 1326
template< class _Tp> 
# 1327
struct remove_const { 
# 1328
typedef _Tp type; }; 
# 1330
template< class _Tp> 
# 1331
struct remove_const< const _Tp>  { 
# 1332
typedef _Tp type; }; 
# 1335
template< class _Tp> 
# 1336
struct remove_volatile { 
# 1337
typedef _Tp type; }; 
# 1339
template< class _Tp> 
# 1340
struct remove_volatile< volatile _Tp>  { 
# 1341
typedef _Tp type; }; 
# 1344
template< class _Tp> 
# 1345
struct remove_cv { 
# 1348
typedef typename remove_const< typename remove_volatile< _Tp> ::type> ::type type; 
# 1349
}; 
# 1352
template< class _Tp> 
# 1353
struct add_const { 
# 1354
typedef const _Tp type; }; 
# 1357
template< class _Tp> 
# 1358
struct add_volatile { 
# 1359
typedef volatile _Tp type; }; 
# 1362
template< class _Tp> 
# 1363
struct add_cv { 
# 1366
typedef typename add_const< typename add_volatile< _Tp> ::type> ::type type; 
# 1367
}; 
# 1373
template< class _Tp> 
# 1374
struct remove_reference { 
# 1375
typedef _Tp type; }; 
# 1377
template< class _Tp> 
# 1378
struct remove_reference< _Tp &>  { 
# 1379
typedef _Tp type; }; 
# 1381
template< class _Tp> 
# 1382
struct remove_reference< _Tp &&>  { 
# 1383
typedef _Tp type; }; 
# 1385
template< class _Tp, bool 
# 1386
 = __and_< __not_< is_reference< _Tp> > , __not_< is_void< _Tp> > > ::value, bool 
# 1388
 = is_rvalue_reference< _Tp> ::value> 
# 1389
struct __add_lvalue_reference_helper { 
# 1390
typedef _Tp type; }; 
# 1392
template< class _Tp> 
# 1393
struct __add_lvalue_reference_helper< _Tp, true, false>  { 
# 1394
typedef _Tp &type; }; 
# 1396
template< class _Tp> 
# 1397
struct __add_lvalue_reference_helper< _Tp, false, true>  { 
# 1398
typedef typename remove_reference< _Tp> ::type &type; }; 
# 1401
template< class _Tp> 
# 1402
struct add_lvalue_reference : public __add_lvalue_reference_helper< _Tp>  { 
# 1404
}; 
# 1406
template< class _Tp, bool 
# 1407
 = __and_< __not_< is_reference< _Tp> > , __not_< is_void< _Tp> > > ::value> 
# 1409
struct __add_rvalue_reference_helper { 
# 1410
typedef _Tp type; }; 
# 1412
template< class _Tp> 
# 1413
struct __add_rvalue_reference_helper< _Tp, true>  { 
# 1414
typedef _Tp &&type; }; 
# 1417
template< class _Tp> 
# 1418
struct add_rvalue_reference : public __add_rvalue_reference_helper< _Tp>  { 
# 1420
}; 
# 1426
template< class _Unqualified, bool _IsConst, bool _IsVol> struct __cv_selector; 
# 1429
template< class _Unqualified> 
# 1430
struct __cv_selector< _Unqualified, false, false>  { 
# 1431
typedef _Unqualified __type; }; 
# 1433
template< class _Unqualified> 
# 1434
struct __cv_selector< _Unqualified, false, true>  { 
# 1435
typedef volatile _Unqualified __type; }; 
# 1437
template< class _Unqualified> 
# 1438
struct __cv_selector< _Unqualified, true, false>  { 
# 1439
typedef const _Unqualified __type; }; 
# 1441
template< class _Unqualified> 
# 1442
struct __cv_selector< _Unqualified, true, true>  { 
# 1443
typedef const volatile _Unqualified __type; }; 
# 1445
template< class _Qualified, class _Unqualified, bool 
# 1446
_IsConst = is_const< _Qualified> ::value, bool 
# 1447
_IsVol = is_volatile< _Qualified> ::value> 
# 1448
class __match_cv_qualifiers { 
# 1450
typedef __cv_selector< _Unqualified, _IsConst, _IsVol>  __match; 
# 1453
public: typedef typename __cv_selector< _Unqualified, _IsConst, _IsVol> ::__type __type; 
# 1454
}; 
# 1457
template< class _Tp> 
# 1458
struct __make_unsigned { 
# 1459
typedef _Tp __type; }; 
# 1462
template<> struct __make_unsigned< char>  { 
# 1463
typedef unsigned char __type; }; 
# 1466
template<> struct __make_unsigned< signed char>  { 
# 1467
typedef unsigned char __type; }; 
# 1470
template<> struct __make_unsigned< short>  { 
# 1471
typedef unsigned short __type; }; 
# 1474
template<> struct __make_unsigned< int>  { 
# 1475
typedef unsigned __type; }; 
# 1478
template<> struct __make_unsigned< long>  { 
# 1479
typedef unsigned long __type; }; 
# 1482
template<> struct __make_unsigned< long long>  { 
# 1483
typedef unsigned long long __type; }; 
# 1492
template< class _Tp, bool 
# 1493
_IsInt = is_integral< _Tp> ::value, bool 
# 1494
_IsEnum = is_enum< _Tp> ::value> class __make_unsigned_selector; 
# 1497
template< class _Tp> 
# 1498
class __make_unsigned_selector< _Tp, true, false>  { 
# 1500
typedef __make_unsigned< typename remove_cv< _Tp> ::type>  __unsignedt; 
# 1501
typedef typename __make_unsigned< typename remove_cv< _Tp> ::type> ::__type __unsigned_type; 
# 1502
typedef __match_cv_qualifiers< _Tp, typename __make_unsigned< typename remove_cv< _Tp> ::type> ::__type>  __cv_unsigned; 
# 1505
public: typedef typename __match_cv_qualifiers< _Tp, typename __make_unsigned< typename remove_cv< _Tp> ::type> ::__type> ::__type __type; 
# 1506
}; 
# 1508
template< class _Tp> 
# 1509
class __make_unsigned_selector< _Tp, false, true>  { 
# 1512
typedef unsigned char __smallest; 
# 1513
static const bool __b0 = (sizeof(_Tp) <= sizeof(__smallest)); 
# 1514
static const bool __b1 = (sizeof(_Tp) <= sizeof(unsigned short)); 
# 1515
static const bool __b2 = (sizeof(_Tp) <= sizeof(unsigned)); 
# 1516
typedef conditional< __b2, unsigned, unsigned long>  __cond2; 
# 1517
typedef typename conditional< __b2, unsigned, unsigned long> ::type __cond2_type; 
# 1518
typedef conditional< __b1, unsigned short, typename conditional< __b2, unsigned, unsigned long> ::type>  __cond1; 
# 1519
typedef typename conditional< __b1, unsigned short, typename conditional< __b2, unsigned, unsigned long> ::type> ::type __cond1_type; 
# 1522
public: typedef typename conditional< __b0, unsigned char, typename conditional< __b1, unsigned short, typename conditional< __b2, unsigned, unsigned long> ::type> ::type> ::type __type; 
# 1523
}; 
# 1529
template< class _Tp> 
# 1530
struct make_unsigned { 
# 1531
typedef typename __make_unsigned_selector< _Tp> ::__type type; }; 
# 1535
template<> struct make_unsigned< bool> ; 
# 1539
template< class _Tp> 
# 1540
struct __make_signed { 
# 1541
typedef _Tp __type; }; 
# 1544
template<> struct __make_signed< char>  { 
# 1545
typedef signed char __type; }; 
# 1548
template<> struct __make_signed< unsigned char>  { 
# 1549
typedef signed char __type; }; 
# 1552
template<> struct __make_signed< unsigned short>  { 
# 1553
typedef signed short __type; }; 
# 1556
template<> struct __make_signed< unsigned>  { 
# 1557
typedef signed int __type; }; 
# 1560
template<> struct __make_signed< unsigned long>  { 
# 1561
typedef signed long __type; }; 
# 1564
template<> struct __make_signed< unsigned long long>  { 
# 1565
typedef signed long long __type; }; 
# 1574
template< class _Tp, bool 
# 1575
_IsInt = is_integral< _Tp> ::value, bool 
# 1576
_IsEnum = is_enum< _Tp> ::value> class __make_signed_selector; 
# 1579
template< class _Tp> 
# 1580
class __make_signed_selector< _Tp, true, false>  { 
# 1582
typedef __make_signed< typename remove_cv< _Tp> ::type>  __signedt; 
# 1583
typedef typename __make_signed< typename remove_cv< _Tp> ::type> ::__type __signed_type; 
# 1584
typedef __match_cv_qualifiers< _Tp, typename __make_signed< typename remove_cv< _Tp> ::type> ::__type>  __cv_signed; 
# 1587
public: typedef typename __match_cv_qualifiers< _Tp, typename __make_signed< typename remove_cv< _Tp> ::type> ::__type> ::__type __type; 
# 1588
}; 
# 1590
template< class _Tp> 
# 1591
class __make_signed_selector< _Tp, false, true>  { 
# 1594
typedef signed char __smallest; 
# 1595
static const bool __b0 = (sizeof(_Tp) <= sizeof(__smallest)); 
# 1596
static const bool __b1 = (sizeof(_Tp) <= sizeof(signed short)); 
# 1597
static const bool __b2 = (sizeof(_Tp) <= sizeof(signed int)); 
# 1598
typedef conditional< __b2, signed int, signed long>  __cond2; 
# 1599
typedef typename conditional< __b2, signed int, signed long> ::type __cond2_type; 
# 1600
typedef conditional< __b1, signed short, typename conditional< __b2, signed int, signed long> ::type>  __cond1; 
# 1601
typedef typename conditional< __b1, signed short, typename conditional< __b2, signed int, signed long> ::type> ::type __cond1_type; 
# 1604
public: typedef typename conditional< __b0, signed char, typename conditional< __b1, signed short, typename conditional< __b2, signed int, signed long> ::type> ::type> ::type __type; 
# 1605
}; 
# 1611
template< class _Tp> 
# 1612
struct make_signed { 
# 1613
typedef typename __make_signed_selector< _Tp> ::__type type; }; 
# 1617
template<> struct make_signed< bool> ; 
# 1623
template< class _Tp> 
# 1624
struct remove_extent { 
# 1625
typedef _Tp type; }; 
# 1627
template< class _Tp, size_t _Size> 
# 1628
struct remove_extent< _Tp [_Size]>  { 
# 1629
typedef _Tp type; }; 
# 1631
template< class _Tp> 
# 1632
struct remove_extent< _Tp []>  { 
# 1633
typedef _Tp type; }; 
# 1636
template< class _Tp> 
# 1637
struct remove_all_extents { 
# 1638
typedef _Tp type; }; 
# 1640
template< class _Tp, size_t _Size> 
# 1641
struct remove_all_extents< _Tp [_Size]>  { 
# 1642
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 1644
template< class _Tp> 
# 1645
struct remove_all_extents< _Tp []>  { 
# 1646
typedef typename std::remove_all_extents< _Tp> ::type type; }; 
# 1651
template< class _Tp, class > 
# 1652
struct __remove_pointer_helper { 
# 1653
typedef _Tp type; }; 
# 1655
template< class _Tp, class _Up> 
# 1656
struct __remove_pointer_helper< _Tp, _Up *>  { 
# 1657
typedef _Up type; }; 
# 1660
template< class _Tp> 
# 1661
struct remove_pointer : public __remove_pointer_helper< _Tp, typename remove_cv< _Tp> ::type>  { 
# 1663
}; 
# 1666
template< class _Tp> 
# 1667
struct add_pointer { 
# 1668
typedef typename remove_reference< _Tp> ::type *type; }; 
# 1671
template< size_t _Len> 
# 1672
struct __aligned_storage_msa { 
# 1674
union __type { 
# 1676
unsigned char __data[_Len]; 
# 1677
struct __attribute((__aligned__)) { } __align; 
# 1678
}; 
# 1679
}; 
# 1691
template< size_t _Len, size_t _Align = __alignof__(typename __aligned_storage_msa< _Len> ::__type)> 
# 1693
struct aligned_storage { 
# 1695
union type { 
# 1697
unsigned char __data[_Len]; 
# 1698
struct __attribute((__aligned__(_Align))) { } __align; 
# 1699
}; 
# 1700
}; 
# 1705
template< class _Up, bool 
# 1706
_IsArray = is_array< _Up> ::value, bool 
# 1707
_IsFunction = is_function< _Up> ::value> struct __decay_selector; 
# 1711
template< class _Up> 
# 1712
struct __decay_selector< _Up, false, false>  { 
# 1713
typedef typename remove_cv< _Up> ::type __type; }; 
# 1715
template< class _Up> 
# 1716
struct __decay_selector< _Up, true, false>  { 
# 1717
typedef typename remove_extent< _Up> ::type *__type; }; 
# 1719
template< class _Up> 
# 1720
struct __decay_selector< _Up, false, true>  { 
# 1721
typedef typename add_pointer< _Up> ::type __type; }; 
# 1724
template< class _Tp> 
# 1725
class decay { 
# 1727
typedef typename remove_reference< _Tp> ::type __remove_type; 
# 1730
public: typedef typename __decay_selector< typename remove_reference< _Tp> ::type> ::__type type; 
# 1731
}; 
# 1733
template< class _Tp> class reference_wrapper; 
# 1737
template< class _Tp> 
# 1738
struct __strip_reference_wrapper { 
# 1740
typedef _Tp __type; 
# 1741
}; 
# 1743
template< class _Tp> 
# 1744
struct __strip_reference_wrapper< reference_wrapper< _Tp> >  { 
# 1746
typedef _Tp &__type; 
# 1747
}; 
# 1749
template< class _Tp> 
# 1750
struct __strip_reference_wrapper< const reference_wrapper< _Tp> >  { 
# 1752
typedef _Tp &__type; 
# 1753
}; 
# 1755
template< class _Tp> 
# 1756
struct __decay_and_strip { 
# 1759
typedef typename __strip_reference_wrapper< typename decay< _Tp> ::type> ::__type __type; 
# 1760
}; 
# 1765
template< bool , class _Tp = void> 
# 1766
struct enable_if { 
# 1767
}; 
# 1770
template< class _Tp> 
# 1771
struct enable_if< true, _Tp>  { 
# 1772
typedef _Tp type; }; 
# 1774
template< class ..._Cond> using _Require = typename enable_if< __and_< _Cond...> ::value> ::type; 
# 1779
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 1780
struct conditional { 
# 1781
typedef _Iftrue type; }; 
# 1784
template< class _Iftrue, class _Iffalse> 
# 1785
struct conditional< false, _Iftrue, _Iffalse>  { 
# 1786
typedef _Iffalse type; }; 
# 1789
template< class ..._Tp> struct common_type; 
# 1794
struct __do_common_type_impl { 
# 1796
template< class _Tp, class _Up> static __success_type< typename decay< __decltype((true ? std::declval< _Tp> () : std::declval< _Up> ()))> ::type>  _S_test(int); 
# 1801
template< class , class > static __failure_type _S_test(...); 
# 1803
}; 
# 1805
template< class _Tp, class _Up> 
# 1806
struct __common_type_impl : private __do_common_type_impl { 
# 1809
typedef __decltype((_S_test< _Tp, _Up> (0))) type; 
# 1810
}; 
# 1812
struct __do_member_type_wrapper { 
# 1814
template< class _Tp> static __success_type< typename _Tp::type>  _S_test(int); 
# 1817
template< class > static __failure_type _S_test(...); 
# 1819
}; 
# 1821
template< class _Tp> 
# 1822
struct __member_type_wrapper : private __do_member_type_wrapper { 
# 1825
typedef __decltype((_S_test< _Tp> (0))) type; 
# 1826
}; 
# 1828
template< class _CTp, class ..._Args> 
# 1829
struct __expanded_common_type_wrapper { 
# 1831
typedef common_type< typename _CTp::type, _Args...>  type; 
# 1832
}; 
# 1834
template< class ..._Args> 
# 1835
struct __expanded_common_type_wrapper< __failure_type, _Args...>  { 
# 1836
typedef __failure_type type; }; 
# 1838
template< class _Tp> 
# 1839
struct common_type< _Tp>  { 
# 1840
typedef typename decay< _Tp> ::type type; }; 
# 1842
template< class _Tp, class _Up> 
# 1843
struct common_type< _Tp, _Up>  : public __common_type_impl< _Tp, _Up> ::type { 
# 1845
}; 
# 1847
template< class _Tp, class _Up, class ..._Vp> 
# 1848
struct common_type< _Tp, _Up, _Vp...>  : public __expanded_common_type_wrapper< typename __member_type_wrapper< common_type< _Tp, _Up> > ::type, _Vp...> ::type { 
# 1851
}; 
# 1854
template< class _Tp> 
# 1855
struct underlying_type { 
# 1857
typedef __underlying_type(_Tp) type; 
# 1858
}; 
# 1860
template< class _Tp> 
# 1861
struct __declval_protector { 
# 1863
static const bool __stop = false; 
# 1864
static typename add_rvalue_reference< _Tp> ::type __delegate(); 
# 1865
}; 
# 1867
template< class _Tp> inline typename add_rvalue_reference< _Tp> ::type 
# 1869
declval() noexcept 
# 1870
{ 
# 1871
static_assert((__declval_protector< _Tp> ::__stop), "declval() must not be used!");
# 1873
return __declval_protector< _Tp> ::__delegate(); 
# 1874
} 
# 1877
template< class _Signature> class result_of; 
# 1883
struct __result_of_memfun_ref_impl { 
# 1885
template< class _Fp, class _Tp1, class ..._Args> static __success_type< __decltype(((std::declval< _Tp1> ().*std::declval< _Fp> ())(std::declval< _Args> ()...)))>  _S_test(int); 
# 1890
template< class ...> static __failure_type _S_test(...); 
# 1892
}; 
# 1894
template< class _MemPtr, class _Arg, class ..._Args> 
# 1895
struct __result_of_memfun_ref : private __result_of_memfun_ref_impl { 
# 1898
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 1899
}; 
# 1902
struct __result_of_memfun_deref_impl { 
# 1904
template< class _Fp, class _Tp1, class ..._Args> static __success_type< __decltype((((*std::declval< _Tp1> ()).*std::declval< _Fp> ())(std::declval< _Args> ()...)))>  _S_test(int); 
# 1909
template< class ...> static __failure_type _S_test(...); 
# 1911
}; 
# 1913
template< class _MemPtr, class _Arg, class ..._Args> 
# 1914
struct __result_of_memfun_deref : private __result_of_memfun_deref_impl { 
# 1917
typedef __decltype((_S_test< _MemPtr, _Arg, _Args...> (0))) type; 
# 1918
}; 
# 1921
struct __result_of_memobj_ref_impl { 
# 1923
template< class _Fp, class _Tp1> static __success_type< __decltype((std::declval< _Tp1> ().*std::declval< _Fp> ()))>  _S_test(int); 
# 1928
template< class , class > static __failure_type _S_test(...); 
# 1930
}; 
# 1932
template< class _MemPtr, class _Arg> 
# 1933
struct __result_of_memobj_ref : private __result_of_memobj_ref_impl { 
# 1936
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 1937
}; 
# 1940
struct __result_of_memobj_deref_impl { 
# 1942
template< class _Fp, class _Tp1> static __success_type< __decltype(((*std::declval< _Tp1> ()).*std::declval< _Fp> ()))>  _S_test(int); 
# 1947
template< class , class > static __failure_type _S_test(...); 
# 1949
}; 
# 1951
template< class _MemPtr, class _Arg> 
# 1952
struct __result_of_memobj_deref : private __result_of_memobj_deref_impl { 
# 1955
typedef __decltype((_S_test< _MemPtr, _Arg> (0))) type; 
# 1956
}; 
# 1958
template< class _MemPtr, class _Arg> struct __result_of_memobj; 
# 1961
template< class _Res, class _Class, class _Arg> 
# 1962
struct __result_of_memobj< _Res (_Class::*), _Arg>  { 
# 1965
typedef typename remove_cv< typename remove_reference< _Arg> ::type> ::type _Argval; 
# 1966
typedef _Res (_Class::*_MemPtr); 
# 1971
typedef typename conditional< __or_< is_same< typename remove_cv< typename remove_reference< _Arg> ::type> ::type, _Class> , is_base_of< _Class, typename remove_cv< typename remove_reference< _Arg> ::type> ::type> > ::value, __result_of_memobj_ref< _Res (_Class::*), _Arg> , __result_of_memobj_deref< _Res (_Class::*), _Arg> > ::type::type type; 
# 1972
}; 
# 1974
template< class _MemPtr, class _Arg, class ..._Args> struct __result_of_memfun; 
# 1977
template< class _Res, class _Class, class _Arg, class ..._Args> 
# 1978
struct __result_of_memfun< _Res (_Class::*), _Arg, _Args...>  { 
# 1981
typedef typename remove_cv< typename remove_reference< _Arg> ::type> ::type _Argval; 
# 1982
typedef _Res (_Class::*_MemPtr); 
# 1987
typedef typename conditional< __or_< is_same< typename remove_cv< typename remove_reference< _Arg> ::type> ::type, _Class> , is_base_of< _Class, typename remove_cv< typename remove_reference< _Arg> ::type> ::type> > ::value, __result_of_memfun_ref< _Res (_Class::*), _Arg, _Args...> , __result_of_memfun_deref< _Res (_Class::*), _Arg, _Args...> > ::type::type type; 
# 1988
}; 
# 1990
template< bool , bool , class _Functor, class ..._ArgTypes> 
# 1991
struct __result_of_impl { 
# 1993
typedef __failure_type type; 
# 1994
}; 
# 1996
template< class _MemPtr, class _Arg> 
# 1997
struct __result_of_impl< true, false, _MemPtr, _Arg>  : public __result_of_memobj< typename decay< _MemPtr> ::type, _Arg>  { 
# 1999
}; 
# 2001
template< class _MemPtr, class _Arg, class ..._Args> 
# 2002
struct __result_of_impl< false, true, _MemPtr, _Arg, _Args...>  : public __result_of_memfun< typename decay< _MemPtr> ::type, _Arg, _Args...>  { 
# 2004
}; 
# 2007
struct __result_of_other_impl { 
# 2009
template< class _Fn, class ..._Args> static __success_type< __decltype((std::declval< _Fn> ()(std::declval< _Args> ()...)))>  _S_test(int); 
# 2014
template< class ...> static __failure_type _S_test(...); 
# 2016
}; 
# 2018
template< class _Functor, class ..._ArgTypes> 
# 2019
struct __result_of_impl< false, false, _Functor, _ArgTypes...>  : private __result_of_other_impl { 
# 2022
typedef __decltype((_S_test< _Functor, _ArgTypes...> (0))) type; 
# 2023
}; 
# 2025
template< class _Functor, class ..._ArgTypes> 
# 2026
struct result_of< _Functor (_ArgTypes ...)>  : public __result_of_impl< is_member_object_pointer< typename remove_reference< _Functor> ::type> ::value, is_member_function_pointer< typename remove_reference< _Functor> ::type> ::value, _Functor, _ArgTypes...> ::type { 
# 2036
}; 
# 2070
}
# 59 "/usr/include/c++/4.8/bits/move.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 74
template< class _Tp> constexpr _Tp &&
# 76
forward(typename remove_reference< _Tp> ::type &__t) noexcept 
# 77
{ return static_cast< _Tp &&>(__t); } 
# 85
template< class _Tp> constexpr _Tp &&
# 87
forward(typename remove_reference< _Tp> ::type &&__t) noexcept 
# 88
{ 
# 89
static_assert((!std::is_lvalue_reference< _Tp> ::value), "template argument substituting _Tp is an lvalue reference type");
# 91
return static_cast< _Tp &&>(__t); 
# 92
} 
# 99
template< class _Tp> constexpr typename remove_reference< _Tp> ::type &&
# 101
move(_Tp &&__t) noexcept 
# 102
{ return static_cast< typename remove_reference< _Tp> ::type &&>(__t); } 
# 105
template< class _Tp> 
# 106
struct __move_if_noexcept_cond : public __and_< __not_< is_nothrow_move_constructible< _Tp> > , is_copy_constructible< _Tp> > ::type { 
# 108
}; 
# 118
template< class _Tp> constexpr typename conditional< __move_if_noexcept_cond< _Tp> ::value, const _Tp &, _Tp &&> ::type 
# 121
move_if_noexcept(_Tp &__x) noexcept 
# 122
{ return std::move(__x); } 
# 133
template< class _Tp> inline _Tp *
# 135
addressof(_Tp &__r) noexcept 
# 136
{ return std::__addressof(__r); } 
# 140
}
# 149
namespace std __attribute((__visibility__("default"))) { 
# 164
template< class _Tp> inline void 
# 166
swap(_Tp &__a, _Tp &__b) noexcept(__and_< is_nothrow_move_constructible< _Tp> , is_nothrow_move_assignable< _Tp> > ::value) 
# 171
{ 
# 175
_Tp __tmp = std::move(__a); 
# 176
__a = std::move(__b); 
# 177
__b = std::move(__tmp); 
# 178
} 
# 183
template< class _Tp, size_t _Nm> inline void 
# 185
swap(_Tp (&__a)[_Nm], _Tp (&__b)[_Nm]) noexcept(noexcept(swap(*(__a), *(__b)))) 
# 189
{ 
# 190
for (size_t __n = (0); __n < _Nm; ++__n) { 
# 191
swap((__a)[__n], (__b)[__n]); }  
# 192
} 
# 196
}
# 65 "/usr/include/c++/4.8/bits/stl_pair.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 76
struct piecewise_construct_t { }; 
# 79
constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t(); 
# 82
template< class ...> class tuple; 
# 85
template< size_t ...> struct _Index_tuple; 
# 95
template< class _T1, class _T2> 
# 96
struct pair { 
# 98
typedef _T1 first_type; 
# 99
typedef _T2 second_type; 
# 101
_T1 first; 
# 102
_T2 second; 
# 108
constexpr pair() : first(), second() 
# 109
{ } 
# 112
constexpr pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 113
{ } 
# 121
template< class _U1, class _U2, class  = typename enable_if< __and_< is_convertible< const _U1 &, _T1> , is_convertible< const _U2 &, _T2> > ::value> ::type> constexpr 
# 124
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 125
{ } 
# 127
constexpr pair(const pair &) = default;
# 128
constexpr pair(pair &&) = default;
# 131
template< class _U1, class  = typename enable_if< is_convertible< _U1, _T1> ::value> ::type> constexpr 
# 133
pair(_U1 &&__x, const _T2 &__y) : first(std::forward< _U1> (__x)), second(__y) 
# 134
{ } 
# 136
template< class _U2, class  = typename enable_if< is_convertible< _U2, _T2> ::value> ::type> constexpr 
# 138
pair(const _T1 &__x, _U2 &&__y) : first(__x), second(std::forward< _U2> (__y)) 
# 139
{ } 
# 141
template< class _U1, class _U2, class  = typename enable_if< __and_< is_convertible< _U1, _T1> , is_convertible< _U2, _T2> > ::value> ::type> constexpr 
# 144
pair(_U1 &&__x, _U2 &&__y) : first(std::forward< _U1> (__x)), second(std::forward< _U2> (__y)) 
# 145
{ } 
# 147
template< class _U1, class _U2, class  = typename enable_if< __and_< is_convertible< _U1, _T1> , is_convertible< _U2, _T2> > ::value> ::type> constexpr 
# 150
pair(std::pair< _U1, _U2>  &&__p) : first(std::forward< _U1> ((__p.first))), second(std::forward< _U2> ((__p.second))) 
# 152
{ } 
# 154
template< class ..._Args1, class ..._Args2> pair(piecewise_construct_t, tuple< _Args1...> , tuple< _Args2...> ); 
# 158
pair &operator=(const pair &__p) 
# 159
{ 
# 160
(first) = (__p.first); 
# 161
(second) = (__p.second); 
# 162
return *this; 
# 163
} 
# 166
pair &operator=(pair &&__p) noexcept(__and_< is_nothrow_move_assignable< _T1> , is_nothrow_move_assignable< _T2> > ::value) 
# 169
{ 
# 170
(first) = std::forward< first_type> (__p.first); 
# 171
(second) = std::forward< second_type> (__p.second); 
# 172
return *this; 
# 173
} 
# 175
template< class _U1, class _U2> pair &
# 177
operator=(const std::pair< _U1, _U2>  &__p) 
# 178
{ 
# 179
(first) = (__p.first); 
# 180
(second) = (__p.second); 
# 181
return *this; 
# 182
} 
# 184
template< class _U1, class _U2> pair &
# 186
operator=(std::pair< _U1, _U2>  &&__p) 
# 187
{ 
# 188
(first) = std::forward< _U1> ((__p.first)); 
# 189
(second) = std::forward< _U2> ((__p.second)); 
# 190
return *this; 
# 191
} 
# 194
void swap(pair &__p) noexcept(noexcept(swap(first, __p.first)) && noexcept(swap(second, __p.second))) 
# 197
{ 
# 198
using std::swap;
# 199
swap(first, __p.first); 
# 200
swap(second, __p.second); 
# 201
} 
# 204
private: template< class ..._Args1, size_t ..._Indexes1, class ...
# 205
_Args2, size_t ..._Indexes2> 
# 204
pair(tuple< _Args1...>  &, tuple< _Args2...>  &, _Index_tuple< _Indexes1...> , _Index_tuple< _Indexes2...> ); 
# 209
}; 
# 212
template< class _T1, class _T2> constexpr bool 
# 214
operator==(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 215
{ return ((__x.first) == (__y.first)) && ((__x.second) == (__y.second)); } 
# 218
template< class _T1, class _T2> constexpr bool 
# 220
operator<(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 221
{ return ((__x.first) < (__y.first)) || ((!((__y.first) < (__x.first))) && ((__x.second) < (__y.second))); 
# 222
} 
# 225
template< class _T1, class _T2> constexpr bool 
# 227
operator!=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 228
{ return !(__x == __y); } 
# 231
template< class _T1, class _T2> constexpr bool 
# 233
operator>(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 234
{ return __y < __x; } 
# 237
template< class _T1, class _T2> constexpr bool 
# 239
operator<=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 240
{ return !(__y < __x); } 
# 243
template< class _T1, class _T2> constexpr bool 
# 245
operator>=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 246
{ return !(__x < __y); } 
# 252
template< class _T1, class _T2> inline void 
# 254
swap(pair< _T1, _T2>  &__x, pair< _T1, _T2>  &__y) noexcept(noexcept((__x.swap(__y)))) 
# 256
{ (__x.swap(__y)); } 
# 273
template< class _T1, class _T2> constexpr pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  
# 276
make_pair(_T1 &&__x, _T2 &&__y) 
# 277
{ 
# 278
typedef typename __decay_and_strip< _T1> ::__type __ds_type1; 
# 279
typedef typename __decay_and_strip< _T2> ::__type __ds_type2; 
# 280
typedef pair< typename __decay_and_strip< _T1> ::__type, typename __decay_and_strip< _T2> ::__type>  __pair_type; 
# 281
return __pair_type(std::forward< _T1> (__x), std::forward< _T2> (__y)); 
# 282
} 
# 293
}
# 70 "/usr/include/c++/4.8/bits/stl_iterator_base_types.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 89
struct input_iterator_tag { }; 
# 92
struct output_iterator_tag { }; 
# 95
struct forward_iterator_tag : public input_iterator_tag { }; 
# 99
struct bidirectional_iterator_tag : public forward_iterator_tag { }; 
# 103
struct random_access_iterator_tag : public bidirectional_iterator_tag { }; 
# 116
template< class _Category, class _Tp, class _Distance = ptrdiff_t, class 
# 117
_Pointer = _Tp *, class _Reference = _Tp &> 
# 118
struct iterator { 
# 121
typedef _Category iterator_category; 
# 123
typedef _Tp value_type; 
# 125
typedef _Distance difference_type; 
# 127
typedef _Pointer pointer; 
# 129
typedef _Reference reference; 
# 130
}; 
# 142
template< class _Tp> class __has_iterator_category_helper : private __sfinae_types { template< class _Up> struct _Wrap_type { }; template< class _Up> static __one __test(_Wrap_type< typename _Up::iterator_category>  *); template< class _Up> static __two __test(...); public: static constexpr bool value = (sizeof(__test< _Tp> (0)) == (1)); }; template< class _Tp> struct __has_iterator_category : public integral_constant< bool, __has_iterator_category_helper< typename remove_cv< _Tp> ::type> ::value>  { }; 
# 144
template< class _Iterator, bool 
# 145
 = __has_iterator_category< _Iterator> ::value> 
# 146
struct __iterator_traits { }; 
# 148
template< class _Iterator> 
# 149
struct __iterator_traits< _Iterator, true>  { 
# 151
typedef typename _Iterator::iterator_category iterator_category; 
# 152
typedef typename _Iterator::value_type value_type; 
# 153
typedef typename _Iterator::difference_type difference_type; 
# 154
typedef typename _Iterator::pointer pointer; 
# 155
typedef typename _Iterator::reference reference; 
# 156
}; 
# 158
template< class _Iterator> 
# 159
struct iterator_traits : public __iterator_traits< _Iterator>  { 
# 160
}; 
# 174
template< class _Tp> 
# 175
struct iterator_traits< _Tp *>  { 
# 177
typedef random_access_iterator_tag iterator_category; 
# 178
typedef _Tp value_type; 
# 179
typedef ptrdiff_t difference_type; 
# 180
typedef _Tp *pointer; 
# 181
typedef _Tp &reference; 
# 182
}; 
# 185
template< class _Tp> 
# 186
struct iterator_traits< const _Tp *>  { 
# 188
typedef random_access_iterator_tag iterator_category; 
# 189
typedef _Tp value_type; 
# 190
typedef ptrdiff_t difference_type; 
# 191
typedef const _Tp *pointer; 
# 192
typedef const _Tp &reference; 
# 193
}; 
# 199
template< class _Iter> inline typename iterator_traits< _Iter> ::iterator_category 
# 201
__iterator_category(const _Iter &) 
# 202
{ return typename iterator_traits< _Iter> ::iterator_category(); } 
# 208
template< class _Iterator, bool _HasBase> 
# 209
struct _Iter_base { 
# 211
typedef _Iterator iterator_type; 
# 212
static iterator_type _S_base(_Iterator __it) 
# 213
{ return __it; } 
# 214
}; 
# 216
template< class _Iterator> 
# 217
struct _Iter_base< _Iterator, true>  { 
# 219
typedef typename _Iterator::iterator_type iterator_type; 
# 220
static iterator_type _S_base(_Iterator __it) 
# 221
{ return (__it.base()); } 
# 222
}; 
# 225
template< class _InIter> using _RequireInputIter = typename enable_if< is_convertible< typename iterator_traits< _InIter> ::iterator_category, input_iterator_tag> ::value> ::type; 
# 233
}
# 46 "/usr/include/c++/4.8/debug/debug.h" 3
namespace std { 
# 48
namespace __debug { }
# 49
}
# 54
namespace __gnu_debug { 
# 56
using namespace std::__debug;
# 57
}
# 67 "/usr/include/c++/4.8/bits/stl_iterator_base_funcs.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 71
template< class _InputIterator> inline typename iterator_traits< _InputIterator> ::difference_type 
# 73
__distance(_InputIterator __first, _InputIterator __last, input_iterator_tag) 
# 75
{ 
# 79
typename iterator_traits< _InputIterator> ::difference_type __n = (0); 
# 80
while (__first != __last) 
# 81
{ 
# 82
++__first; 
# 83
++__n; 
# 84
}  
# 85
return __n; 
# 86
} 
# 88
template< class _RandomAccessIterator> inline typename iterator_traits< _RandomAccessIterator> ::difference_type 
# 90
__distance(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag) 
# 92
{ 
# 96
return __last - __first; 
# 97
} 
# 112
template< class _InputIterator> inline typename iterator_traits< _InputIterator> ::difference_type 
# 114
distance(_InputIterator __first, _InputIterator __last) 
# 115
{ 
# 117
return std::__distance(__first, __last, std::__iterator_category(__first)); 
# 119
} 
# 121
template< class _InputIterator, class _Distance> inline void 
# 123
__advance(_InputIterator &__i, _Distance __n, input_iterator_tag) 
# 124
{ 
# 127
; 
# 128
while (__n--) { 
# 129
++__i; }  
# 130
} 
# 132
template< class _BidirectionalIterator, class _Distance> inline void 
# 134
__advance(_BidirectionalIterator &__i, _Distance __n, bidirectional_iterator_tag) 
# 136
{ 
# 140
if (__n > 0) { 
# 141
while (__n--) { 
# 142
++__i; }  } else { 
# 144
while (__n++) { 
# 145
--__i; }  }  
# 146
} 
# 148
template< class _RandomAccessIterator, class _Distance> inline void 
# 150
__advance(_RandomAccessIterator &__i, _Distance __n, random_access_iterator_tag) 
# 152
{ 
# 156
__i += __n; 
# 157
} 
# 171
template< class _InputIterator, class _Distance> inline void 
# 173
advance(_InputIterator &__i, _Distance __n) 
# 174
{ 
# 176
typename iterator_traits< _InputIterator> ::difference_type __d = __n; 
# 177
std::__advance(__i, __d, std::__iterator_category(__i)); 
# 178
} 
# 182
template< class _ForwardIterator> inline _ForwardIterator 
# 184
next(_ForwardIterator __x, typename iterator_traits< _ForwardIterator> ::difference_type 
# 185
__n = 1) 
# 186
{ 
# 187
std::advance(__x, __n); 
# 188
return __x; 
# 189
} 
# 191
template< class _BidirectionalIterator> inline _BidirectionalIterator 
# 193
prev(_BidirectionalIterator __x, typename iterator_traits< _BidirectionalIterator> ::difference_type 
# 194
__n = 1) 
# 195
{ 
# 196
std::advance(__x, -__n); 
# 197
return __x; 
# 198
} 
# 203
}
# 67 "/usr/include/c++/4.8/bits/stl_iterator.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 95
template< class _Iterator> 
# 96
class reverse_iterator : public iterator< typename iterator_traits< _Iterator> ::iterator_category, typename iterator_traits< _Iterator> ::value_type, typename iterator_traits< _Iterator> ::difference_type, typename iterator_traits< _Iterator> ::pointer, typename iterator_traits< _Iterator> ::reference>  { 
# 104
protected: _Iterator current; 
# 106
typedef iterator_traits< _Iterator>  __traits_type; 
# 109
public: typedef _Iterator iterator_type; 
# 110
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 111
typedef typename iterator_traits< _Iterator> ::pointer pointer; 
# 112
typedef typename iterator_traits< _Iterator> ::reference reference; 
# 120
reverse_iterator() : current() { } 
# 126
explicit reverse_iterator(iterator_type __x) : current(__x) { } 
# 131
reverse_iterator(const reverse_iterator &__x) : current(__x.current) 
# 132
{ } 
# 138
template< class _Iter> 
# 139
reverse_iterator(const ::std::reverse_iterator< _Iter>  &__x) : current((__x.base())) 
# 140
{ } 
# 146
iterator_type base() const 
# 147
{ return current; } 
# 160
reference operator*() const 
# 161
{ 
# 162
_Iterator __tmp = current; 
# 163
return *(--__tmp); 
# 164
} 
# 172
pointer operator->() const 
# 173
{ return &operator*(); } 
# 181
reverse_iterator &operator++() 
# 182
{ 
# 183
--(current); 
# 184
return *this; 
# 185
} 
# 193
reverse_iterator operator++(int) 
# 194
{ 
# 195
reverse_iterator __tmp = *this; 
# 196
--(current); 
# 197
return __tmp; 
# 198
} 
# 206
reverse_iterator &operator--() 
# 207
{ 
# 208
++(current); 
# 209
return *this; 
# 210
} 
# 218
reverse_iterator operator--(int) 
# 219
{ 
# 220
reverse_iterator __tmp = *this; 
# 221
++(current); 
# 222
return __tmp; 
# 223
} 
# 231
reverse_iterator operator+(difference_type __n) const 
# 232
{ return ((reverse_iterator)((current) - __n)); } 
# 241
reverse_iterator &operator+=(difference_type __n) 
# 242
{ 
# 243
(current) -= __n; 
# 244
return *this; 
# 245
} 
# 253
reverse_iterator operator-(difference_type __n) const 
# 254
{ return ((reverse_iterator)((current) + __n)); } 
# 263
reverse_iterator &operator-=(difference_type __n) 
# 264
{ 
# 265
(current) += __n; 
# 266
return *this; 
# 267
} 
# 275
reference operator[](difference_type __n) const 
# 276
{ return *((*this) + __n); } 
# 277
}; 
# 289
template< class _Iterator> inline bool 
# 291
operator==(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 292
__y) 
# 293
{ return (__x.base()) == (__y.base()); } 
# 295
template< class _Iterator> inline bool 
# 297
operator<(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 298
__y) 
# 299
{ return (__y.base()) < (__x.base()); } 
# 301
template< class _Iterator> inline bool 
# 303
operator!=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 304
__y) 
# 305
{ return !(__x == __y); } 
# 307
template< class _Iterator> inline bool 
# 309
operator>(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 310
__y) 
# 311
{ return __y < __x; } 
# 313
template< class _Iterator> inline bool 
# 315
operator<=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 316
__y) 
# 317
{ return !(__y < __x); } 
# 319
template< class _Iterator> inline bool 
# 321
operator>=(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 322
__y) 
# 323
{ return !(__x < __y); } 
# 325
template< class _Iterator> inline typename reverse_iterator< _Iterator> ::difference_type 
# 327
operator-(const reverse_iterator< _Iterator>  &__x, const reverse_iterator< _Iterator>  &
# 328
__y) 
# 329
{ return (__y.base()) - (__x.base()); } 
# 331
template< class _Iterator> inline reverse_iterator< _Iterator>  
# 333
operator+(typename reverse_iterator< _Iterator> ::difference_type __n, const reverse_iterator< _Iterator>  &
# 334
__x) 
# 335
{ return ((reverse_iterator< _Iterator> )((__x.base()) - __n)); } 
# 339
template< class _IteratorL, class _IteratorR> inline bool 
# 341
operator==(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 342
__y) 
# 343
{ return (__x.base()) == (__y.base()); } 
# 345
template< class _IteratorL, class _IteratorR> inline bool 
# 347
operator<(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 348
__y) 
# 349
{ return (__y.base()) < (__x.base()); } 
# 351
template< class _IteratorL, class _IteratorR> inline bool 
# 353
operator!=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 354
__y) 
# 355
{ return !(__x == __y); } 
# 357
template< class _IteratorL, class _IteratorR> inline bool 
# 359
operator>(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 360
__y) 
# 361
{ return __y < __x; } 
# 363
template< class _IteratorL, class _IteratorR> inline bool 
# 365
operator<=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 366
__y) 
# 367
{ return !(__y < __x); } 
# 369
template< class _IteratorL, class _IteratorR> inline bool 
# 371
operator>=(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 372
__y) 
# 373
{ return !(__x < __y); } 
# 375
template< class _IteratorL, class _IteratorR> inline auto 
# 379
operator-(const reverse_iterator< _IteratorL>  &__x, const reverse_iterator< _IteratorR>  &
# 380
__y)->__decltype(((__y.base()) - (__x.base()))) 
# 387
{ return (__y.base()) - (__x.base()); } 
# 401
template< class _Container> 
# 402
class back_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 406
protected: _Container *container; 
# 410
public: typedef _Container container_type; 
# 414
explicit back_insert_iterator(_Container &__x) : container((&__x)) { } 
# 436
back_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 437
{ 
# 438
((container)->push_back(__value)); 
# 439
return *this; 
# 440
} 
# 443
back_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 444
{ 
# 445
((container)->push_back(std::move(__value))); 
# 446
return *this; 
# 447
} 
# 452
back_insert_iterator &operator*() 
# 453
{ return *this; } 
# 457
back_insert_iterator &operator++() 
# 458
{ return *this; } 
# 462
back_insert_iterator operator++(int) 
# 463
{ return *this; } 
# 464
}; 
# 477
template< class _Container> inline back_insert_iterator< _Container>  
# 479
back_inserter(_Container &__x) 
# 480
{ return ((back_insert_iterator< _Container> )(__x)); } 
# 492
template< class _Container> 
# 493
class front_insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 497
protected: _Container *container; 
# 501
public: typedef _Container container_type; 
# 504
explicit front_insert_iterator(_Container &__x) : container((&__x)) { } 
# 526
front_insert_iterator &operator=(const typename _Container::value_type &__value) 
# 527
{ 
# 528
((container)->push_front(__value)); 
# 529
return *this; 
# 530
} 
# 533
front_insert_iterator &operator=(typename _Container::value_type &&__value) 
# 534
{ 
# 535
((container)->push_front(std::move(__value))); 
# 536
return *this; 
# 537
} 
# 542
front_insert_iterator &operator*() 
# 543
{ return *this; } 
# 547
front_insert_iterator &operator++() 
# 548
{ return *this; } 
# 552
front_insert_iterator operator++(int) 
# 553
{ return *this; } 
# 554
}; 
# 567
template< class _Container> inline front_insert_iterator< _Container>  
# 569
front_inserter(_Container &__x) 
# 570
{ return ((front_insert_iterator< _Container> )(__x)); } 
# 586
template< class _Container> 
# 587
class insert_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 591
protected: _Container *container; 
# 592
typename _Container::iterator iter; 
# 596
public: typedef _Container container_type; 
# 602
insert_iterator(_Container &__x, typename _Container::iterator __i) : container((&__x)), iter(__i) 
# 603
{ } 
# 638
insert_iterator &operator=(const typename _Container::value_type &__value) 
# 639
{ 
# 640
(iter) = ((container)->insert(iter, __value)); 
# 641
++(iter); 
# 642
return *this; 
# 643
} 
# 646
insert_iterator &operator=(typename _Container::value_type &&__value) 
# 647
{ 
# 648
(iter) = ((container)->insert(iter, std::move(__value))); 
# 649
++(iter); 
# 650
return *this; 
# 651
} 
# 656
insert_iterator &operator*() 
# 657
{ return *this; } 
# 661
insert_iterator &operator++() 
# 662
{ return *this; } 
# 666
insert_iterator &operator++(int) 
# 667
{ return *this; } 
# 668
}; 
# 681
template< class _Container, class _Iterator> inline insert_iterator< _Container>  
# 683
inserter(_Container &__x, _Iterator __i) 
# 684
{ 
# 685
return insert_iterator< _Container> (__x, (typename _Container::iterator)__i); 
# 687
} 
# 692
}
# 694
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 705
using std::iterator_traits;
# 706
using std::iterator;
# 707
template< class _Iterator, class _Container> 
# 708
class __normal_iterator { 
# 711
protected: _Iterator _M_current; 
# 713
typedef std::iterator_traits< _Iterator>  __traits_type; 
# 716
public: typedef _Iterator iterator_type; 
# 717
typedef typename std::iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 718
typedef typename std::iterator_traits< _Iterator> ::value_type value_type; 
# 719
typedef typename std::iterator_traits< _Iterator> ::difference_type difference_type; 
# 720
typedef typename std::iterator_traits< _Iterator> ::reference reference; 
# 721
typedef typename std::iterator_traits< _Iterator> ::pointer pointer; 
# 723
constexpr __normal_iterator() : _M_current(_Iterator()) { } 
# 726
explicit __normal_iterator(const _Iterator &__i) : _M_current(__i) { } 
# 729
template< class _Iter> 
# 730
__normal_iterator(const __gnu_cxx::__normal_iterator< _Iter, typename __enable_if< std::__are_same< _Iter, typename _Container::pointer> ::__value, _Container> ::__type>  &
# 733
__i) : _M_current((__i.base())) 
# 734
{ } 
# 738
reference operator*() const 
# 739
{ return *(_M_current); } 
# 742
pointer operator->() const 
# 743
{ return _M_current; } 
# 746
__normal_iterator &operator++() 
# 747
{ 
# 748
++(_M_current); 
# 749
return *this; 
# 750
} 
# 753
__normal_iterator operator++(int) 
# 754
{ return ((__normal_iterator)((_M_current)++)); } 
# 758
__normal_iterator &operator--() 
# 759
{ 
# 760
--(_M_current); 
# 761
return *this; 
# 762
} 
# 765
__normal_iterator operator--(int) 
# 766
{ return ((__normal_iterator)((_M_current)--)); } 
# 770
reference operator[](const difference_type &__n) const 
# 771
{ return (_M_current)[__n]; } 
# 774
__normal_iterator &operator+=(const difference_type &__n) 
# 775
{ (_M_current) += __n; return *this; } 
# 778
__normal_iterator operator+(const difference_type &__n) const 
# 779
{ return ((__normal_iterator)((_M_current) + __n)); } 
# 782
__normal_iterator &operator-=(const difference_type &__n) 
# 783
{ (_M_current) -= __n; return *this; } 
# 786
__normal_iterator operator-(const difference_type &__n) const 
# 787
{ return ((__normal_iterator)((_M_current) - __n)); } 
# 790
const _Iterator &base() const 
# 791
{ return _M_current; } 
# 792
}; 
# 803
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 805
operator==(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 806
__rhs) 
# 807
{ return (__lhs.base()) == (__rhs.base()); } 
# 809
template< class _Iterator, class _Container> inline bool 
# 811
operator==(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 812
__rhs) 
# 813
{ return (__lhs.base()) == (__rhs.base()); } 
# 815
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 817
operator!=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 818
__rhs) 
# 819
{ return (__lhs.base()) != (__rhs.base()); } 
# 821
template< class _Iterator, class _Container> inline bool 
# 823
operator!=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 824
__rhs) 
# 825
{ return (__lhs.base()) != (__rhs.base()); } 
# 828
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 830
operator<(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 831
__rhs) 
# 832
{ return (__lhs.base()) < (__rhs.base()); } 
# 834
template< class _Iterator, class _Container> inline bool 
# 836
operator<(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 837
__rhs) 
# 838
{ return (__lhs.base()) < (__rhs.base()); } 
# 840
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 842
operator>(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 843
__rhs) 
# 844
{ return (__lhs.base()) > (__rhs.base()); } 
# 846
template< class _Iterator, class _Container> inline bool 
# 848
operator>(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 849
__rhs) 
# 850
{ return (__lhs.base()) > (__rhs.base()); } 
# 852
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 854
operator<=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 855
__rhs) 
# 856
{ return (__lhs.base()) <= (__rhs.base()); } 
# 858
template< class _Iterator, class _Container> inline bool 
# 860
operator<=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 861
__rhs) 
# 862
{ return (__lhs.base()) <= (__rhs.base()); } 
# 864
template< class _IteratorL, class _IteratorR, class _Container> inline bool 
# 866
operator>=(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 867
__rhs) 
# 868
{ return (__lhs.base()) >= (__rhs.base()); } 
# 870
template< class _Iterator, class _Container> inline bool 
# 872
operator>=(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 873
__rhs) 
# 874
{ return (__lhs.base()) >= (__rhs.base()); } 
# 880
template< class _IteratorL, class _IteratorR, class _Container> inline auto 
# 884
operator-(const __normal_iterator< _IteratorL, _Container>  &__lhs, const __normal_iterator< _IteratorR, _Container>  &
# 885
__rhs)->__decltype(((__lhs.base()) - (__rhs.base()))) 
# 892
{ return (__lhs.base()) - (__rhs.base()); } 
# 894
template< class _Iterator, class _Container> inline typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 896
operator-(const __normal_iterator< _Iterator, _Container>  &__lhs, const __normal_iterator< _Iterator, _Container>  &
# 897
__rhs) 
# 898
{ return (__lhs.base()) - (__rhs.base()); } 
# 900
template< class _Iterator, class _Container> inline __normal_iterator< _Iterator, _Container>  
# 902
operator+(typename __normal_iterator< _Iterator, _Container> ::difference_type 
# 903
__n, const __normal_iterator< _Iterator, _Container>  &__i) 
# 904
{ return ((__normal_iterator< _Iterator, _Container> )((__i.base()) + __n)); } 
# 907
}
# 911
namespace std __attribute((__visibility__("default"))) { 
# 929
template< class _Iterator> 
# 930
class move_iterator { 
# 933
protected: _Iterator _M_current; 
# 935
typedef iterator_traits< _Iterator>  __traits_type; 
# 938
public: typedef _Iterator iterator_type; 
# 939
typedef typename iterator_traits< _Iterator> ::iterator_category iterator_category; 
# 940
typedef typename iterator_traits< _Iterator> ::value_type value_type; 
# 941
typedef typename iterator_traits< _Iterator> ::difference_type difference_type; 
# 943
typedef _Iterator pointer; 
# 944
typedef value_type &&reference; 
# 946
move_iterator() : _M_current() 
# 947
{ } 
# 950
explicit move_iterator(iterator_type __i) : _M_current(__i) 
# 951
{ } 
# 953
template< class _Iter> 
# 954
move_iterator(const std::move_iterator< _Iter>  &__i) : _M_current((__i.base())) 
# 955
{ } 
# 958
iterator_type base() const 
# 959
{ return _M_current; } 
# 962
reference operator*() const 
# 963
{ return std::move(*(_M_current)); } 
# 966
pointer operator->() const 
# 967
{ return _M_current; } 
# 970
move_iterator &operator++() 
# 971
{ 
# 972
++(_M_current); 
# 973
return *this; 
# 974
} 
# 977
move_iterator operator++(int) 
# 978
{ 
# 979
move_iterator __tmp = *this; 
# 980
++(_M_current); 
# 981
return __tmp; 
# 982
} 
# 985
move_iterator &operator--() 
# 986
{ 
# 987
--(_M_current); 
# 988
return *this; 
# 989
} 
# 992
move_iterator operator--(int) 
# 993
{ 
# 994
move_iterator __tmp = *this; 
# 995
--(_M_current); 
# 996
return __tmp; 
# 997
} 
# 1000
move_iterator operator+(difference_type __n) const 
# 1001
{ return ((move_iterator)((_M_current) + __n)); } 
# 1004
move_iterator &operator+=(difference_type __n) 
# 1005
{ 
# 1006
(_M_current) += __n; 
# 1007
return *this; 
# 1008
} 
# 1011
move_iterator operator-(difference_type __n) const 
# 1012
{ return ((move_iterator)((_M_current) - __n)); } 
# 1015
move_iterator &operator-=(difference_type __n) 
# 1016
{ 
# 1017
(_M_current) -= __n; 
# 1018
return *this; 
# 1019
} 
# 1022
reference operator[](difference_type __n) const 
# 1023
{ return std::move((_M_current)[__n]); } 
# 1024
}; 
# 1029
template< class _IteratorL, class _IteratorR> inline bool 
# 1031
operator==(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1032
__y) 
# 1033
{ return (__x.base()) == (__y.base()); } 
# 1035
template< class _Iterator> inline bool 
# 1037
operator==(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1038
__y) 
# 1039
{ return (__x.base()) == (__y.base()); } 
# 1041
template< class _IteratorL, class _IteratorR> inline bool 
# 1043
operator!=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1044
__y) 
# 1045
{ return !(__x == __y); } 
# 1047
template< class _Iterator> inline bool 
# 1049
operator!=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1050
__y) 
# 1051
{ return !(__x == __y); } 
# 1053
template< class _IteratorL, class _IteratorR> inline bool 
# 1055
operator<(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1056
__y) 
# 1057
{ return (__x.base()) < (__y.base()); } 
# 1059
template< class _Iterator> inline bool 
# 1061
operator<(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1062
__y) 
# 1063
{ return (__x.base()) < (__y.base()); } 
# 1065
template< class _IteratorL, class _IteratorR> inline bool 
# 1067
operator<=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1068
__y) 
# 1069
{ return !(__y < __x); } 
# 1071
template< class _Iterator> inline bool 
# 1073
operator<=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1074
__y) 
# 1075
{ return !(__y < __x); } 
# 1077
template< class _IteratorL, class _IteratorR> inline bool 
# 1079
operator>(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1080
__y) 
# 1081
{ return __y < __x; } 
# 1083
template< class _Iterator> inline bool 
# 1085
operator>(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1086
__y) 
# 1087
{ return __y < __x; } 
# 1089
template< class _IteratorL, class _IteratorR> inline bool 
# 1091
operator>=(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1092
__y) 
# 1093
{ return !(__x < __y); } 
# 1095
template< class _Iterator> inline bool 
# 1097
operator>=(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1098
__y) 
# 1099
{ return !(__x < __y); } 
# 1102
template< class _IteratorL, class _IteratorR> inline auto 
# 1104
operator-(const move_iterator< _IteratorL>  &__x, const move_iterator< _IteratorR>  &
# 1105
__y)->__decltype(((__x.base()) - (__y.base()))) 
# 1107
{ return (__x.base()) - (__y.base()); } 
# 1109
template< class _Iterator> inline auto 
# 1111
operator-(const move_iterator< _Iterator>  &__x, const move_iterator< _Iterator>  &
# 1112
__y)->__decltype(((__x.base()) - (__y.base()))) 
# 1114
{ return (__x.base()) - (__y.base()); } 
# 1116
template< class _Iterator> inline move_iterator< _Iterator>  
# 1118
operator+(typename move_iterator< _Iterator> ::difference_type __n, const move_iterator< _Iterator>  &
# 1119
__x) 
# 1120
{ return __x + __n; } 
# 1122
template< class _Iterator> inline move_iterator< _Iterator>  
# 1124
make_move_iterator(_Iterator __i) 
# 1125
{ return ((move_iterator< _Iterator> )(__i)); } 
# 1127
template< class _Iterator, class _ReturnType = typename conditional< __move_if_noexcept_cond< typename iterator_traits< _Iterator> ::value_type> ::value, _Iterator, move_iterator< _Iterator> > ::type> inline _ReturnType 
# 1132
__make_move_if_noexcept_iterator(_Iterator __i) 
# 1133
{ return (_ReturnType)__i; } 
# 1138
}
# 72 "/usr/include/c++/4.8/bits/stl_algobase.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 117
template< class _ForwardIterator1, class _ForwardIterator2> inline void 
# 119
iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b) 
# 120
{ 
# 147
swap(*__a, *__b); 
# 149
} 
# 163
template< class _ForwardIterator1, class _ForwardIterator2> _ForwardIterator2 
# 165
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 
# 166
__first2) 
# 167
{ 
# 173
; 
# 175
for (; __first1 != __last1; (++__first1), (++__first2)) { 
# 176
std::iter_swap(__first1, __first2); }  
# 177
return __first2; 
# 178
} 
# 191
template< class _Tp> inline const _Tp &
# 193
min(const _Tp &__a, const _Tp &__b) 
# 194
{ 
# 198
if (__b < __a) { 
# 199
return __b; }  
# 200
return __a; 
# 201
} 
# 214
template< class _Tp> inline const _Tp &
# 216
max(const _Tp &__a, const _Tp &__b) 
# 217
{ 
# 221
if (__a < __b) { 
# 222
return __b; }  
# 223
return __a; 
# 224
} 
# 237
template< class _Tp, class _Compare> inline const _Tp &
# 239
min(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 240
{ 
# 242
if (__comp(__b, __a)) { 
# 243
return __b; }  
# 244
return __a; 
# 245
} 
# 258
template< class _Tp, class _Compare> inline const _Tp &
# 260
max(const _Tp &__a, const _Tp &__b, _Compare __comp) 
# 261
{ 
# 263
if (__comp(__a, __b)) { 
# 264
return __b; }  
# 265
return __a; 
# 266
} 
# 270
template< class _Iterator> 
# 271
struct _Niter_base : public _Iter_base< _Iterator, __is_normal_iterator< _Iterator> ::__value>  { 
# 273
}; 
# 275
template< class _Iterator> inline typename _Niter_base< _Iterator> ::iterator_type 
# 277
__niter_base(_Iterator __it) 
# 278
{ return std::_Niter_base< _Iterator> ::_S_base(__it); } 
# 281
template< class _Iterator> 
# 282
struct _Miter_base : public _Iter_base< _Iterator, __is_move_iterator< _Iterator> ::__value>  { 
# 284
}; 
# 286
template< class _Iterator> inline typename _Miter_base< _Iterator> ::iterator_type 
# 288
__miter_base(_Iterator __it) 
# 289
{ return std::_Miter_base< _Iterator> ::_S_base(__it); } 
# 297
template< bool , bool , class > 
# 298
struct __copy_move { 
# 300
template< class _II, class _OI> static _OI 
# 302
__copy_m(_II __first, _II __last, _OI __result) 
# 303
{ 
# 304
for (; __first != __last; (++__result), (++__first)) { 
# 305
(*__result) = (*__first); }  
# 306
return __result; 
# 307
} 
# 308
}; 
# 311
template< class _Category> 
# 312
struct __copy_move< true, false, _Category>  { 
# 314
template< class _II, class _OI> static _OI 
# 316
__copy_m(_II __first, _II __last, _OI __result) 
# 317
{ 
# 318
for (; __first != __last; (++__result), (++__first)) { 
# 319
(*__result) = std::move(*__first); }  
# 320
return __result; 
# 321
} 
# 322
}; 
# 326
template<> struct __copy_move< false, false, random_access_iterator_tag>  { 
# 328
template< class _II, class _OI> static _OI 
# 330
__copy_m(_II __first, _II __last, _OI __result) 
# 331
{ 
# 332
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 333
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 334
{ 
# 335
(*__result) = (*__first); 
# 336
++__first; 
# 337
++__result; 
# 338
}  
# 339
return __result; 
# 340
} 
# 341
}; 
# 345
template<> struct __copy_move< true, false, random_access_iterator_tag>  { 
# 347
template< class _II, class _OI> static _OI 
# 349
__copy_m(_II __first, _II __last, _OI __result) 
# 350
{ 
# 351
typedef typename iterator_traits< _II> ::difference_type _Distance; 
# 352
for (_Distance __n = __last - __first; __n > 0; --__n) 
# 353
{ 
# 354
(*__result) = std::move(*__first); 
# 355
++__first; 
# 356
++__result; 
# 357
}  
# 358
return __result; 
# 359
} 
# 360
}; 
# 363
template< bool _IsMove> 
# 364
struct __copy_move< _IsMove, true, random_access_iterator_tag>  { 
# 366
template< class _Tp> static _Tp *
# 368
__copy_m(const _Tp *__first, const _Tp *__last, _Tp *__result) 
# 369
{ 
# 370
const ptrdiff_t _Num = __last - __first; 
# 371
if (_Num) { 
# 372
__builtin_memmove(__result, __first, sizeof(_Tp) * _Num); }  
# 373
return __result + _Num; 
# 374
} 
# 375
}; 
# 377
template< bool _IsMove, class _II, class _OI> inline _OI 
# 379
__copy_move_a(_II __first, _II __last, _OI __result) 
# 380
{ 
# 381
typedef typename iterator_traits< _II> ::value_type _ValueTypeI; 
# 382
typedef typename iterator_traits< _OI> ::value_type _ValueTypeO; 
# 383
typedef typename iterator_traits< _II> ::iterator_category _Category; 
# 384
const bool __simple = (__is_trivial(_ValueTypeI) && __is_pointer< _II> ::__value && __is_pointer< _OI> ::__value && __are_same< typename iterator_traits< _II> ::value_type, typename iterator_traits< _OI> ::value_type> ::__value); 
# 389
return std::__copy_move< _IsMove, __simple, typename iterator_traits< _II> ::iterator_category> ::__copy_m(__first, __last, __result); 
# 391
} 
# 395
template< class _CharT> struct char_traits; 
# 398
template< class _CharT, class _Traits> class istreambuf_iterator; 
# 401
template< class _CharT, class _Traits> class ostreambuf_iterator; 
# 404
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(_CharT *, _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 410
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type __copy_move_a2(const _CharT *, const _CharT *, ostreambuf_iterator< _CharT, char_traits< _CharT> > ); 
# 416
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type __copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> > , istreambuf_iterator< _CharT, char_traits< _CharT> > , _CharT *); 
# 422
template< bool _IsMove, class _II, class _OI> inline _OI 
# 424
__copy_move_a2(_II __first, _II __last, _OI __result) 
# 425
{ 
# 426
return (_OI)std::__copy_move_a< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result)); 
# 429
} 
# 448
template< class _II, class _OI> inline _OI 
# 450
copy(_II __first, _II __last, _OI __result) 
# 451
{ 
# 456
; 
# 458
return std::__copy_move_a2< __is_move_iterator< _II> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 461
} 
# 481
template< class _II, class _OI> inline _OI 
# 483
move(_II __first, _II __last, _OI __result) 
# 484
{ 
# 489
; 
# 491
return std::__copy_move_a2< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 493
} 
# 500
template< bool , bool , class > 
# 501
struct __copy_move_backward { 
# 503
template< class _BI1, class _BI2> static _BI2 
# 505
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 506
{ 
# 507
while (__first != __last) { 
# 508
(*(--__result)) = (*(--__last)); }  
# 509
return __result; 
# 510
} 
# 511
}; 
# 514
template< class _Category> 
# 515
struct __copy_move_backward< true, false, _Category>  { 
# 517
template< class _BI1, class _BI2> static _BI2 
# 519
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 520
{ 
# 521
while (__first != __last) { 
# 522
(*(--__result)) = std::move(*(--__last)); }  
# 523
return __result; 
# 524
} 
# 525
}; 
# 529
template<> struct __copy_move_backward< false, false, random_access_iterator_tag>  { 
# 531
template< class _BI1, class _BI2> static _BI2 
# 533
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 534
{ 
# 535
typename iterator_traits< _BI1> ::difference_type __n; 
# 536
for (__n = (__last - __first); __n > 0; --__n) { 
# 537
(*(--__result)) = (*(--__last)); }  
# 538
return __result; 
# 539
} 
# 540
}; 
# 544
template<> struct __copy_move_backward< true, false, random_access_iterator_tag>  { 
# 546
template< class _BI1, class _BI2> static _BI2 
# 548
__copy_move_b(_BI1 __first, _BI1 __last, _BI2 __result) 
# 549
{ 
# 550
typename iterator_traits< _BI1> ::difference_type __n; 
# 551
for (__n = (__last - __first); __n > 0; --__n) { 
# 552
(*(--__result)) = std::move(*(--__last)); }  
# 553
return __result; 
# 554
} 
# 555
}; 
# 558
template< bool _IsMove> 
# 559
struct __copy_move_backward< _IsMove, true, random_access_iterator_tag>  { 
# 561
template< class _Tp> static _Tp *
# 563
__copy_move_b(const _Tp *__first, const _Tp *__last, _Tp *__result) 
# 564
{ 
# 565
const ptrdiff_t _Num = __last - __first; 
# 566
if (_Num) { 
# 567
__builtin_memmove(__result - _Num, __first, sizeof(_Tp) * _Num); }  
# 568
return __result - _Num; 
# 569
} 
# 570
}; 
# 572
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 574
__copy_move_backward_a(_BI1 __first, _BI1 __last, _BI2 __result) 
# 575
{ 
# 576
typedef typename iterator_traits< _BI1> ::value_type _ValueType1; 
# 577
typedef typename iterator_traits< _BI2> ::value_type _ValueType2; 
# 578
typedef typename iterator_traits< _BI1> ::iterator_category _Category; 
# 579
const bool __simple = (__is_trivial(_ValueType1) && __is_pointer< _BI1> ::__value && __is_pointer< _BI2> ::__value && __are_same< typename iterator_traits< _BI1> ::value_type, typename iterator_traits< _BI2> ::value_type> ::__value); 
# 584
return std::__copy_move_backward< _IsMove, __simple, typename iterator_traits< _BI1> ::iterator_category> ::__copy_move_b(__first, __last, __result); 
# 588
} 
# 590
template< bool _IsMove, class _BI1, class _BI2> inline _BI2 
# 592
__copy_move_backward_a2(_BI1 __first, _BI1 __last, _BI2 __result) 
# 593
{ 
# 594
return (_BI2)std::__copy_move_backward_a< _IsMove> (std::__niter_base(__first), std::__niter_base(__last), std::__niter_base(__result)); 
# 597
} 
# 617
template< class _BI1, class _BI2> inline _BI2 
# 619
copy_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 620
{ 
# 627
; 
# 629
return std::__copy_move_backward_a2< __is_move_iterator< _BI1> ::__value> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 632
} 
# 653
template< class _BI1, class _BI2> inline _BI2 
# 655
move_backward(_BI1 __first, _BI1 __last, _BI2 __result) 
# 656
{ 
# 663
; 
# 665
return std::__copy_move_backward_a2< true> (std::__miter_base(__first), std::__miter_base(__last), __result); 
# 668
} 
# 675
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, void> ::__type 
# 678
__fill_a(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 679
__value) 
# 680
{ 
# 681
for (; __first != __last; ++__first) { 
# 682
(*__first) = __value; }  
# 683
} 
# 685
template< class _ForwardIterator, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, void> ::__type 
# 688
__fill_a(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 689
__value) 
# 690
{ 
# 691
const _Tp __tmp = __value; 
# 692
for (; __first != __last; ++__first) { 
# 693
(*__first) = __tmp; }  
# 694
} 
# 697
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_byte< _Tp> ::__value, void> ::__type 
# 700
__fill_a(_Tp *__first, _Tp *__last, const _Tp &__c) 
# 701
{ 
# 702
const _Tp __tmp = __c; 
# 703
__builtin_memset(__first, static_cast< unsigned char>(__tmp), __last - __first); 
# 705
} 
# 719
template< class _ForwardIterator, class _Tp> inline void 
# 721
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp &__value) 
# 722
{ 
# 726
; 
# 728
std::__fill_a(std::__niter_base(__first), std::__niter_base(__last), __value); 
# 730
} 
# 732
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< !__is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 735
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 736
{ 
# 737
for (__decltype((__n + 0)) __niter = __n; __niter > 0; (--__niter), (++__first)) { 
# 739
(*__first) = __value; }  
# 740
return __first; 
# 741
} 
# 743
template< class _OutputIterator, class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< __is_scalar< _Tp> ::__value, _OutputIterator> ::__type 
# 746
__fill_n_a(_OutputIterator __first, _Size __n, const _Tp &__value) 
# 747
{ 
# 748
const _Tp __tmp = __value; 
# 749
for (__decltype((__n + 0)) __niter = __n; __niter > 0; (--__niter), (++__first)) { 
# 751
(*__first) = __tmp; }  
# 752
return __first; 
# 753
} 
# 755
template< class _Size, class _Tp> inline typename __gnu_cxx::__enable_if< __is_byte< _Tp> ::__value, _Tp *> ::__type 
# 758
__fill_n_a(_Tp *__first, _Size __n, const _Tp &__c) 
# 759
{ 
# 760
std::__fill_a(__first, __first + __n, __c); 
# 761
return __first + __n; 
# 762
} 
# 779
template< class _OI, class _Size, class _Tp> inline _OI 
# 781
fill_n(_OI __first, _Size __n, const _Tp &__value) 
# 782
{ 
# 786
return (_OI)std::__fill_n_a(std::__niter_base(__first), __n, __value); 
# 787
} 
# 789
template< bool _BoolType> 
# 790
struct __equal { 
# 792
template< class _II1, class _II2> static bool 
# 794
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 795
{ 
# 796
for (; __first1 != __last1; (++__first1), (++__first2)) { 
# 797
if (!((*__first1) == (*__first2))) { 
# 798
return false; }  }  
# 799
return true; 
# 800
} 
# 801
}; 
# 804
template<> struct __equal< true>  { 
# 806
template< class _Tp> static bool 
# 808
equal(const _Tp *__first1, const _Tp *__last1, const _Tp *__first2) 
# 809
{ 
# 810
return !(__builtin_memcmp(__first1, __first2, sizeof(_Tp) * (__last1 - __first1))); 
# 812
} 
# 813
}; 
# 815
template< class _II1, class _II2> inline bool 
# 817
__equal_aux(_II1 __first1, _II1 __last1, _II2 __first2) 
# 818
{ 
# 819
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 820
typedef typename iterator_traits< _II2> ::value_type _ValueType2; 
# 821
const bool __simple = ((__is_integer< typename iterator_traits< _II1> ::value_type> ::__value || __is_pointer< typename iterator_traits< _II1> ::value_type> ::__value) && __is_pointer< _II1> ::__value && __is_pointer< _II2> ::__value && __are_same< typename iterator_traits< _II1> ::value_type, typename iterator_traits< _II2> ::value_type> ::__value); 
# 827
return std::__equal< __simple> ::equal(__first1, __last1, __first2); 
# 828
} 
# 831
template< class , class > 
# 832
struct __lc_rai { 
# 834
template< class _II1, class _II2> static _II1 
# 836
__newlast1(_II1, _II1 __last1, _II2, _II2) 
# 837
{ return __last1; } 
# 839
template< class _II> static bool 
# 841
__cnd2(_II __first, _II __last) 
# 842
{ return __first != __last; } 
# 843
}; 
# 846
template<> struct __lc_rai< random_access_iterator_tag, random_access_iterator_tag>  { 
# 848
template< class _RAI1, class _RAI2> static _RAI1 
# 850
__newlast1(_RAI1 __first1, _RAI1 __last1, _RAI2 
# 851
__first2, _RAI2 __last2) 
# 852
{ 
# 854
const typename iterator_traits< _RAI1> ::difference_type __diff1 = __last1 - __first1; 
# 856
const typename iterator_traits< _RAI2> ::difference_type __diff2 = __last2 - __first2; 
# 857
return (__diff2 < __diff1) ? __first1 + __diff2 : __last1; 
# 858
} 
# 860
template< class _RAI> static bool 
# 862
__cnd2(_RAI, _RAI) 
# 863
{ return true; } 
# 864
}; 
# 866
template< bool _BoolType> 
# 867
struct __lexicographical_compare { 
# 869
template< class _II1, class _II2> static bool __lc(_II1, _II1, _II2, _II2); 
# 871
}; 
# 873
template< bool _BoolType> 
# 874
template< class _II1, class _II2> bool 
# 877
__lexicographical_compare< _BoolType> ::__lc(_II1 __first1, _II1 __last1, _II2 __first2, _II2 __last2) 
# 878
{ 
# 879
typedef typename iterator_traits< _II1> ::iterator_category _Category1; 
# 880
typedef typename iterator_traits< _II2> ::iterator_category _Category2; 
# 881
typedef __lc_rai< typename iterator_traits< _II1> ::iterator_category, typename iterator_traits< _II2> ::iterator_category>  __rai_type; 
# 883
__last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2); 
# 885
for (; (__first1 != __last1) && __rai_type::__cnd2(__first2, __last2); (++__first1), (++__first2)) 
# 887
{ 
# 888
if ((*__first1) < (*__first2)) { 
# 889
return true; }  
# 890
if ((*__first2) < (*__first1)) { 
# 891
return false; }  
# 892
}  
# 893
return (__first1 == __last1) && (__first2 != __last2); 
# 894
} 
# 897
template<> struct __lexicographical_compare< true>  { 
# 899
template< class _Tp, class _Up> static bool 
# 901
__lc(const _Tp *__first1, const _Tp *__last1, const _Up *
# 902
__first2, const _Up *__last2) 
# 903
{ 
# 904
const size_t __len1 = __last1 - __first1; 
# 905
const size_t __len2 = __last2 - __first2; 
# 906
const int __result = __builtin_memcmp(__first1, __first2, std::min(__len1, __len2)); 
# 908
return (__result != 0) ? __result < 0 : (__len1 < __len2); 
# 909
} 
# 910
}; 
# 912
template< class _II1, class _II2> inline bool 
# 914
__lexicographical_compare_aux(_II1 __first1, _II1 __last1, _II2 
# 915
__first2, _II2 __last2) 
# 916
{ 
# 917
typedef typename iterator_traits< _II1> ::value_type _ValueType1; 
# 918
typedef typename iterator_traits< _II2> ::value_type _ValueType2; 
# 919
const bool __simple = (__is_byte< typename iterator_traits< _II1> ::value_type> ::__value && __is_byte< typename iterator_traits< _II2> ::value_type> ::__value && (!__gnu_cxx::__numeric_traits< typename iterator_traits< _II1> ::value_type> ::__is_signed) && (!__gnu_cxx::__numeric_traits< typename iterator_traits< _II2> ::value_type> ::__is_signed) && __is_pointer< _II1> ::__value && __is_pointer< _II2> ::__value); 
# 926
return std::__lexicographical_compare< __simple> ::__lc(__first1, __last1, __first2, __last2); 
# 928
} 
# 941
template< class _ForwardIterator, class _Tp> _ForwardIterator 
# 943
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp &
# 944
__val) 
# 945
{ 
# 951
typedef typename iterator_traits< _ForwardIterator> ::difference_type _DistanceType; 
# 956
; 
# 958
_DistanceType __len = std::distance(__first, __last); 
# 960
while (__len > 0) 
# 961
{ 
# 962
_DistanceType __half = __len >> 1; 
# 963
_ForwardIterator __middle = __first; 
# 964
std::advance(__middle, __half); 
# 965
if ((*__middle) < __val) 
# 966
{ 
# 967
__first = __middle; 
# 968
++__first; 
# 969
__len = ((__len - __half) - 1); 
# 970
} else { 
# 972
__len = __half; }  
# 973
}  
# 974
return __first; 
# 975
} 
# 980
constexpr int __lg(int __n) 
# 981
{ return ((sizeof(int) * (8)) - (1)) - (__builtin_clz(__n)); } 
# 984
constexpr unsigned __lg(unsigned __n) 
# 985
{ return ((sizeof(int) * (8)) - (1)) - (__builtin_clz(__n)); } 
# 988
constexpr long __lg(long __n) 
# 989
{ return ((sizeof(long) * (8)) - (1)) - (__builtin_clzl(__n)); } 
# 992
constexpr unsigned long __lg(unsigned long __n) 
# 993
{ return ((sizeof(long) * (8)) - (1)) - (__builtin_clzl(__n)); } 
# 996
constexpr long long __lg(long long __n) 
# 997
{ return ((sizeof(long long) * (8)) - (1)) - (__builtin_clzll(__n)); } 
# 1000
constexpr unsigned long long __lg(unsigned long long __n) 
# 1001
{ return ((sizeof(long long) * (8)) - (1)) - (__builtin_clzll(__n)); } 
# 1019
template< class _II1, class _II2> inline bool 
# 1021
equal(_II1 __first1, _II1 __last1, _II2 __first2) 
# 1022
{ 
# 1029
; 
# 1031
return std::__equal_aux(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2)); 
# 1034
} 
# 1051
template< class _IIter1, class _IIter2, class _BinaryPredicate> inline bool 
# 1053
equal(_IIter1 __first1, _IIter1 __last1, _IIter2 
# 1054
__first2, _BinaryPredicate __binary_pred) 
# 1055
{ 
# 1059
; 
# 1061
for (; __first1 != __last1; (++__first1), (++__first2)) { 
# 1062
if (!((bool)__binary_pred(*__first1, *__first2))) { 
# 1063
return false; }  }  
# 1064
return true; 
# 1065
} 
# 1082
template< class _II1, class _II2> inline bool 
# 1084
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1085
__first2, _II2 __last2) 
# 1086
{ 
# 1096
; 
# 1097
; 
# 1099
return std::__lexicographical_compare_aux(std::__niter_base(__first1), std::__niter_base(__last1), std::__niter_base(__first2), std::__niter_base(__last2)); 
# 1103
} 
# 1118
template< class _II1, class _II2, class _Compare> bool 
# 1120
lexicographical_compare(_II1 __first1, _II1 __last1, _II2 
# 1121
__first2, _II2 __last2, _Compare __comp) 
# 1122
{ 
# 1123
typedef typename iterator_traits< _II1> ::iterator_category _Category1; 
# 1124
typedef typename iterator_traits< _II2> ::iterator_category _Category2; 
# 1125
typedef __lc_rai< typename iterator_traits< _II1> ::iterator_category, typename iterator_traits< _II2> ::iterator_category>  __rai_type; 
# 1130
; 
# 1131
; 
# 1133
__last1 = __rai_type::__newlast1(__first1, __last1, __first2, __last2); 
# 1134
for (; (__first1 != __last1) && __rai_type::__cnd2(__first2, __last2); (++__first1), (++__first2)) 
# 1136
{ 
# 1137
if (__comp(*__first1, *__first2)) { 
# 1138
return true; }  
# 1139
if (__comp(*__first2, *__first1)) { 
# 1140
return false; }  
# 1141
}  
# 1142
return (__first1 == __last1) && (__first2 != __last2); 
# 1143
} 
# 1158
template< class _InputIterator1, class _InputIterator2> pair< _InputIterator1, _InputIterator2>  
# 1160
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1161
__first2) 
# 1162
{ 
# 1169
; 
# 1171
while ((__first1 != __last1) && ((*__first1) == (*__first2))) 
# 1172
{ 
# 1173
++__first1; 
# 1174
++__first2; 
# 1175
}  
# 1176
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1177
} 
# 1195
template< class _InputIterator1, class _InputIterator2, class 
# 1196
_BinaryPredicate> pair< _InputIterator1, _InputIterator2>  
# 1198
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 
# 1199
__first2, _BinaryPredicate __binary_pred) 
# 1200
{ 
# 1204
; 
# 1206
while ((__first1 != __last1) && ((bool)__binary_pred(*__first1, *__first2))) 
# 1207
{ 
# 1208
++__first1; 
# 1209
++__first2; 
# 1210
}  
# 1211
return pair< _InputIterator1, _InputIterator2> (__first1, __first2); 
# 1212
} 
# 1215
}
# 43 "/usr/include/c++/4.8/bits/char_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 57
template< class _CharT> 
# 58
struct _Char_types { 
# 60
typedef unsigned long int_type; 
# 61
typedef std::streampos pos_type; 
# 62
typedef std::streamoff off_type; 
# 63
typedef mbstate_t state_type; 
# 64
}; 
# 82
template< class _CharT> 
# 83
struct char_traits { 
# 85
typedef _CharT char_type; 
# 86
typedef typename _Char_types< _CharT> ::int_type int_type; 
# 87
typedef typename _Char_types< _CharT> ::pos_type pos_type; 
# 88
typedef typename _Char_types< _CharT> ::off_type off_type; 
# 89
typedef typename _Char_types< _CharT> ::state_type state_type; 
# 92
static void assign(char_type &__c1, const char_type &__c2) 
# 93
{ __c1 = __c2; } 
# 96
static constexpr bool eq(const char_type &__c1, const char_type &__c2) 
# 97
{ return __c1 == __c2; } 
# 100
static constexpr bool lt(const char_type &__c1, const char_type &__c2) 
# 101
{ return __c1 < __c2; } 
# 104
static int compare(const char_type * __s1, const char_type * __s2, std::size_t __n); 
# 107
static std::size_t length(const char_type * __s); 
# 110
static const char_type *find(const char_type * __s, std::size_t __n, const char_type & __a); 
# 113
static char_type *move(char_type * __s1, const char_type * __s2, std::size_t __n); 
# 116
static char_type *copy(char_type * __s1, const char_type * __s2, std::size_t __n); 
# 119
static char_type *assign(char_type * __s, std::size_t __n, char_type __a); 
# 122
static constexpr char_type to_char_type(const int_type &__c) 
# 123
{ return static_cast< char_type>(__c); } 
# 126
static constexpr int_type to_int_type(const char_type &__c) 
# 127
{ return static_cast< int_type>(__c); } 
# 130
static constexpr bool eq_int_type(const int_type &__c1, const int_type &__c2) 
# 131
{ return __c1 == __c2; } 
# 134
static constexpr int_type eof() 
# 135
{ return static_cast< int_type>(-1); } 
# 138
static constexpr int_type not_eof(const int_type &__c) 
# 139
{ return (!(eq_int_type)(__c, (eof)())) ? __c : (to_int_type)(char_type()); } 
# 140
}; 
# 142
template< class _CharT> int 
# 145
char_traits< _CharT> ::compare(const char_type *__s1, const char_type *__s2, std::size_t __n) 
# 146
{ 
# 147
for (std::size_t __i = (0); __i < __n; ++__i) { 
# 148
if ((lt)(__s1[__i], __s2[__i])) { 
# 149
return -1; } else { 
# 150
if ((lt)(__s2[__i], __s1[__i])) { 
# 151
return 1; }  }  }  
# 152
return 0; 
# 153
} 
# 155
template< class _CharT> std::size_t 
# 158
char_traits< _CharT> ::length(const char_type *__p) 
# 159
{ 
# 160
std::size_t __i = (0); 
# 161
while (!(eq)(__p[__i], char_type())) { 
# 162
++__i; }  
# 163
return __i; 
# 164
} 
# 166
template< class _CharT> const typename char_traits< _CharT> ::char_type *
# 169
char_traits< _CharT> ::find(const char_type *__s, std::size_t __n, const char_type &__a) 
# 170
{ 
# 171
for (std::size_t __i = (0); __i < __n; ++__i) { 
# 172
if ((eq)(__s[__i], __a)) { 
# 173
return __s + __i; }  }  
# 174
return 0; 
# 175
} 
# 177
template< class _CharT> typename char_traits< _CharT> ::char_type *
# 180
char_traits< _CharT> ::move(char_type *__s1, const char_type *__s2, std::size_t __n) 
# 181
{ 
# 182
return static_cast< _CharT *>(__builtin_memmove(__s1, __s2, __n * sizeof(char_type))); 
# 184
} 
# 186
template< class _CharT> typename char_traits< _CharT> ::char_type *
# 189
char_traits< _CharT> ::copy(char_type *__s1, const char_type *__s2, std::size_t __n) 
# 190
{ 
# 192
std::copy(__s2, __s2 + __n, __s1); 
# 193
return __s1; 
# 194
} 
# 196
template< class _CharT> typename char_traits< _CharT> ::char_type *
# 199
char_traits< _CharT> ::assign(char_type *__s, std::size_t __n, char_type __a) 
# 200
{ 
# 202
std::fill_n(__s, __n, __a); 
# 203
return __s; 
# 204
} 
# 207
}
# 209
namespace std __attribute((__visibility__("default"))) { 
# 226
template< class _CharT> 
# 227
struct char_traits : public __gnu_cxx::char_traits< _CharT>  { 
# 228
}; 
# 233
template<> struct char_traits< char>  { 
# 235
typedef char char_type; 
# 236
typedef int int_type; 
# 237
typedef streampos pos_type; 
# 238
typedef streamoff off_type; 
# 239
typedef mbstate_t state_type; 
# 242
static void assign(char_type &__c1, const char_type &__c2) noexcept 
# 243
{ __c1 = __c2; } 
# 246
static constexpr bool eq(const char_type &__c1, const char_type &__c2) noexcept 
# 247
{ return __c1 == __c2; } 
# 250
static constexpr bool lt(const char_type &__c1, const char_type &__c2) noexcept 
# 251
{ return __c1 < __c2; } 
# 254
static int compare(const char_type *__s1, const char_type *__s2, size_t __n) 
# 255
{ return __builtin_memcmp(__s1, __s2, __n); } 
# 258
static size_t length(const char_type *__s) 
# 259
{ return __builtin_strlen(__s); } 
# 262
static const char_type *find(const char_type *__s, size_t __n, const char_type &__a) 
# 263
{ return static_cast< const char_type *>(__builtin_memchr(__s, __a, __n)); } 
# 266
static char_type *move(char_type *__s1, const char_type *__s2, size_t __n) 
# 267
{ return static_cast< char_type *>(__builtin_memmove(__s1, __s2, __n)); } 
# 270
static char_type *copy(char_type *__s1, const char_type *__s2, size_t __n) 
# 271
{ return static_cast< char_type *>(__builtin_memcpy(__s1, __s2, __n)); } 
# 274
static char_type *assign(char_type *__s, size_t __n, char_type __a) 
# 275
{ return static_cast< char_type *>(__builtin_memset(__s, __a, __n)); } 
# 278
static constexpr char_type to_char_type(const int_type &__c) noexcept 
# 279
{ return static_cast< char_type>(__c); } 
# 284
static constexpr int_type to_int_type(const char_type &__c) noexcept 
# 285
{ return static_cast< int_type>(static_cast< unsigned char>(__c)); } 
# 288
static constexpr bool eq_int_type(const int_type &__c1, const int_type &__c2) noexcept 
# 289
{ return __c1 == __c2; } 
# 292
static constexpr int_type eof() noexcept 
# 293
{ return static_cast< int_type>(-1); } 
# 296
static constexpr int_type not_eof(const int_type &__c) noexcept 
# 297
{ return (__c == eof()) ? 0 : __c; } 
# 298
}; 
# 304
template<> struct char_traits< wchar_t>  { 
# 306
typedef wchar_t char_type; 
# 307
typedef wint_t int_type; 
# 308
typedef streamoff off_type; 
# 309
typedef wstreampos pos_type; 
# 310
typedef mbstate_t state_type; 
# 313
static void assign(char_type &__c1, const char_type &__c2) noexcept 
# 314
{ __c1 = __c2; } 
# 317
static constexpr bool eq(const char_type &__c1, const char_type &__c2) noexcept 
# 318
{ return __c1 == __c2; } 
# 321
static constexpr bool lt(const char_type &__c1, const char_type &__c2) noexcept 
# 322
{ return __c1 < __c2; } 
# 325
static int compare(const char_type *__s1, const char_type *__s2, size_t __n) 
# 326
{ return wmemcmp(__s1, __s2, __n); } 
# 329
static size_t length(const char_type *__s) 
# 330
{ return wcslen(__s); } 
# 333
static const char_type *find(const char_type *__s, size_t __n, const char_type &__a) 
# 334
{ return wmemchr(__s, __a, __n); } 
# 337
static char_type *move(char_type *__s1, const char_type *__s2, size_t __n) 
# 338
{ return wmemmove(__s1, __s2, __n); } 
# 341
static char_type *copy(char_type *__s1, const char_type *__s2, size_t __n) 
# 342
{ return wmemcpy(__s1, __s2, __n); } 
# 345
static char_type *assign(char_type *__s, size_t __n, char_type __a) 
# 346
{ return wmemset(__s, __a, __n); } 
# 349
static constexpr char_type to_char_type(const int_type &__c) noexcept 
# 350
{ return (char_type)__c; } 
# 353
static constexpr int_type to_int_type(const char_type &__c) noexcept 
# 354
{ return (int_type)__c; } 
# 357
static constexpr bool eq_int_type(const int_type &__c1, const int_type &__c2) noexcept 
# 358
{ return __c1 == __c2; } 
# 361
static constexpr int_type eof() noexcept 
# 362
{ return static_cast< int_type>(4294967295U); } 
# 365
static constexpr int_type not_eof(const int_type &__c) noexcept 
# 366
{ return eq_int_type(__c, eof()) ? 0 : __c; } 
# 367
}; 
# 371
}
# 48 "/usr/include/stdint.h" 3
typedef unsigned char uint8_t; 
# 49
typedef unsigned short uint16_t; 
# 51
typedef unsigned uint32_t; 
# 55
typedef unsigned long uint64_t; 
# 65
typedef signed char int_least8_t; 
# 66
typedef short int_least16_t; 
# 67
typedef int int_least32_t; 
# 69
typedef long int_least64_t; 
# 76
typedef unsigned char uint_least8_t; 
# 77
typedef unsigned short uint_least16_t; 
# 78
typedef unsigned uint_least32_t; 
# 80
typedef unsigned long uint_least64_t; 
# 90
typedef signed char int_fast8_t; 
# 92
typedef long int_fast16_t; 
# 93
typedef long int_fast32_t; 
# 94
typedef long int_fast64_t; 
# 103
typedef unsigned char uint_fast8_t; 
# 105
typedef unsigned long uint_fast16_t; 
# 106
typedef unsigned long uint_fast32_t; 
# 107
typedef unsigned long uint_fast64_t; 
# 119
typedef long intptr_t; 
# 122
typedef unsigned long uintptr_t; 
# 134
typedef long intmax_t; 
# 135
typedef unsigned long uintmax_t; 
# 46 "/usr/include/c++/4.8/cstdint" 3
namespace std { 
# 48
using ::int8_t;
# 49
using ::int16_t;
# 50
using ::int32_t;
# 51
using ::int64_t;
# 53
using ::int_fast8_t;
# 54
using ::int_fast16_t;
# 55
using ::int_fast32_t;
# 56
using ::int_fast64_t;
# 58
using ::int_least8_t;
# 59
using ::int_least16_t;
# 60
using ::int_least32_t;
# 61
using ::int_least64_t;
# 63
using ::intmax_t;
# 64
using ::intptr_t;
# 66
using ::uint8_t;
# 67
using ::uint16_t;
# 68
using ::uint32_t;
# 69
using ::uint64_t;
# 71
using ::uint_fast8_t;
# 72
using ::uint_fast16_t;
# 73
using ::uint_fast32_t;
# 74
using ::uint_fast64_t;
# 76
using ::uint_least8_t;
# 77
using ::uint_least16_t;
# 78
using ::uint_least32_t;
# 79
using ::uint_least64_t;
# 81
using ::uintmax_t;
# 82
using ::uintptr_t;
# 83
}
# 378 "/usr/include/c++/4.8/bits/char_traits.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 383
template<> struct char_traits< char16_t>  { 
# 385
typedef char16_t char_type; 
# 386
typedef uint_least16_t int_type; 
# 387
typedef streamoff off_type; 
# 388
typedef u16streampos pos_type; 
# 389
typedef mbstate_t state_type; 
# 392
static void assign(char_type &__c1, const char_type &__c2) noexcept 
# 393
{ __c1 = __c2; } 
# 396
static constexpr bool eq(const char_type &__c1, const char_type &__c2) noexcept 
# 397
{ return __c1 == __c2; } 
# 400
static constexpr bool lt(const char_type &__c1, const char_type &__c2) noexcept 
# 401
{ return __c1 < __c2; } 
# 404
static int compare(const char_type *__s1, const char_type *__s2, size_t __n) 
# 405
{ 
# 406
for (size_t __i = (0); __i < __n; ++__i) { 
# 407
if (lt(__s1[__i], __s2[__i])) { 
# 408
return -1; } else { 
# 409
if (lt(__s2[__i], __s1[__i])) { 
# 410
return 1; }  }  }  
# 411
return 0; 
# 412
} 
# 415
static size_t length(const char_type *__s) 
# 416
{ 
# 417
size_t __i = (0); 
# 418
while (!eq(__s[__i], char_type())) { 
# 419
++__i; }  
# 420
return __i; 
# 421
} 
# 424
static const char_type *find(const char_type *__s, size_t __n, const char_type &__a) 
# 425
{ 
# 426
for (size_t __i = (0); __i < __n; ++__i) { 
# 427
if (eq(__s[__i], __a)) { 
# 428
return __s + __i; }  }  
# 429
return 0; 
# 430
} 
# 433
static char_type *move(char_type *__s1, const char_type *__s2, size_t __n) 
# 434
{ 
# 435
return static_cast< char_type *>(__builtin_memmove(__s1, __s2, __n * sizeof(char_type))); 
# 437
} 
# 440
static char_type *copy(char_type *__s1, const char_type *__s2, size_t __n) 
# 441
{ 
# 442
return static_cast< char_type *>(__builtin_memcpy(__s1, __s2, __n * sizeof(char_type))); 
# 444
} 
# 447
static char_type *assign(char_type *__s, size_t __n, char_type __a) 
# 448
{ 
# 449
for (size_t __i = (0); __i < __n; ++__i) { 
# 450
assign(__s[__i], __a); }  
# 451
return __s; 
# 452
} 
# 455
static constexpr char_type to_char_type(const int_type &__c) noexcept 
# 456
{ return (char_type)__c; } 
# 459
static constexpr int_type to_int_type(const char_type &__c) noexcept 
# 460
{ return (int_type)__c; } 
# 463
static constexpr bool eq_int_type(const int_type &__c1, const int_type &__c2) noexcept 
# 464
{ return __c1 == __c2; } 
# 467
static constexpr int_type eof() noexcept 
# 468
{ return static_cast< int_type>(-1); } 
# 471
static constexpr int_type not_eof(const int_type &__c) noexcept 
# 472
{ return eq_int_type(__c, eof()) ? 0 : (__c); } 
# 473
}; 
# 476
template<> struct char_traits< char32_t>  { 
# 478
typedef char32_t char_type; 
# 479
typedef uint_least32_t int_type; 
# 480
typedef streamoff off_type; 
# 481
typedef u32streampos pos_type; 
# 482
typedef mbstate_t state_type; 
# 485
static void assign(char_type &__c1, const char_type &__c2) noexcept 
# 486
{ __c1 = __c2; } 
# 489
static constexpr bool eq(const char_type &__c1, const char_type &__c2) noexcept 
# 490
{ return __c1 == __c2; } 
# 493
static constexpr bool lt(const char_type &__c1, const char_type &__c2) noexcept 
# 494
{ return __c1 < __c2; } 
# 497
static int compare(const char_type *__s1, const char_type *__s2, size_t __n) 
# 498
{ 
# 499
for (size_t __i = (0); __i < __n; ++__i) { 
# 500
if (lt(__s1[__i], __s2[__i])) { 
# 501
return -1; } else { 
# 502
if (lt(__s2[__i], __s1[__i])) { 
# 503
return 1; }  }  }  
# 504
return 0; 
# 505
} 
# 508
static size_t length(const char_type *__s) 
# 509
{ 
# 510
size_t __i = (0); 
# 511
while (!eq(__s[__i], char_type())) { 
# 512
++__i; }  
# 513
return __i; 
# 514
} 
# 517
static const char_type *find(const char_type *__s, size_t __n, const char_type &__a) 
# 518
{ 
# 519
for (size_t __i = (0); __i < __n; ++__i) { 
# 520
if (eq(__s[__i], __a)) { 
# 521
return __s + __i; }  }  
# 522
return 0; 
# 523
} 
# 526
static char_type *move(char_type *__s1, const char_type *__s2, size_t __n) 
# 527
{ 
# 528
return static_cast< char_type *>(__builtin_memmove(__s1, __s2, __n * sizeof(char_type))); 
# 530
} 
# 533
static char_type *copy(char_type *__s1, const char_type *__s2, size_t __n) 
# 534
{ 
# 535
return static_cast< char_type *>(__builtin_memcpy(__s1, __s2, __n * sizeof(char_type))); 
# 537
} 
# 540
static char_type *assign(char_type *__s, size_t __n, char_type __a) 
# 541
{ 
# 542
for (size_t __i = (0); __i < __n; ++__i) { 
# 543
assign(__s[__i], __a); }  
# 544
return __s; 
# 545
} 
# 548
static constexpr char_type to_char_type(const int_type &__c) noexcept 
# 549
{ return (char_type)__c; } 
# 552
static constexpr int_type to_int_type(const char_type &__c) noexcept 
# 553
{ return (int_type)__c; } 
# 556
static constexpr bool eq_int_type(const int_type &__c1, const int_type &__c2) noexcept 
# 557
{ return __c1 == __c2; } 
# 560
static constexpr int_type eof() noexcept 
# 561
{ return static_cast< int_type>(-1); } 
# 564
static constexpr int_type not_eof(const int_type &__c) noexcept 
# 565
{ return eq_int_type(__c, eof()) ? 0 : __c; } 
# 566
}; 
# 569
}
# 31 "/usr/include/locale.h" 3
extern "C" {
# 53
struct lconv { 
# 57
char *decimal_point; 
# 58
char *thousands_sep; 
# 64
char *grouping; 
# 70
char *int_curr_symbol; 
# 71
char *currency_symbol; 
# 72
char *mon_decimal_point; 
# 73
char *mon_thousands_sep; 
# 74
char *mon_grouping; 
# 75
char *positive_sign; 
# 76
char *negative_sign; 
# 77
char int_frac_digits; 
# 78
char frac_digits; 
# 80
char p_cs_precedes; 
# 82
char p_sep_by_space; 
# 84
char n_cs_precedes; 
# 86
char n_sep_by_space; 
# 93
char p_sign_posn; 
# 94
char n_sign_posn; 
# 97
char int_p_cs_precedes; 
# 99
char int_p_sep_by_space; 
# 101
char int_n_cs_precedes; 
# 103
char int_n_sep_by_space; 
# 110
char int_p_sign_posn; 
# 111
char int_n_sign_posn; 
# 120
}; 
# 124
extern char *setlocale(int __category, const char * __locale) throw(); 
# 127
extern lconv *localeconv() throw(); 
# 151
extern __locale_t newlocale(int __category_mask, const char * __locale, __locale_t __base) throw(); 
# 186
extern __locale_t duplocale(__locale_t __dataset) throw(); 
# 190
extern void freelocale(__locale_t __dataset) throw(); 
# 197
extern __locale_t uselocale(__locale_t __dataset) throw(); 
# 205
}
# 51 "/usr/include/c++/4.8/clocale" 3
namespace std { 
# 53
using ::lconv;
# 54
using ::setlocale;
# 55
using ::localeconv;
# 56
}
# 48 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/c++locale.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 52
extern "C" __typeof__(uselocale) __uselocale; 
# 55
}
# 58
namespace std __attribute((__visibility__("default"))) { 
# 62
typedef __locale_t __c_locale; 
# 69
inline int __convert_from_v(const __c_locale &__cloc __attribute((__unused__)), char *
# 70
__out, const int 
# 71
__size __attribute((__unused__)), const char *
# 72
__fmt, ...) 
# 73
{ 
# 75
__c_locale __old = __gnu_cxx::__uselocale(__cloc); 
# 88
__builtin_va_list __args; 
# 89
__builtin_va_start((__args),__fmt); 
# 92
const int __ret = __builtin_vsnprintf(__out, __size, __fmt, __args); 
# 97
__builtin_va_end(__args); 
# 100
__gnu_cxx::__uselocale(__old); 
# 108
return __ret; 
# 109
} 
# 112
}
# 28 "/usr/include/ctype.h" 3
extern "C" {
# 47
enum { 
# 48
_ISupper = 256, 
# 49
_ISlower = 512, 
# 50
_ISalpha = 1024, 
# 51
_ISdigit = 2048, 
# 52
_ISxdigit = 4096, 
# 53
_ISspace = 8192, 
# 54
_ISprint = 16384, 
# 55
_ISgraph = 32768, 
# 56
_ISblank = 1, 
# 57
_IScntrl, 
# 58
_ISpunct = 4, 
# 59
_ISalnum = 8
# 60
}; 
# 79
extern const unsigned short **__ctype_b_loc() throw()
# 80
 __attribute((const)); 
# 81
extern const __int32_t **__ctype_tolower_loc() throw()
# 82
 __attribute((const)); 
# 83
extern const __int32_t **__ctype_toupper_loc() throw()
# 84
 __attribute((const)); 
# 110
extern int isalnum(int) throw(); 
# 111
extern int isalpha(int) throw(); 
# 112
extern int iscntrl(int) throw(); 
# 113
extern int isdigit(int) throw(); 
# 114
extern int islower(int) throw(); 
# 115
extern int isgraph(int) throw(); 
# 116
extern int isprint(int) throw(); 
# 117
extern int ispunct(int) throw(); 
# 118
extern int isspace(int) throw(); 
# 119
extern int isupper(int) throw(); 
# 120
extern int isxdigit(int) throw(); 
# 124
extern int tolower(int __c) throw(); 
# 127
extern int toupper(int __c) throw(); 
# 136
extern int isblank(int) throw(); 
# 143
extern int isctype(int __c, int __mask) throw(); 
# 150
extern int isascii(int __c) throw(); 
# 154
extern int toascii(int __c) throw(); 
# 158
extern int _toupper(int) throw(); 
# 159
extern int _tolower(int) throw(); 
# 271
extern int isalnum_l(int, __locale_t) throw(); 
# 272
extern int isalpha_l(int, __locale_t) throw(); 
# 273
extern int iscntrl_l(int, __locale_t) throw(); 
# 274
extern int isdigit_l(int, __locale_t) throw(); 
# 275
extern int islower_l(int, __locale_t) throw(); 
# 276
extern int isgraph_l(int, __locale_t) throw(); 
# 277
extern int isprint_l(int, __locale_t) throw(); 
# 278
extern int ispunct_l(int, __locale_t) throw(); 
# 279
extern int isspace_l(int, __locale_t) throw(); 
# 280
extern int isupper_l(int, __locale_t) throw(); 
# 281
extern int isxdigit_l(int, __locale_t) throw(); 
# 283
extern int isblank_l(int, __locale_t) throw(); 
# 287
extern int __tolower_l(int __c, __locale_t __l) throw(); 
# 288
extern int tolower_l(int __c, __locale_t __l) throw(); 
# 291
extern int __toupper_l(int __c, __locale_t __l) throw(); 
# 292
extern int toupper_l(int __c, __locale_t __l) throw(); 
# 347
}
# 62 "/usr/include/c++/4.8/cctype" 3
namespace std { 
# 64
using ::isalnum;
# 65
using ::isalpha;
# 66
using ::iscntrl;
# 67
using ::isdigit;
# 68
using ::isgraph;
# 69
using ::islower;
# 70
using ::isprint;
# 71
using ::ispunct;
# 72
using ::isspace;
# 73
using ::isupper;
# 74
using ::isxdigit;
# 75
using ::tolower;
# 76
using ::toupper;
# 77
}
# 85
namespace std { 
# 87
using ::isblank;
# 88
}
# 44 "/usr/include/c++/4.8/bits/localefwd.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 55
class locale; 
# 57
template< class _Facet> bool has_facet(const locale &) throw(); 
# 61
template< class _Facet> const _Facet &use_facet(const locale &); 
# 66
template< class _CharT> inline bool isspace(_CharT, const locale &); 
# 70
template< class _CharT> inline bool isprint(_CharT, const locale &); 
# 74
template< class _CharT> inline bool iscntrl(_CharT, const locale &); 
# 78
template< class _CharT> inline bool isupper(_CharT, const locale &); 
# 82
template< class _CharT> inline bool islower(_CharT, const locale &); 
# 86
template< class _CharT> inline bool isalpha(_CharT, const locale &); 
# 90
template< class _CharT> inline bool isdigit(_CharT, const locale &); 
# 94
template< class _CharT> inline bool ispunct(_CharT, const locale &); 
# 98
template< class _CharT> inline bool isxdigit(_CharT, const locale &); 
# 102
template< class _CharT> inline bool isalnum(_CharT, const locale &); 
# 106
template< class _CharT> inline bool isgraph(_CharT, const locale &); 
# 110
template< class _CharT> inline _CharT toupper(_CharT, const locale &); 
# 114
template< class _CharT> inline _CharT tolower(_CharT, const locale &); 
# 119
struct ctype_base; 
# 120
template< class _CharT> class ctype; 
# 122
template<> class ctype< char> ; 
# 124
template<> class ctype< wchar_t> ; 
# 126
template< class _CharT> class ctype_byname; 
# 130
class codecvt_base; 
# 131
template< class _InternT, class _ExternT, class _StateT> class codecvt; 
# 133
template<> class codecvt< char, char, __mbstate_t> ; 
# 135
template<> class codecvt< wchar_t, char, __mbstate_t> ; 
# 137
template< class _InternT, class _ExternT, class _StateT> class codecvt_byname; 
# 142
template< class _CharT, class _InIter = istreambuf_iterator< _CharT, char_traits< _CharT> > > class num_get; 
# 144
template< class _CharT, class _OutIter = ostreambuf_iterator< _CharT, char_traits< _CharT> > > class num_put; 
# 147
template< class _CharT> class numpunct; 
# 148
template< class _CharT> class numpunct_byname; 
# 151
template< class _CharT> class collate; 
# 153
template< class _CharT> class collate_byname; 
# 157
class time_base; 
# 158
template< class _CharT, class _InIter = istreambuf_iterator< _CharT, char_traits< _CharT> > > class time_get; 
# 160
template< class _CharT, class _InIter = istreambuf_iterator< _CharT, char_traits< _CharT> > > class time_get_byname; 
# 162
template< class _CharT, class _OutIter = ostreambuf_iterator< _CharT, char_traits< _CharT> > > class time_put; 
# 164
template< class _CharT, class _OutIter = ostreambuf_iterator< _CharT, char_traits< _CharT> > > class time_put_byname; 
# 168
class money_base; 
# 170
template< class _CharT, class _InIter = istreambuf_iterator< _CharT, char_traits< _CharT> > > class money_get; 
# 172
template< class _CharT, class _OutIter = ostreambuf_iterator< _CharT, char_traits< _CharT> > > class money_put; 
# 175
template< class _CharT, bool _Intl = false> class moneypunct; 
# 177
template< class _CharT, bool _Intl = false> class moneypunct_byname; 
# 181
class messages_base; 
# 182
template< class _CharT> class messages; 
# 184
template< class _CharT> class messages_byname; 
# 188
}
# 30 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/gthr.h" 3
#pragma GCC visibility push ( default )
# 72 "/usr/include/x86_64-linux-gnu/bits/sched.h" 3
struct sched_param { 
# 74
int __sched_priority; 
# 75
}; 
# 77
extern "C" {
# 81
extern int clone(int (* __fn)(void * __arg), void * __child_stack, int __flags, void * __arg, ...) throw(); 
# 85
extern int unshare(int __flags) throw(); 
# 88
extern int sched_getcpu() throw(); 
# 91
extern int setns(int __fd, int __nstype) throw(); 
# 95
}
# 103
struct __sched_param { 
# 105
int __sched_priority; 
# 106
}; 
# 118
typedef unsigned long __cpu_mask; 
# 128
typedef 
# 126
struct { 
# 127
__cpu_mask __bits[(1024) / ((8) * sizeof(__cpu_mask))]; 
# 128
} cpu_set_t; 
# 201
extern "C" {
# 203
extern int __sched_cpucount(size_t __setsize, const cpu_set_t * __setp) throw(); 
# 205
extern cpu_set_t *__sched_cpualloc(size_t __count) throw(); 
# 206
extern void __sched_cpufree(cpu_set_t * __set) throw(); 
# 208
}
# 48 "/usr/include/sched.h" 3
extern "C" {
# 51
extern int sched_setparam(__pid_t __pid, const sched_param * __param) throw(); 
# 55
extern int sched_getparam(__pid_t __pid, sched_param * __param) throw(); 
# 58
extern int sched_setscheduler(__pid_t __pid, int __policy, const sched_param * __param) throw(); 
# 62
extern int sched_getscheduler(__pid_t __pid) throw(); 
# 65
extern int sched_yield() throw(); 
# 68
extern int sched_get_priority_max(int __algorithm) throw(); 
# 71
extern int sched_get_priority_min(int __algorithm) throw(); 
# 74
extern int sched_rr_get_interval(__pid_t __pid, timespec * __t) throw(); 
# 118
extern int sched_setaffinity(__pid_t __pid, size_t __cpusetsize, const cpu_set_t * __cpuset) throw(); 
# 122
extern int sched_getaffinity(__pid_t __pid, size_t __cpusetsize, cpu_set_t * __cpuset) throw(); 
# 126
}
# 31 "/usr/include/x86_64-linux-gnu/bits/setjmp.h" 3
typedef long __jmp_buf[8]; 
# 33 "/usr/include/pthread.h" 3
enum { 
# 34
PTHREAD_CREATE_JOINABLE, 
# 36
PTHREAD_CREATE_DETACHED
# 38
}; 
# 43
enum { 
# 44
PTHREAD_MUTEX_TIMED_NP, 
# 45
PTHREAD_MUTEX_RECURSIVE_NP, 
# 46
PTHREAD_MUTEX_ERRORCHECK_NP, 
# 47
PTHREAD_MUTEX_ADAPTIVE_NP, 
# 50
PTHREAD_MUTEX_NORMAL = 0, 
# 51
PTHREAD_MUTEX_RECURSIVE, 
# 52
PTHREAD_MUTEX_ERRORCHECK, 
# 53
PTHREAD_MUTEX_DEFAULT = 0, 
# 57
PTHREAD_MUTEX_FAST_NP = 0
# 59
}; 
# 65
enum { 
# 66
PTHREAD_MUTEX_STALLED, 
# 67
PTHREAD_MUTEX_STALLED_NP = 0, 
# 68
PTHREAD_MUTEX_ROBUST, 
# 69
PTHREAD_MUTEX_ROBUST_NP = 1
# 70
}; 
# 77
enum { 
# 78
PTHREAD_PRIO_NONE, 
# 79
PTHREAD_PRIO_INHERIT, 
# 80
PTHREAD_PRIO_PROTECT
# 81
}; 
# 115
enum { 
# 116
PTHREAD_RWLOCK_PREFER_READER_NP, 
# 117
PTHREAD_RWLOCK_PREFER_WRITER_NP, 
# 118
PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP, 
# 119
PTHREAD_RWLOCK_DEFAULT_NP = 0
# 120
}; 
# 156
enum { 
# 157
PTHREAD_INHERIT_SCHED, 
# 159
PTHREAD_EXPLICIT_SCHED
# 161
}; 
# 166
enum { 
# 167
PTHREAD_SCOPE_SYSTEM, 
# 169
PTHREAD_SCOPE_PROCESS
# 171
}; 
# 176
enum { 
# 177
PTHREAD_PROCESS_PRIVATE, 
# 179
PTHREAD_PROCESS_SHARED
# 181
}; 
# 190
struct _pthread_cleanup_buffer { 
# 192
void (*__routine)(void *); 
# 193
void *__arg; 
# 194
int __canceltype; 
# 195
_pthread_cleanup_buffer *__prev; 
# 196
}; 
# 200
enum { 
# 201
PTHREAD_CANCEL_ENABLE, 
# 203
PTHREAD_CANCEL_DISABLE
# 205
}; 
# 207
enum { 
# 208
PTHREAD_CANCEL_DEFERRED, 
# 210
PTHREAD_CANCEL_ASYNCHRONOUS
# 212
}; 
# 228
extern "C" {
# 233
extern int pthread_create(pthread_t *__restrict__ __newthread, const pthread_attr_t *__restrict__ __attr, void *(* __start_routine)(void *), void *__restrict__ __arg) throw()
# 236
 __attribute((__nonnull__(1, 3))); 
# 242
extern void pthread_exit(void * __retval) __attribute((__noreturn__)); 
# 250
extern int pthread_join(pthread_t __th, void ** __thread_return); 
# 255
extern int pthread_tryjoin_np(pthread_t __th, void ** __thread_return) throw(); 
# 263
extern int pthread_timedjoin_np(pthread_t __th, void ** __thread_return, const timespec * __abstime); 
# 271
extern int pthread_detach(pthread_t __th) throw(); 
# 275
extern pthread_t pthread_self() throw() __attribute((const)); 
# 278
extern int pthread_equal(pthread_t __thread1, pthread_t __thread2) throw()
# 279
 __attribute((const)); 
# 287
extern int pthread_attr_init(pthread_attr_t * __attr) throw() __attribute((__nonnull__(1))); 
# 290
extern int pthread_attr_destroy(pthread_attr_t * __attr) throw()
# 291
 __attribute((__nonnull__(1))); 
# 294
extern int pthread_attr_getdetachstate(const pthread_attr_t * __attr, int * __detachstate) throw()
# 296
 __attribute((__nonnull__(1, 2))); 
# 299
extern int pthread_attr_setdetachstate(pthread_attr_t * __attr, int __detachstate) throw()
# 301
 __attribute((__nonnull__(1))); 
# 305
extern int pthread_attr_getguardsize(const pthread_attr_t * __attr, size_t * __guardsize) throw()
# 307
 __attribute((__nonnull__(1, 2))); 
# 310
extern int pthread_attr_setguardsize(pthread_attr_t * __attr, size_t __guardsize) throw()
# 312
 __attribute((__nonnull__(1))); 
# 316
extern int pthread_attr_getschedparam(const pthread_attr_t *__restrict__ __attr, sched_param *__restrict__ __param) throw()
# 318
 __attribute((__nonnull__(1, 2))); 
# 321
extern int pthread_attr_setschedparam(pthread_attr_t *__restrict__ __attr, const sched_param *__restrict__ __param) throw()
# 323
 __attribute((__nonnull__(1, 2))); 
# 326
extern int pthread_attr_getschedpolicy(const pthread_attr_t *__restrict__ __attr, int *__restrict__ __policy) throw()
# 328
 __attribute((__nonnull__(1, 2))); 
# 331
extern int pthread_attr_setschedpolicy(pthread_attr_t * __attr, int __policy) throw()
# 332
 __attribute((__nonnull__(1))); 
# 335
extern int pthread_attr_getinheritsched(const pthread_attr_t *__restrict__ __attr, int *__restrict__ __inherit) throw()
# 337
 __attribute((__nonnull__(1, 2))); 
# 340
extern int pthread_attr_setinheritsched(pthread_attr_t * __attr, int __inherit) throw()
# 342
 __attribute((__nonnull__(1))); 
# 346
extern int pthread_attr_getscope(const pthread_attr_t *__restrict__ __attr, int *__restrict__ __scope) throw()
# 348
 __attribute((__nonnull__(1, 2))); 
# 351
extern int pthread_attr_setscope(pthread_attr_t * __attr, int __scope) throw()
# 352
 __attribute((__nonnull__(1))); 
# 355
extern int pthread_attr_getstackaddr(const pthread_attr_t *__restrict__ __attr, void **__restrict__ __stackaddr) throw()
# 357
 __attribute((__nonnull__(1, 2))) __attribute((__deprecated__)); 
# 363
extern int pthread_attr_setstackaddr(pthread_attr_t * __attr, void * __stackaddr) throw()
# 365
 __attribute((__nonnull__(1))) __attribute((__deprecated__)); 
# 368
extern int pthread_attr_getstacksize(const pthread_attr_t *__restrict__ __attr, size_t *__restrict__ __stacksize) throw()
# 370
 __attribute((__nonnull__(1, 2))); 
# 375
extern int pthread_attr_setstacksize(pthread_attr_t * __attr, size_t __stacksize) throw()
# 377
 __attribute((__nonnull__(1))); 
# 381
extern int pthread_attr_getstack(const pthread_attr_t *__restrict__ __attr, void **__restrict__ __stackaddr, size_t *__restrict__ __stacksize) throw()
# 384
 __attribute((__nonnull__(1, 2, 3))); 
# 389
extern int pthread_attr_setstack(pthread_attr_t * __attr, void * __stackaddr, size_t __stacksize) throw()
# 390
 __attribute((__nonnull__(1))); 
# 396
extern int pthread_attr_setaffinity_np(pthread_attr_t * __attr, size_t __cpusetsize, const cpu_set_t * __cpuset) throw()
# 399
 __attribute((__nonnull__(1, 3))); 
# 403
extern int pthread_attr_getaffinity_np(const pthread_attr_t * __attr, size_t __cpusetsize, cpu_set_t * __cpuset) throw()
# 406
 __attribute((__nonnull__(1, 3))); 
# 409
extern int pthread_getattr_default_np(pthread_attr_t * __attr) throw()
# 410
 __attribute((__nonnull__(1))); 
# 414
extern int pthread_setattr_default_np(const pthread_attr_t * __attr) throw()
# 415
 __attribute((__nonnull__(1))); 
# 420
extern int pthread_getattr_np(pthread_t __th, pthread_attr_t * __attr) throw()
# 421
 __attribute((__nonnull__(2))); 
# 429
extern int pthread_setschedparam(pthread_t __target_thread, int __policy, const sched_param * __param) throw()
# 431
 __attribute((__nonnull__(3))); 
# 434
extern int pthread_getschedparam(pthread_t __target_thread, int *__restrict__ __policy, sched_param *__restrict__ __param) throw()
# 437
 __attribute((__nonnull__(2, 3))); 
# 440
extern int pthread_setschedprio(pthread_t __target_thread, int __prio) throw(); 
# 446
extern int pthread_getname_np(pthread_t __target_thread, char * __buf, size_t __buflen) throw()
# 448
 __attribute((__nonnull__(2))); 
# 451
extern int pthread_setname_np(pthread_t __target_thread, const char * __name) throw()
# 452
 __attribute((__nonnull__(2))); 
# 458
extern int pthread_getconcurrency() throw(); 
# 461
extern int pthread_setconcurrency(int __level) throw(); 
# 469
extern int pthread_yield() throw(); 
# 474
extern int pthread_setaffinity_np(pthread_t __th, size_t __cpusetsize, const cpu_set_t * __cpuset) throw()
# 476
 __attribute((__nonnull__(3))); 
# 479
extern int pthread_getaffinity_np(pthread_t __th, size_t __cpusetsize, cpu_set_t * __cpuset) throw()
# 481
 __attribute((__nonnull__(3))); 
# 494
extern int pthread_once(pthread_once_t * __once_control, void (* __init_routine)(void))
# 495
 __attribute((__nonnull__(1, 2))); 
# 506
extern int pthread_setcancelstate(int __state, int * __oldstate); 
# 510
extern int pthread_setcanceltype(int __type, int * __oldtype); 
# 513
extern int pthread_cancel(pthread_t __th); 
# 518
extern void pthread_testcancel(); 
# 531
typedef 
# 524
struct { 
# 526
struct { 
# 527
__jmp_buf __cancel_jmp_buf; 
# 528
int __mask_was_saved; 
# 529
} __cancel_jmp_buf[1]; 
# 530
void *__pad[4]; 
# 531
} __pthread_unwind_buf_t __attribute((__aligned__)); 
# 540
struct __pthread_cleanup_frame { 
# 542
void (*__cancel_routine)(void *); 
# 543
void *__cancel_arg; 
# 544
int __do_it; 
# 545
int __cancel_type; 
# 546
}; 
# 551
class __pthread_cleanup_class { 
# 553
void (*__cancel_routine)(void *); 
# 554
void *__cancel_arg; 
# 555
int __do_it; 
# 556
int __cancel_type; 
# 559
public: __pthread_cleanup_class(void (*__fct)(void *), void *__arg) : __cancel_routine(__fct), __cancel_arg(__arg), __do_it(1) 
# 560
{ } 
# 561
~__pthread_cleanup_class() { if (__do_it) { (__cancel_routine)(__cancel_arg); }  } 
# 562
void __setdoit(int __newval) { (__do_it) = __newval; } 
# 563
void __defer() { pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, &(__cancel_type)); 
# 564
} 
# 565
void __restore() const { pthread_setcanceltype(__cancel_type, 0); } 
# 566
}; 
# 742
struct __jmp_buf_tag; 
# 743
extern int __sigsetjmp(__jmp_buf_tag * __env, int __savemask) throw(); 
# 749
extern int pthread_mutex_init(pthread_mutex_t * __mutex, const pthread_mutexattr_t * __mutexattr) throw()
# 751
 __attribute((__nonnull__(1))); 
# 754
extern int pthread_mutex_destroy(pthread_mutex_t * __mutex) throw()
# 755
 __attribute((__nonnull__(1))); 
# 758
extern int pthread_mutex_trylock(pthread_mutex_t * __mutex) throw()
# 759
 __attribute((__nonnull__(1))); 
# 762
extern int pthread_mutex_lock(pthread_mutex_t * __mutex) throw()
# 763
 __attribute((__nonnull__(1))); 
# 767
extern int pthread_mutex_timedlock(pthread_mutex_t *__restrict__ __mutex, const timespec *__restrict__ __abstime) throw()
# 769
 __attribute((__nonnull__(1, 2))); 
# 773
extern int pthread_mutex_unlock(pthread_mutex_t * __mutex) throw()
# 774
 __attribute((__nonnull__(1))); 
# 778
extern int pthread_mutex_getprioceiling(const pthread_mutex_t *__restrict__ __mutex, int *__restrict__ __prioceiling) throw()
# 781
 __attribute((__nonnull__(1, 2))); 
# 785
extern int pthread_mutex_setprioceiling(pthread_mutex_t *__restrict__ __mutex, int __prioceiling, int *__restrict__ __old_ceiling) throw()
# 788
 __attribute((__nonnull__(1, 3))); 
# 793
extern int pthread_mutex_consistent(pthread_mutex_t * __mutex) throw()
# 794
 __attribute((__nonnull__(1))); 
# 796
extern int pthread_mutex_consistent_np(pthread_mutex_t * __mutex) throw()
# 797
 __attribute((__nonnull__(1))); 
# 806
extern int pthread_mutexattr_init(pthread_mutexattr_t * __attr) throw()
# 807
 __attribute((__nonnull__(1))); 
# 810
extern int pthread_mutexattr_destroy(pthread_mutexattr_t * __attr) throw()
# 811
 __attribute((__nonnull__(1))); 
# 814
extern int pthread_mutexattr_getpshared(const pthread_mutexattr_t *__restrict__ __attr, int *__restrict__ __pshared) throw()
# 817
 __attribute((__nonnull__(1, 2))); 
# 820
extern int pthread_mutexattr_setpshared(pthread_mutexattr_t * __attr, int __pshared) throw()
# 822
 __attribute((__nonnull__(1))); 
# 826
extern int pthread_mutexattr_gettype(const pthread_mutexattr_t *__restrict__ __attr, int *__restrict__ __kind) throw()
# 828
 __attribute((__nonnull__(1, 2))); 
# 833
extern int pthread_mutexattr_settype(pthread_mutexattr_t * __attr, int __kind) throw()
# 834
 __attribute((__nonnull__(1))); 
# 838
extern int pthread_mutexattr_getprotocol(const pthread_mutexattr_t *__restrict__ __attr, int *__restrict__ __protocol) throw()
# 841
 __attribute((__nonnull__(1, 2))); 
# 845
extern int pthread_mutexattr_setprotocol(pthread_mutexattr_t * __attr, int __protocol) throw()
# 847
 __attribute((__nonnull__(1))); 
# 850
extern int pthread_mutexattr_getprioceiling(const pthread_mutexattr_t *__restrict__ __attr, int *__restrict__ __prioceiling) throw()
# 853
 __attribute((__nonnull__(1, 2))); 
# 856
extern int pthread_mutexattr_setprioceiling(pthread_mutexattr_t * __attr, int __prioceiling) throw()
# 858
 __attribute((__nonnull__(1))); 
# 862
extern int pthread_mutexattr_getrobust(const pthread_mutexattr_t * __attr, int * __robustness) throw()
# 864
 __attribute((__nonnull__(1, 2))); 
# 866
extern int pthread_mutexattr_getrobust_np(const pthread_mutexattr_t * __attr, int * __robustness) throw()
# 868
 __attribute((__nonnull__(1, 2))); 
# 872
extern int pthread_mutexattr_setrobust(pthread_mutexattr_t * __attr, int __robustness) throw()
# 874
 __attribute((__nonnull__(1))); 
# 876
extern int pthread_mutexattr_setrobust_np(pthread_mutexattr_t * __attr, int __robustness) throw()
# 878
 __attribute((__nonnull__(1))); 
# 888
extern int pthread_rwlock_init(pthread_rwlock_t *__restrict__ __rwlock, const pthread_rwlockattr_t *__restrict__ __attr) throw()
# 890
 __attribute((__nonnull__(1))); 
# 893
extern int pthread_rwlock_destroy(pthread_rwlock_t * __rwlock) throw()
# 894
 __attribute((__nonnull__(1))); 
# 897
extern int pthread_rwlock_rdlock(pthread_rwlock_t * __rwlock) throw()
# 898
 __attribute((__nonnull__(1))); 
# 901
extern int pthread_rwlock_tryrdlock(pthread_rwlock_t * __rwlock) throw()
# 902
 __attribute((__nonnull__(1))); 
# 906
extern int pthread_rwlock_timedrdlock(pthread_rwlock_t *__restrict__ __rwlock, const timespec *__restrict__ __abstime) throw()
# 908
 __attribute((__nonnull__(1, 2))); 
# 912
extern int pthread_rwlock_wrlock(pthread_rwlock_t * __rwlock) throw()
# 913
 __attribute((__nonnull__(1))); 
# 916
extern int pthread_rwlock_trywrlock(pthread_rwlock_t * __rwlock) throw()
# 917
 __attribute((__nonnull__(1))); 
# 921
extern int pthread_rwlock_timedwrlock(pthread_rwlock_t *__restrict__ __rwlock, const timespec *__restrict__ __abstime) throw()
# 923
 __attribute((__nonnull__(1, 2))); 
# 927
extern int pthread_rwlock_unlock(pthread_rwlock_t * __rwlock) throw()
# 928
 __attribute((__nonnull__(1))); 
# 934
extern int pthread_rwlockattr_init(pthread_rwlockattr_t * __attr) throw()
# 935
 __attribute((__nonnull__(1))); 
# 938
extern int pthread_rwlockattr_destroy(pthread_rwlockattr_t * __attr) throw()
# 939
 __attribute((__nonnull__(1))); 
# 942
extern int pthread_rwlockattr_getpshared(const pthread_rwlockattr_t *__restrict__ __attr, int *__restrict__ __pshared) throw()
# 945
 __attribute((__nonnull__(1, 2))); 
# 948
extern int pthread_rwlockattr_setpshared(pthread_rwlockattr_t * __attr, int __pshared) throw()
# 950
 __attribute((__nonnull__(1))); 
# 953
extern int pthread_rwlockattr_getkind_np(const pthread_rwlockattr_t *__restrict__ __attr, int *__restrict__ __pref) throw()
# 956
 __attribute((__nonnull__(1, 2))); 
# 959
extern int pthread_rwlockattr_setkind_np(pthread_rwlockattr_t * __attr, int __pref) throw()
# 960
 __attribute((__nonnull__(1))); 
# 968
extern int pthread_cond_init(pthread_cond_t *__restrict__ __cond, const pthread_condattr_t *__restrict__ __cond_attr) throw()
# 970
 __attribute((__nonnull__(1))); 
# 973
extern int pthread_cond_destroy(pthread_cond_t * __cond) throw()
# 974
 __attribute((__nonnull__(1))); 
# 977
extern int pthread_cond_signal(pthread_cond_t * __cond) throw()
# 978
 __attribute((__nonnull__(1))); 
# 981
extern int pthread_cond_broadcast(pthread_cond_t * __cond) throw()
# 982
 __attribute((__nonnull__(1))); 
# 989
extern int pthread_cond_wait(pthread_cond_t *__restrict__ __cond, pthread_mutex_t *__restrict__ __mutex)
# 991
 __attribute((__nonnull__(1, 2))); 
# 1000
extern int pthread_cond_timedwait(pthread_cond_t *__restrict__ __cond, pthread_mutex_t *__restrict__ __mutex, const timespec *__restrict__ __abstime)
# 1003
 __attribute((__nonnull__(1, 2, 3))); 
# 1008
extern int pthread_condattr_init(pthread_condattr_t * __attr) throw()
# 1009
 __attribute((__nonnull__(1))); 
# 1012
extern int pthread_condattr_destroy(pthread_condattr_t * __attr) throw()
# 1013
 __attribute((__nonnull__(1))); 
# 1016
extern int pthread_condattr_getpshared(const pthread_condattr_t *__restrict__ __attr, int *__restrict__ __pshared) throw()
# 1019
 __attribute((__nonnull__(1, 2))); 
# 1022
extern int pthread_condattr_setpshared(pthread_condattr_t * __attr, int __pshared) throw()
# 1023
 __attribute((__nonnull__(1))); 
# 1027
extern int pthread_condattr_getclock(const pthread_condattr_t *__restrict__ __attr, __clockid_t *__restrict__ __clock_id) throw()
# 1030
 __attribute((__nonnull__(1, 2))); 
# 1033
extern int pthread_condattr_setclock(pthread_condattr_t * __attr, __clockid_t __clock_id) throw()
# 1035
 __attribute((__nonnull__(1))); 
# 1044
extern int pthread_spin_init(pthread_spinlock_t * __lock, int __pshared) throw()
# 1045
 __attribute((__nonnull__(1))); 
# 1048
extern int pthread_spin_destroy(pthread_spinlock_t * __lock) throw()
# 1049
 __attribute((__nonnull__(1))); 
# 1052
extern int pthread_spin_lock(pthread_spinlock_t * __lock) throw()
# 1053
 __attribute((__nonnull__(1))); 
# 1056
extern int pthread_spin_trylock(pthread_spinlock_t * __lock) throw()
# 1057
 __attribute((__nonnull__(1))); 
# 1060
extern int pthread_spin_unlock(pthread_spinlock_t * __lock) throw()
# 1061
 __attribute((__nonnull__(1))); 
# 1068
extern int pthread_barrier_init(pthread_barrier_t *__restrict__ __barrier, const pthread_barrierattr_t *__restrict__ __attr, unsigned __count) throw()
# 1071
 __attribute((__nonnull__(1))); 
# 1074
extern int pthread_barrier_destroy(pthread_barrier_t * __barrier) throw()
# 1075
 __attribute((__nonnull__(1))); 
# 1078
extern int pthread_barrier_wait(pthread_barrier_t * __barrier) throw()
# 1079
 __attribute((__nonnull__(1))); 
# 1083
extern int pthread_barrierattr_init(pthread_barrierattr_t * __attr) throw()
# 1084
 __attribute((__nonnull__(1))); 
# 1087
extern int pthread_barrierattr_destroy(pthread_barrierattr_t * __attr) throw()
# 1088
 __attribute((__nonnull__(1))); 
# 1091
extern int pthread_barrierattr_getpshared(const pthread_barrierattr_t *__restrict__ __attr, int *__restrict__ __pshared) throw()
# 1094
 __attribute((__nonnull__(1, 2))); 
# 1097
extern int pthread_barrierattr_setpshared(pthread_barrierattr_t * __attr, int __pshared) throw()
# 1099
 __attribute((__nonnull__(1))); 
# 1111
extern int pthread_key_create(pthread_key_t * __key, void (* __destr_function)(void *)) throw()
# 1113
 __attribute((__nonnull__(1))); 
# 1116
extern int pthread_key_delete(pthread_key_t __key) throw(); 
# 1119
extern void *pthread_getspecific(pthread_key_t __key) throw(); 
# 1122
extern int pthread_setspecific(pthread_key_t __key, const void * __pointer) throw(); 
# 1128
extern int pthread_getcpuclockid(pthread_t __thread_id, __clockid_t * __clock_id) throw()
# 1130
 __attribute((__nonnull__(2))); 
# 1145
extern int pthread_atfork(void (* __prepare)(void), void (* __parent)(void), void (* __child)(void)) throw(); 
# 1159
}
# 47 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/gthr-default.h" 3
typedef pthread_t __gthread_t; 
# 48
typedef pthread_key_t __gthread_key_t; 
# 49
typedef pthread_once_t __gthread_once_t; 
# 50
typedef pthread_mutex_t __gthread_mutex_t; 
# 51
typedef pthread_mutex_t __gthread_recursive_mutex_t; 
# 52
typedef pthread_cond_t __gthread_cond_t; 
# 53
typedef timespec __gthread_time_t; 
# 101
static __typeof__(pthread_once) __gthrw_pthread_once __attribute((__weakref__("pthread_once"))); 
# 102
static __typeof__(pthread_getspecific) __gthrw_pthread_getspecific __attribute((__weakref__("pthread_getspecific"))); 
# 103
static __typeof__(pthread_setspecific) __gthrw_pthread_setspecific __attribute((__weakref__("pthread_setspecific"))); 
# 105
static __typeof__(pthread_create) __gthrw_pthread_create __attribute((__weakref__("pthread_create"))); 
# 106
static __typeof__(pthread_join) __gthrw_pthread_join __attribute((__weakref__("pthread_join"))); 
# 107
static __typeof__(pthread_equal) __gthrw_pthread_equal __attribute((__weakref__("pthread_equal"))); 
# 108
static __typeof__(pthread_self) __gthrw_pthread_self __attribute((__weakref__("pthread_self"))); 
# 109
static __typeof__(pthread_detach) __gthrw_pthread_detach __attribute((__weakref__("pthread_detach"))); 
# 111
static __typeof__(pthread_cancel) __gthrw_pthread_cancel __attribute((__weakref__("pthread_cancel"))); 
# 113
static __typeof__(sched_yield) __gthrw_sched_yield __attribute((__weakref__("sched_yield"))); 
# 115
static __typeof__(pthread_mutex_lock) __gthrw_pthread_mutex_lock __attribute((__weakref__("pthread_mutex_lock"))); 
# 116
static __typeof__(pthread_mutex_trylock) __gthrw_pthread_mutex_trylock __attribute((__weakref__("pthread_mutex_trylock"))); 
# 118
static __typeof__(pthread_mutex_timedlock) __gthrw_pthread_mutex_timedlock __attribute((__weakref__("pthread_mutex_timedlock"))); 
# 120
static __typeof__(pthread_mutex_unlock) __gthrw_pthread_mutex_unlock __attribute((__weakref__("pthread_mutex_unlock"))); 
# 121
static __typeof__(pthread_mutex_init) __gthrw_pthread_mutex_init __attribute((__weakref__("pthread_mutex_init"))); 
# 122
static __typeof__(pthread_mutex_destroy) __gthrw_pthread_mutex_destroy __attribute((__weakref__("pthread_mutex_destroy"))); 
# 124
static __typeof__(pthread_cond_init) __gthrw_pthread_cond_init __attribute((__weakref__("pthread_cond_init"))); 
# 125
static __typeof__(pthread_cond_broadcast) __gthrw_pthread_cond_broadcast __attribute((__weakref__("pthread_cond_broadcast"))); 
# 126
static __typeof__(pthread_cond_signal) __gthrw_pthread_cond_signal __attribute((__weakref__("pthread_cond_signal"))); 
# 127
static __typeof__(pthread_cond_wait) __gthrw_pthread_cond_wait __attribute((__weakref__("pthread_cond_wait"))); 
# 128
static __typeof__(pthread_cond_timedwait) __gthrw_pthread_cond_timedwait __attribute((__weakref__("pthread_cond_timedwait"))); 
# 129
static __typeof__(pthread_cond_destroy) __gthrw_pthread_cond_destroy __attribute((__weakref__("pthread_cond_destroy"))); 
# 131
static __typeof__(pthread_key_create) __gthrw_pthread_key_create __attribute((__weakref__("pthread_key_create"))); 
# 132
static __typeof__(pthread_key_delete) __gthrw_pthread_key_delete __attribute((__weakref__("pthread_key_delete"))); 
# 133
static __typeof__(pthread_mutexattr_init) __gthrw_pthread_mutexattr_init __attribute((__weakref__("pthread_mutexattr_init"))); 
# 134
static __typeof__(pthread_mutexattr_settype) __gthrw_pthread_mutexattr_settype __attribute((__weakref__("pthread_mutexattr_settype"))); 
# 135
static __typeof__(pthread_mutexattr_destroy) __gthrw_pthread_mutexattr_destroy __attribute((__weakref__("pthread_mutexattr_destroy"))); 
# 236
static __typeof__(pthread_key_create) __gthrw___pthread_key_create __attribute((__weakref__("__pthread_key_create"))); 
# 247
static inline int __gthread_active_p() 
# 248
{ 
# 249
static void *const __gthread_active_ptr = __extension__ ((void *)(&__gthrw___pthread_key_create)); 
# 251
return __gthread_active_ptr != (0); 
# 252
} 
# 659
static inline int __gthread_create(__gthread_t *__threadid, void *(*__func)(void *), void *
# 660
__args) 
# 661
{ 
# 662
return __gthrw_pthread_create(__threadid, __null, __func, __args); 
# 663
} 
# 666
static inline int __gthread_join(__gthread_t __threadid, void **__value_ptr) 
# 667
{ 
# 668
return __gthrw_pthread_join(__threadid, __value_ptr); 
# 669
} 
# 672
static inline int __gthread_detach(__gthread_t __threadid) 
# 673
{ 
# 674
return __gthrw_pthread_detach(__threadid); 
# 675
} 
# 678
static inline int __gthread_equal(__gthread_t __t1, __gthread_t __t2) 
# 679
{ 
# 680
return __gthrw_pthread_equal(__t1, __t2); 
# 681
} 
# 684
static inline __gthread_t __gthread_self() 
# 685
{ 
# 686
return __gthrw_pthread_self(); 
# 687
} 
# 690
static inline int __gthread_yield() 
# 691
{ 
# 692
return __gthrw_sched_yield(); 
# 693
} 
# 696
static inline int __gthread_once(__gthread_once_t *__once, void (*__func)(void)) 
# 697
{ 
# 698
if (__gthread_active_p()) { 
# 699
return __gthrw_pthread_once(__once, __func); } else { 
# 701
return -1; }  
# 702
} 
# 705
static inline int __gthread_key_create(__gthread_key_t *__key, void (*__dtor)(void *)) 
# 706
{ 
# 707
return __gthrw_pthread_key_create(__key, __dtor); 
# 708
} 
# 711
static inline int __gthread_key_delete(__gthread_key_t __key) 
# 712
{ 
# 713
return __gthrw_pthread_key_delete(__key); 
# 714
} 
# 717
static inline void *__gthread_getspecific(__gthread_key_t __key) 
# 718
{ 
# 719
return __gthrw_pthread_getspecific(__key); 
# 720
} 
# 723
static inline int __gthread_setspecific(__gthread_key_t __key, const void *__ptr) 
# 724
{ 
# 725
return __gthrw_pthread_setspecific(__key, __ptr); 
# 726
} 
# 729
static inline void __gthread_mutex_init_function(__gthread_mutex_t *__mutex) 
# 730
{ 
# 731
if (__gthread_active_p()) { 
# 732
__gthrw_pthread_mutex_init(__mutex, __null); }  
# 733
} 
# 736
static inline int __gthread_mutex_destroy(__gthread_mutex_t *__mutex) 
# 737
{ 
# 738
if (__gthread_active_p()) { 
# 739
return __gthrw_pthread_mutex_destroy(__mutex); } else { 
# 741
return 0; }  
# 742
} 
# 745
static inline int __gthread_mutex_lock(__gthread_mutex_t *__mutex) 
# 746
{ 
# 747
if (__gthread_active_p()) { 
# 748
return __gthrw_pthread_mutex_lock(__mutex); } else { 
# 750
return 0; }  
# 751
} 
# 754
static inline int __gthread_mutex_trylock(__gthread_mutex_t *__mutex) 
# 755
{ 
# 756
if (__gthread_active_p()) { 
# 757
return __gthrw_pthread_mutex_trylock(__mutex); } else { 
# 759
return 0; }  
# 760
} 
# 764
static inline int __gthread_mutex_timedlock(__gthread_mutex_t *__mutex, const __gthread_time_t *
# 765
__abs_timeout) 
# 766
{ 
# 767
if (__gthread_active_p()) { 
# 768
return __gthrw_pthread_mutex_timedlock(__mutex, __abs_timeout); } else { 
# 770
return 0; }  
# 771
} 
# 775
static inline int __gthread_mutex_unlock(__gthread_mutex_t *__mutex) 
# 776
{ 
# 777
if (__gthread_active_p()) { 
# 778
return __gthrw_pthread_mutex_unlock(__mutex); } else { 
# 780
return 0; }  
# 781
} 
# 808
static inline int __gthread_recursive_mutex_lock(__gthread_recursive_mutex_t *__mutex) 
# 809
{ 
# 810
return __gthread_mutex_lock(__mutex); 
# 811
} 
# 814
static inline int __gthread_recursive_mutex_trylock(__gthread_recursive_mutex_t *__mutex) 
# 815
{ 
# 816
return __gthread_mutex_trylock(__mutex); 
# 817
} 
# 821
static inline int __gthread_recursive_mutex_timedlock(__gthread_recursive_mutex_t *__mutex, const __gthread_time_t *
# 822
__abs_timeout) 
# 823
{ 
# 824
return __gthread_mutex_timedlock(__mutex, __abs_timeout); 
# 825
} 
# 829
static inline int __gthread_recursive_mutex_unlock(__gthread_recursive_mutex_t *__mutex) 
# 830
{ 
# 831
return __gthread_mutex_unlock(__mutex); 
# 832
} 
# 835
static inline int __gthread_recursive_mutex_destroy(__gthread_recursive_mutex_t *__mutex) 
# 836
{ 
# 837
return __gthread_mutex_destroy(__mutex); 
# 838
} 
# 850
static inline int __gthread_cond_broadcast(__gthread_cond_t *__cond) 
# 851
{ 
# 852
return __gthrw_pthread_cond_broadcast(__cond); 
# 853
} 
# 856
static inline int __gthread_cond_signal(__gthread_cond_t *__cond) 
# 857
{ 
# 858
return __gthrw_pthread_cond_signal(__cond); 
# 859
} 
# 862
static inline int __gthread_cond_wait(__gthread_cond_t *__cond, __gthread_mutex_t *__mutex) 
# 863
{ 
# 864
return __gthrw_pthread_cond_wait(__cond, __mutex); 
# 865
} 
# 868
static inline int __gthread_cond_timedwait(__gthread_cond_t *__cond, __gthread_mutex_t *__mutex, const __gthread_time_t *
# 869
__abs_timeout) 
# 870
{ 
# 871
return __gthrw_pthread_cond_timedwait(__cond, __mutex, __abs_timeout); 
# 872
} 
# 875
static inline int __gthread_cond_wait_recursive(__gthread_cond_t *__cond, __gthread_recursive_mutex_t *
# 876
__mutex) 
# 877
{ 
# 878
return __gthread_cond_wait(__cond, __mutex); 
# 879
} 
# 882
static inline int __gthread_cond_destroy(__gthread_cond_t *__cond) 
# 883
{ 
# 884
return __gthrw_pthread_cond_destroy(__cond); 
# 885
} 
# 151 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/gthr.h" 3
#pragma GCC visibility pop
# 32 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/atomic_word.h" 3
typedef int _Atomic_word; 
# 38 "/usr/include/c++/4.8/ext/atomicity.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 48
static inline _Atomic_word __exchange_and_add(volatile _Atomic_word *__mem, int __val) 
# 49
{ return __atomic_fetch_add(__mem, __val, 4); } 
# 52
static inline void __atomic_add(volatile _Atomic_word *__mem, int __val) 
# 53
{ __atomic_fetch_add(__mem, __val, 4); } 
# 65
static inline _Atomic_word __exchange_and_add_single(_Atomic_word *__mem, int __val) 
# 66
{ 
# 67
_Atomic_word __result = *__mem; 
# 68
(*__mem) += __val; 
# 69
return __result; 
# 70
} 
# 73
static inline void __atomic_add_single(_Atomic_word *__mem, int __val) 
# 74
{ (*__mem) += __val; } 
# 77
__attribute((__unused__)) static inline _Atomic_word 
# 78
__exchange_and_add_dispatch(_Atomic_word *__mem, int __val) 
# 79
{ 
# 81
if (__gthread_active_p()) { 
# 82
return __exchange_and_add(__mem, __val); } else { 
# 84
return __exchange_and_add_single(__mem, __val); }  
# 88
} 
# 91
__attribute((__unused__)) static inline void 
# 92
__atomic_add_dispatch(_Atomic_word *__mem, int __val) 
# 93
{ 
# 95
if (__gthread_active_p()) { 
# 96
__atomic_add(__mem, __val); } else { 
# 98
__atomic_add_single(__mem, __val); }  
# 102
} 
# 105
}
# 42 "/usr/include/c++/4.8/new" 3
#pragma GCC visibility push ( default )
# 44
extern "C++" {
# 46
namespace std { 
# 54
class bad_alloc : public exception { 
# 57
public: bad_alloc() throw() { } 
# 61
virtual ~bad_alloc() throw(); 
# 64
virtual const char *what() const throw(); 
# 65
}; 
# 67
struct nothrow_t { }; 
# 69
extern const nothrow_t nothrow; 
# 73
typedef void (*new_handler)(void); 
# 77
new_handler set_new_handler(new_handler) throw(); 
# 78
}
# 91
void *operator new(std::size_t)
# 92
 __attribute((__externally_visible__)); 
# 93
void *operator new[](std::size_t)
# 94
 __attribute((__externally_visible__)); 
# 95
void operator delete(void *) noexcept
# 96
 __attribute((__externally_visible__)); 
# 97
void operator delete[](void *) noexcept
# 98
 __attribute((__externally_visible__)); 
# 99
void *operator new(std::size_t, const std::nothrow_t &) noexcept
# 100
 __attribute((__externally_visible__)); 
# 101
void *operator new[](std::size_t, const std::nothrow_t &) noexcept
# 102
 __attribute((__externally_visible__)); 
# 103
void operator delete(void *, const std::nothrow_t &) noexcept
# 104
 __attribute((__externally_visible__)); 
# 105
void operator delete[](void *, const std::nothrow_t &) noexcept
# 106
 __attribute((__externally_visible__)); 
# 109
inline void *operator new(std::size_t, void *__p) noexcept 
# 110
{ return __p; } 
# 111
inline void *operator new[](std::size_t, void *__p) noexcept 
# 112
{ return __p; } 
# 115
inline void operator delete(void *, void *) noexcept { } 
# 116
inline void operator delete[](void *, void *) noexcept { } 
# 118
}
# 120
#pragma GCC visibility pop
# 40 "/usr/include/c++/4.8/ext/new_allocator.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 44
using std::size_t;
# 45
using std::ptrdiff_t;
# 57
template< class _Tp> 
# 58
class new_allocator { 
# 61
public: typedef std::size_t size_type; 
# 62
typedef std::ptrdiff_t difference_type; 
# 63
typedef _Tp *pointer; 
# 64
typedef const _Tp *const_pointer; 
# 65
typedef _Tp &reference; 
# 66
typedef const _Tp &const_reference; 
# 67
typedef _Tp value_type; 
# 69
template< class _Tp1> 
# 70
struct rebind { 
# 71
typedef __gnu_cxx::new_allocator< _Tp1>  other; }; 
# 76
typedef std::true_type propagate_on_container_move_assignment; 
# 79
new_allocator() noexcept { } 
# 81
new_allocator(const new_allocator &) noexcept { } 
# 83
template< class _Tp1> 
# 84
new_allocator(const __gnu_cxx::new_allocator< _Tp1>  &) noexcept { } 
# 86
~new_allocator() noexcept { } 
# 89
pointer address(reference __x) const noexcept 
# 90
{ return std::__addressof(__x); } 
# 93
const_pointer address(const_reference __x) const noexcept 
# 94
{ return std::__addressof(__x); } 
# 99
pointer allocate(size_type __n, const void * = 0) 
# 100
{ 
# 101
if (__n > this->max_size()) { 
# 102
std::__throw_bad_alloc(); }  
# 104
return static_cast< _Tp *>(::operator new(__n * sizeof(_Tp))); 
# 105
} 
# 109
void deallocate(pointer __p, size_type) 
# 110
{ ::operator delete(__p); } 
# 113
size_type max_size() const noexcept 
# 114
{ return ((std::size_t)(-1)) / sizeof(_Tp); } 
# 117
template< class _Up, class ..._Args> void 
# 119
construct(_Up *__p, _Args &&...__args) 
# 120
{ ::new ((void *)__p) (_Up)(std::forward< _Args> (__args)...); } 
# 122
template< class _Up> void 
# 124
destroy(_Up *__p) { (__p->~_Up()); } 
# 135
}; 
# 137
template< class _Tp> inline bool 
# 139
operator==(const new_allocator< _Tp>  &, const new_allocator< _Tp>  &) 
# 140
{ return true; } 
# 142
template< class _Tp> inline bool 
# 144
operator!=(const new_allocator< _Tp>  &, const new_allocator< _Tp>  &) 
# 145
{ return false; } 
# 148
}
# 36 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/c++allocator.h" 3
namespace std { 
# 47
template< class _Tp> using __allocator_base = __gnu_cxx::new_allocator< _Tp> ; 
# 49
}
# 52 "/usr/include/c++/4.8/bits/allocator.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 63
template<> class allocator< void>  { 
# 66
public: typedef size_t size_type; 
# 67
typedef ptrdiff_t difference_type; 
# 68
typedef void *pointer; 
# 69
typedef const void *const_pointer; 
# 70
typedef void value_type; 
# 72
template< class _Tp1> 
# 73
struct rebind { 
# 74
typedef std::allocator< _Tp1>  other; }; 
# 79
typedef true_type propagate_on_container_move_assignment; 
# 81
}; 
# 91
template< class _Tp> 
# 92
class allocator : public __allocator_base< _Tp>  { 
# 95
public: typedef ::std::size_t size_type; 
# 96
typedef ::std::ptrdiff_t difference_type; 
# 97
typedef _Tp *pointer; 
# 98
typedef const _Tp *const_pointer; 
# 99
typedef _Tp &reference; 
# 100
typedef const _Tp &const_reference; 
# 101
typedef _Tp value_type; 
# 103
template< class _Tp1> 
# 104
struct rebind { 
# 105
typedef ::std::allocator< _Tp1>  other; }; 
# 110
typedef ::std::true_type propagate_on_container_move_assignment; 
# 113
allocator() throw() { } 
# 115
allocator(const allocator &__a) throw() : ::std::__allocator_base< _Tp> (__a) 
# 116
{ } 
# 118
template< class _Tp1> 
# 119
allocator(const ::std::allocator< _Tp1>  &) throw() { } 
# 121
~allocator() throw() { } 
# 124
}; 
# 126
template< class _T1, class _T2> inline bool 
# 128
operator==(const allocator< _T1>  &, const allocator< _T2>  &) 
# 129
{ return true; } 
# 131
template< class _Tp> inline bool 
# 133
operator==(const allocator< _Tp>  &, const allocator< _Tp>  &) 
# 134
{ return true; } 
# 136
template< class _T1, class _T2> inline bool 
# 138
operator!=(const allocator< _T1>  &, const allocator< _T2>  &) 
# 139
{ return false; } 
# 141
template< class _Tp> inline bool 
# 143
operator!=(const allocator< _Tp>  &, const allocator< _Tp>  &) 
# 144
{ return false; } 
# 151
extern template class allocator< char> ;
# 152
extern template class allocator< wchar_t> ;
# 159
template< class _Alloc, bool  = __is_empty(_Alloc)> 
# 160
struct __alloc_swap { 
# 161
static void _S_do_it(_Alloc &, _Alloc &) { } }; 
# 163
template< class _Alloc> 
# 164
struct __alloc_swap< _Alloc, false>  { 
# 167
static void _S_do_it(_Alloc &__one, _Alloc &__two) 
# 168
{ 
# 170
if (__one != __two) { 
# 171
swap(__one, __two); }  
# 172
} 
# 173
}; 
# 176
template< class _Alloc, bool  = __is_empty(_Alloc)> 
# 177
struct __alloc_neq { 
# 180
static bool _S_do_it(const _Alloc &, const _Alloc &) 
# 181
{ return false; } 
# 182
}; 
# 184
template< class _Alloc> 
# 185
struct __alloc_neq< _Alloc, false>  { 
# 188
static bool _S_do_it(const _Alloc &__one, const _Alloc &__two) 
# 189
{ return __one != __two; } 
# 190
}; 
# 193
template< class _Tp, bool 
# 194
 = __or_< is_copy_constructible< typename _Tp::value_type> , is_nothrow_move_constructible< typename _Tp::value_type> > ::value> 
# 196
struct __shrink_to_fit_aux { 
# 197
static bool _S_do_it(_Tp &) { return false; } }; 
# 199
template< class _Tp> 
# 200
struct __shrink_to_fit_aux< _Tp, true>  { 
# 203
static bool _S_do_it(_Tp &__c) 
# 204
{ 
# 205
try 
# 206
{ 
# 207
(_Tp(__make_move_if_noexcept_iterator((__c.begin())), __make_move_if_noexcept_iterator((__c.end())), (__c.get_allocator())).swap(__c)); 
# 210
return true; 
# 211
} 
# 212
catch (...) 
# 213
{ return false; }  
# 214
} 
# 215
}; 
# 219
}
# 36 "/usr/include/c++/4.8/bits/cxxabi_forced.h" 3
#pragma GCC visibility push ( default )
# 39
namespace __cxxabiv1 { 
# 48
class __forced_unwind { 
# 50
virtual ~__forced_unwind() throw(); 
# 53
virtual void __pure_dummy() = 0; 
# 54
}; 
# 55
}
# 58
#pragma GCC visibility pop
# 38 "/usr/include/c++/4.8/bits/ostream_insert.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 42
template< class _CharT, class _Traits> inline void 
# 44
__ostream_write(basic_ostream< _CharT, _Traits>  &__out, const _CharT *
# 45
__s, streamsize __n) 
# 46
{ 
# 47
typedef basic_ostream< _CharT, _Traits>  __ostream_type; 
# 48
typedef typename basic_ostream< _CharT, _Traits> ::ios_base __ios_base; 
# 50
const streamsize __put = ((__out.rdbuf())->sputn(__s, __n)); 
# 51
if (__put != __n) { 
# 52
(__out.setstate(__ios_base::badbit)); }  
# 53
} 
# 55
template< class _CharT, class _Traits> inline void 
# 57
__ostream_fill(basic_ostream< _CharT, _Traits>  &__out, streamsize __n) 
# 58
{ 
# 59
typedef basic_ostream< _CharT, _Traits>  __ostream_type; 
# 60
typedef typename basic_ostream< _CharT, _Traits> ::ios_base __ios_base; 
# 62
const _CharT __c = (__out.fill()); 
# 63
for (; __n > (0); --__n) 
# 64
{ 
# 65
const typename _Traits::int_type __put = ((__out.rdbuf())->sputc(__c)); 
# 66
if (_Traits::eq_int_type(__put, _Traits::eof())) 
# 67
{ 
# 68
(__out.setstate(__ios_base::badbit)); 
# 69
break; 
# 70
}  
# 71
}  
# 72
} 
# 74
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 76
__ostream_insert(basic_ostream< _CharT, _Traits>  &__out, const _CharT *
# 77
__s, streamsize __n) 
# 78
{ 
# 79
typedef basic_ostream< _CharT, _Traits>  __ostream_type; 
# 80
typedef typename basic_ostream< _CharT, _Traits> ::ios_base __ios_base; 
# 82
typename basic_ostream< _CharT, _Traits> ::sentry __cerb(__out); 
# 83
if (__cerb) 
# 84
{ 
# 85
try 
# 86
{ 
# 87
const streamsize __w = (__out.width()); 
# 88
if (__w > __n) 
# 89
{ 
# 90
const bool __left = ((__out.flags()) & __ios_base::adjustfield) == __ios_base::left; 
# 93
if (!__left) { 
# 94
__ostream_fill(__out, __w - __n); }  
# 95
if ((__out.good())) { 
# 96
__ostream_write(__out, __s, __n); }  
# 97
if (__left && (__out.good())) { 
# 98
__ostream_fill(__out, __w - __n); }  
# 99
} else { 
# 101
__ostream_write(__out, __s, __n); }  
# 102
(__out.width(0)); 
# 103
} 
# 104
catch (__cxxabiv1::__forced_unwind &) 
# 105
{ 
# 106
(__out._M_setstate(__ios_base::badbit)); 
# 107
throw; 
# 108
} 
# 109
catch (...) 
# 110
{ (__out._M_setstate(__ios_base::badbit)); }  
# 111
}  
# 112
return __out; 
# 113
} 
# 118
extern template basic_ostream< char>  &__ostream_insert(basic_ostream< char>  & __out, const char * __s, streamsize __n);
# 121
extern template basic_ostream< wchar_t>  &__ostream_insert(basic_ostream< wchar_t>  & __out, const wchar_t * __s, streamsize __n);
# 127
}
# 59 "/usr/include/c++/4.8/bits/stl_function.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 100
template< class _Arg, class _Result> 
# 101
struct unary_function { 
# 104
typedef _Arg argument_type; 
# 107
typedef _Result result_type; 
# 108
}; 
# 113
template< class _Arg1, class _Arg2, class _Result> 
# 114
struct binary_function { 
# 117
typedef _Arg1 first_argument_type; 
# 120
typedef _Arg2 second_argument_type; 
# 123
typedef _Result result_type; 
# 124
}; 
# 139
template< class _Tp> 
# 140
struct plus : public binary_function< _Tp, _Tp, _Tp>  { 
# 143
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 144
{ return __x + __y; } 
# 145
}; 
# 148
template< class _Tp> 
# 149
struct minus : public binary_function< _Tp, _Tp, _Tp>  { 
# 152
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 153
{ return __x - __y; } 
# 154
}; 
# 157
template< class _Tp> 
# 158
struct multiplies : public binary_function< _Tp, _Tp, _Tp>  { 
# 161
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 162
{ return __x * __y; } 
# 163
}; 
# 166
template< class _Tp> 
# 167
struct divides : public binary_function< _Tp, _Tp, _Tp>  { 
# 170
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 171
{ return __x / __y; } 
# 172
}; 
# 175
template< class _Tp> 
# 176
struct modulus : public binary_function< _Tp, _Tp, _Tp>  { 
# 179
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 180
{ return __x % __y; } 
# 181
}; 
# 184
template< class _Tp> 
# 185
struct negate : public unary_function< _Tp, _Tp>  { 
# 188
_Tp operator()(const _Tp &__x) const 
# 189
{ return -__x; } 
# 190
}; 
# 203
template< class _Tp> 
# 204
struct equal_to : public binary_function< _Tp, _Tp, bool>  { 
# 207
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 208
{ return __x == __y; } 
# 209
}; 
# 212
template< class _Tp> 
# 213
struct not_equal_to : public binary_function< _Tp, _Tp, bool>  { 
# 216
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 217
{ return __x != __y; } 
# 218
}; 
# 221
template< class _Tp> 
# 222
struct greater : public binary_function< _Tp, _Tp, bool>  { 
# 225
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 226
{ return __x > __y; } 
# 227
}; 
# 230
template< class _Tp> 
# 231
struct less : public binary_function< _Tp, _Tp, bool>  { 
# 234
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 235
{ return __x < __y; } 
# 236
}; 
# 239
template< class _Tp> 
# 240
struct greater_equal : public binary_function< _Tp, _Tp, bool>  { 
# 243
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 244
{ return __x >= __y; } 
# 245
}; 
# 248
template< class _Tp> 
# 249
struct less_equal : public binary_function< _Tp, _Tp, bool>  { 
# 252
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 253
{ return __x <= __y; } 
# 254
}; 
# 267
template< class _Tp> 
# 268
struct logical_and : public binary_function< _Tp, _Tp, bool>  { 
# 271
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 272
{ return __x && __y; } 
# 273
}; 
# 276
template< class _Tp> 
# 277
struct logical_or : public binary_function< _Tp, _Tp, bool>  { 
# 280
bool operator()(const _Tp &__x, const _Tp &__y) const 
# 281
{ return __x || __y; } 
# 282
}; 
# 285
template< class _Tp> 
# 286
struct logical_not : public unary_function< _Tp, bool>  { 
# 289
bool operator()(const _Tp &__x) const 
# 290
{ return !__x; } 
# 291
}; 
# 296
template< class _Tp> 
# 297
struct bit_and : public binary_function< _Tp, _Tp, _Tp>  { 
# 300
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 301
{ return __x & __y; } 
# 302
}; 
# 304
template< class _Tp> 
# 305
struct bit_or : public binary_function< _Tp, _Tp, _Tp>  { 
# 308
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 309
{ return __x | __y; } 
# 310
}; 
# 312
template< class _Tp> 
# 313
struct bit_xor : public binary_function< _Tp, _Tp, _Tp>  { 
# 316
_Tp operator()(const _Tp &__x, const _Tp &__y) const 
# 317
{ return __x ^ __y; } 
# 318
}; 
# 350
template< class _Predicate> 
# 351
class unary_negate : public unary_function< typename _Predicate::argument_type, bool>  { 
# 355
protected: _Predicate _M_pred; 
# 359
public: explicit unary_negate(const _Predicate &__x) : _M_pred(__x) { } 
# 362
bool operator()(const typename _Predicate::argument_type &__x) const 
# 363
{ return !(_M_pred)(__x); } 
# 364
}; 
# 367
template< class _Predicate> inline unary_negate< _Predicate>  
# 369
not1(const _Predicate &__pred) 
# 370
{ return ((unary_negate< _Predicate> )(__pred)); } 
# 373
template< class _Predicate> 
# 374
class binary_negate : public binary_function< typename _Predicate::first_argument_type, typename _Predicate::second_argument_type, bool>  { 
# 379
protected: _Predicate _M_pred; 
# 383
public: explicit binary_negate(const _Predicate &__x) : _M_pred(__x) { } 
# 386
bool operator()(const typename _Predicate::first_argument_type &__x, const typename _Predicate::second_argument_type &
# 387
__y) const 
# 388
{ return !(_M_pred)(__x, __y); } 
# 389
}; 
# 392
template< class _Predicate> inline binary_negate< _Predicate>  
# 394
not2(const _Predicate &__pred) 
# 395
{ return ((binary_negate< _Predicate> )(__pred)); } 
# 421
template< class _Arg, class _Result> 
# 422
class pointer_to_unary_function : public unary_function< _Arg, _Result>  { 
# 425
protected: _Result (*_M_ptr)(_Arg); 
# 428
public: pointer_to_unary_function() { } 
# 431
explicit pointer_to_unary_function(_Result (*__x)(_Arg)) : _M_ptr(__x) 
# 432
{ } 
# 435
_Result operator()(_Arg __x) const 
# 436
{ return (_M_ptr)(__x); } 
# 437
}; 
# 440
template< class _Arg, class _Result> inline pointer_to_unary_function< _Arg, _Result>  
# 442
ptr_fun(_Result (*__x)(_Arg)) 
# 443
{ return ((pointer_to_unary_function< _Arg, _Result> )(__x)); } 
# 446
template< class _Arg1, class _Arg2, class _Result> 
# 447
class pointer_to_binary_function : public binary_function< _Arg1, _Arg2, _Result>  { 
# 451
protected: _Result (*_M_ptr)(_Arg1, _Arg2); 
# 454
public: pointer_to_binary_function() { } 
# 457
explicit pointer_to_binary_function(_Result (*__x)(_Arg1, _Arg2)) : _M_ptr(__x) 
# 458
{ } 
# 461
_Result operator()(_Arg1 __x, _Arg2 __y) const 
# 462
{ return (_M_ptr)(__x, __y); } 
# 463
}; 
# 466
template< class _Arg1, class _Arg2, class _Result> inline pointer_to_binary_function< _Arg1, _Arg2, _Result>  
# 468
ptr_fun(_Result (*__x)(_Arg1, _Arg2)) 
# 469
{ return ((pointer_to_binary_function< _Arg1, _Arg2, _Result> )(__x)); } 
# 472
template< class _Tp> 
# 473
struct _Identity : public unary_function< _Tp, _Tp>  { 
# 477
_Tp &operator()(_Tp &__x) const 
# 478
{ return __x; } 
# 481
const _Tp &operator()(const _Tp &__x) const 
# 482
{ return __x; } 
# 483
}; 
# 485
template< class _Pair> 
# 486
struct _Select1st : public unary_function< _Pair, typename _Pair::first_type>  { 
# 490
typename _Pair::first_type &operator()(_Pair &__x) const 
# 491
{ return __x.first; } 
# 494
const typename _Pair::first_type &operator()(const _Pair &__x) const 
# 495
{ return __x.first; } 
# 498
template< class _Pair2> typename _Pair2::first_type &
# 500
operator()(_Pair2 &__x) const 
# 501
{ return __x.first; } 
# 503
template< class _Pair2> const typename _Pair2::first_type &
# 505
operator()(const _Pair2 &__x) const 
# 506
{ return __x.first; } 
# 508
}; 
# 510
template< class _Pair> 
# 511
struct _Select2nd : public unary_function< _Pair, typename _Pair::second_type>  { 
# 515
typename _Pair::second_type &operator()(_Pair &__x) const 
# 516
{ return __x.second; } 
# 519
const typename _Pair::second_type &operator()(const _Pair &__x) const 
# 520
{ return __x.second; } 
# 521
}; 
# 541
template< class _Ret, class _Tp> 
# 542
class mem_fun_t : public unary_function< _Tp *, _Ret>  { 
# 546
public: explicit mem_fun_t(_Ret (_Tp::*__pf)(void)) : _M_f(__pf) 
# 547
{ } 
# 550
_Ret operator()(_Tp *__p) const 
# 551
{ return (__p->*(_M_f))(); } 
# 554
private: _Ret (_Tp::*_M_f)(void); 
# 555
}; 
# 559
template< class _Ret, class _Tp> 
# 560
class const_mem_fun_t : public unary_function< const _Tp *, _Ret>  { 
# 564
public: explicit const_mem_fun_t(_Ret (_Tp::*__pf)(void) const) : _M_f(__pf) 
# 565
{ } 
# 568
_Ret operator()(const _Tp *__p) const 
# 569
{ return (__p->*(_M_f))(); } 
# 572
private: _Ret (_Tp::*_M_f)(void) const; 
# 573
}; 
# 577
template< class _Ret, class _Tp> 
# 578
class mem_fun_ref_t : public unary_function< _Tp, _Ret>  { 
# 582
public: explicit mem_fun_ref_t(_Ret (_Tp::*__pf)(void)) : _M_f(__pf) 
# 583
{ } 
# 586
_Ret operator()(_Tp &__r) const 
# 587
{ return (__r.*(_M_f))(); } 
# 590
private: _Ret (_Tp::*_M_f)(void); 
# 591
}; 
# 595
template< class _Ret, class _Tp> 
# 596
class const_mem_fun_ref_t : public unary_function< _Tp, _Ret>  { 
# 600
public: explicit const_mem_fun_ref_t(_Ret (_Tp::*__pf)(void) const) : _M_f(__pf) 
# 601
{ } 
# 604
_Ret operator()(const _Tp &__r) const 
# 605
{ return (__r.*(_M_f))(); } 
# 608
private: _Ret (_Tp::*_M_f)(void) const; 
# 609
}; 
# 613
template< class _Ret, class _Tp, class _Arg> 
# 614
class mem_fun1_t : public binary_function< _Tp *, _Arg, _Ret>  { 
# 618
public: explicit mem_fun1_t(_Ret (_Tp::*__pf)(_Arg)) : _M_f(__pf) 
# 619
{ } 
# 622
_Ret operator()(_Tp *__p, _Arg __x) const 
# 623
{ return (__p->*(_M_f))(__x); } 
# 626
private: _Ret (_Tp::*_M_f)(_Arg); 
# 627
}; 
# 631
template< class _Ret, class _Tp, class _Arg> 
# 632
class const_mem_fun1_t : public binary_function< const _Tp *, _Arg, _Ret>  { 
# 636
public: explicit const_mem_fun1_t(_Ret (_Tp::*__pf)(_Arg) const) : _M_f(__pf) 
# 637
{ } 
# 640
_Ret operator()(const _Tp *__p, _Arg __x) const 
# 641
{ return (__p->*(_M_f))(__x); } 
# 644
private: _Ret (_Tp::*_M_f)(_Arg) const; 
# 645
}; 
# 649
template< class _Ret, class _Tp, class _Arg> 
# 650
class mem_fun1_ref_t : public binary_function< _Tp, _Arg, _Ret>  { 
# 654
public: explicit mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg)) : _M_f(__pf) 
# 655
{ } 
# 658
_Ret operator()(_Tp &__r, _Arg __x) const 
# 659
{ return (__r.*(_M_f))(__x); } 
# 662
private: _Ret (_Tp::*_M_f)(_Arg); 
# 663
}; 
# 667
template< class _Ret, class _Tp, class _Arg> 
# 668
class const_mem_fun1_ref_t : public binary_function< _Tp, _Arg, _Ret>  { 
# 672
public: explicit const_mem_fun1_ref_t(_Ret (_Tp::*__pf)(_Arg) const) : _M_f(__pf) 
# 673
{ } 
# 676
_Ret operator()(const _Tp &__r, _Arg __x) const 
# 677
{ return (__r.*(_M_f))(__x); } 
# 680
private: _Ret (_Tp::*_M_f)(_Arg) const; 
# 681
}; 
# 685
template< class _Ret, class _Tp> inline mem_fun_t< _Ret, _Tp>  
# 687
mem_fun(_Ret (_Tp::*__f)(void)) 
# 688
{ return ((mem_fun_t< _Ret, _Tp> )(__f)); } 
# 690
template< class _Ret, class _Tp> inline const_mem_fun_t< _Ret, _Tp>  
# 692
mem_fun(_Ret (_Tp::*__f)(void) const) 
# 693
{ return ((const_mem_fun_t< _Ret, _Tp> )(__f)); } 
# 695
template< class _Ret, class _Tp> inline mem_fun_ref_t< _Ret, _Tp>  
# 697
mem_fun_ref(_Ret (_Tp::*__f)(void)) 
# 698
{ return ((mem_fun_ref_t< _Ret, _Tp> )(__f)); } 
# 700
template< class _Ret, class _Tp> inline const_mem_fun_ref_t< _Ret, _Tp>  
# 702
mem_fun_ref(_Ret (_Tp::*__f)(void) const) 
# 703
{ return ((const_mem_fun_ref_t< _Ret, _Tp> )(__f)); } 
# 705
template< class _Ret, class _Tp, class _Arg> inline mem_fun1_t< _Ret, _Tp, _Arg>  
# 707
mem_fun(_Ret (_Tp::*__f)(_Arg)) 
# 708
{ return ((mem_fun1_t< _Ret, _Tp, _Arg> )(__f)); } 
# 710
template< class _Ret, class _Tp, class _Arg> inline const_mem_fun1_t< _Ret, _Tp, _Arg>  
# 712
mem_fun(_Ret (_Tp::*__f)(_Arg) const) 
# 713
{ return ((const_mem_fun1_t< _Ret, _Tp, _Arg> )(__f)); } 
# 715
template< class _Ret, class _Tp, class _Arg> inline mem_fun1_ref_t< _Ret, _Tp, _Arg>  
# 717
mem_fun_ref(_Ret (_Tp::*__f)(_Arg)) 
# 718
{ return ((mem_fun1_ref_t< _Ret, _Tp, _Arg> )(__f)); } 
# 720
template< class _Ret, class _Tp, class _Arg> inline const_mem_fun1_ref_t< _Ret, _Tp, _Arg>  
# 722
mem_fun_ref(_Ret (_Tp::*__f)(_Arg) const) 
# 723
{ return ((const_mem_fun1_ref_t< _Ret, _Tp, _Arg> )(__f)); } 
# 728
}
# 59 "/usr/include/c++/4.8/backward/binders.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 103
template< class _Operation> 
# 104
class binder1st : public unary_function< typename _Operation::second_argument_type, typename _Operation::result_type>  { 
# 109
protected: _Operation op; 
# 110
typename _Operation::first_argument_type value; 
# 113
public: binder1st(const _Operation &__x, const typename _Operation::first_argument_type &
# 114
__y) : op(__x), value(__y) 
# 115
{ } 
# 118
typename _Operation::result_type operator()(const typename _Operation::second_argument_type &__x) const 
# 119
{ return (op)(value, __x); } 
# 124
typename _Operation::result_type operator()(typename _Operation::second_argument_type &__x) const 
# 125
{ return (op)(value, __x); } 
# 126
} __attribute((__deprecated__)); 
# 129
template< class _Operation, class _Tp> inline binder1st< _Operation>  
# 131
bind1st(const _Operation &__fn, const _Tp &__x) 
# 132
{ 
# 133
typedef typename _Operation::first_argument_type _Arg1_type; 
# 134
return binder1st< _Operation> (__fn, (_Arg1_type)__x); 
# 135
} 
# 138
template< class _Operation> 
# 139
class binder2nd : public unary_function< typename _Operation::first_argument_type, typename _Operation::result_type>  { 
# 144
protected: _Operation op; 
# 145
typename _Operation::second_argument_type value; 
# 148
public: binder2nd(const _Operation &__x, const typename _Operation::second_argument_type &
# 149
__y) : op(__x), value(__y) 
# 150
{ } 
# 153
typename _Operation::result_type operator()(const typename _Operation::first_argument_type &__x) const 
# 154
{ return (op)(__x, value); } 
# 159
typename _Operation::result_type operator()(typename _Operation::first_argument_type &__x) const 
# 160
{ return (op)(__x, value); } 
# 161
} __attribute((__deprecated__)); 
# 164
template< class _Operation, class _Tp> inline binder2nd< _Operation>  
# 166
bind2nd(const _Operation &__fn, const _Tp &__x) 
# 167
{ 
# 168
typedef typename _Operation::second_argument_type _Arg2_type; 
# 169
return binder2nd< _Operation> (__fn, (_Arg2_type)__x); 
# 170
} 
# 174
}
# 37 "/usr/include/c++/4.8/bits/range_access.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 46
template< class _Container> inline auto 
# 48
begin(_Container &__cont)->__decltype(((__cont.begin()))) 
# 49
{ return (__cont.begin()); } 
# 56
template< class _Container> inline auto 
# 58
begin(const _Container &__cont)->__decltype(((__cont.begin()))) 
# 59
{ return (__cont.begin()); } 
# 66
template< class _Container> inline auto 
# 68
end(_Container &__cont)->__decltype(((__cont.end()))) 
# 69
{ return (__cont.end()); } 
# 76
template< class _Container> inline auto 
# 78
end(const _Container &__cont)->__decltype(((__cont.end()))) 
# 79
{ return (__cont.end()); } 
# 85
template< class _Tp, size_t _Nm> inline _Tp *
# 87
begin(_Tp (&__arr)[_Nm]) 
# 88
{ return __arr; } 
# 95
template< class _Tp, size_t _Nm> inline _Tp *
# 97
end(_Tp (&__arr)[_Nm]) 
# 98
{ return (__arr) + _Nm; } 
# 101
}
# 39 "/usr/include/c++/4.8/initializer_list" 3
#pragma GCC visibility push ( default )
# 43
namespace std { 
# 46
template< class _E> 
# 47
class initializer_list { 
# 50
public: typedef _E value_type; 
# 51
typedef const _E &reference; 
# 52
typedef const _E &const_reference; 
# 53
typedef size_t size_type; 
# 54
typedef const _E *iterator; 
# 55
typedef const _E *const_iterator; 
# 58
private: iterator _M_array; 
# 59
size_type _M_len; 
# 62
constexpr initializer_list(const_iterator __a, size_type __l) : _M_array(__a), _M_len(__l) 
# 63
{ } 
# 66
public: constexpr initializer_list() noexcept : _M_array((0)), _M_len((0)) 
# 67
{ } 
# 71
constexpr size_type size() const noexcept { return _M_len; } 
# 75
constexpr const_iterator begin() const noexcept { return _M_array; } 
# 79
constexpr const_iterator end() const noexcept { return begin() + size(); } 
# 80
}; 
# 87
template< class _Tp> constexpr const _Tp *
# 89
begin(initializer_list< _Tp>  __ils) noexcept 
# 90
{ return (__ils.begin()); } 
# 97
template< class _Tp> constexpr const _Tp *
# 99
end(initializer_list< _Tp>  __ils) noexcept 
# 100
{ return (__ils.end()); } 
# 101
}
# 103
#pragma GCC visibility pop
# 45 "/usr/include/c++/4.8/bits/basic_string.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 111
template< class _CharT, class _Traits, class _Alloc> 
# 112
class basic_string { 
# 114
typedef typename _Alloc::template rebind< _CharT> ::other _CharT_alloc_type; 
# 118
public: typedef _Traits traits_type; 
# 119
typedef typename _Traits::char_type value_type; 
# 120
typedef _Alloc allocator_type; 
# 121
typedef typename _Alloc::template rebind< _CharT> ::other::size_type size_type; 
# 122
typedef typename _Alloc::template rebind< _CharT> ::other::difference_type difference_type; 
# 123
typedef typename _Alloc::template rebind< _CharT> ::other::reference reference; 
# 124
typedef typename _Alloc::template rebind< _CharT> ::other::const_reference const_reference; 
# 125
typedef typename _Alloc::template rebind< _CharT> ::other::pointer pointer; 
# 126
typedef typename _Alloc::template rebind< _CharT> ::other::const_pointer const_pointer; 
# 127
typedef __gnu_cxx::__normal_iterator< typename _Alloc::template rebind< _CharT> ::other::pointer, basic_string>  iterator; 
# 129
typedef __gnu_cxx::__normal_iterator< typename _Alloc::template rebind< _CharT> ::other::const_pointer, basic_string>  const_iterator; 
# 130
typedef std::reverse_iterator< __gnu_cxx::__normal_iterator< typename _Alloc::template rebind< _CharT> ::other::const_pointer, basic_string> >  const_reverse_iterator; 
# 131
typedef std::reverse_iterator< __gnu_cxx::__normal_iterator< typename _Alloc::template rebind< _CharT> ::other::pointer, basic_string> >  reverse_iterator; 
# 148
private: struct _Rep_base { 
# 150
size_type _M_length; 
# 151
size_type _M_capacity; 
# 152
_Atomic_word _M_refcount; 
# 153
}; 
# 155
struct _Rep : public _Rep_base { 
# 158
typedef typename _Alloc::template rebind< char> ::other _Raw_bytes_alloc; 
# 173
static const typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type _S_max_size; 
# 174
static const _CharT _S_terminal; 
# 178
static typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type _S_empty_rep_storage[]; 
# 181
static _Rep &_S_empty_rep() 
# 182
{ 
# 186
void *__p = (reinterpret_cast< void *>(&_S_empty_rep_storage)); 
# 187
return *(reinterpret_cast< _Rep *>(__p)); 
# 188
} 
# 191
bool _M_is_leaked() const 
# 192
{ return (this->_M_refcount) < 0; } 
# 195
bool _M_is_shared() const 
# 196
{ return (this->_M_refcount) > 0; } 
# 199
void _M_set_leaked() 
# 200
{ (this->_M_refcount) = (-1); } 
# 203
void _M_set_sharable() 
# 204
{ (this->_M_refcount) = 0; } 
# 207
void _M_set_length_and_sharable(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __n) 
# 208
{ 
# 210
if (__builtin_expect(this != (&(_S_empty_rep)()), false)) 
# 212
{ 
# 213
this->_M_set_sharable(); 
# 214
(this->_M_length) = __n; 
# 215
traits_type::assign(this->_M_refdata()[__n], _S_terminal); 
# 218
}  
# 219
} 
# 222
_CharT *_M_refdata() throw() 
# 223
{ return reinterpret_cast< _CharT *>(this + 1); } 
# 226
_CharT *_M_grab(const _Alloc &__alloc1, const _Alloc &__alloc2) 
# 227
{ 
# 228
return ((!_M_is_leaked()) && (__alloc1 == __alloc2)) ? _M_refcopy() : _M_clone(__alloc1); 
# 230
} 
# 234
static _Rep *_S_create(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type, typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type, const _Alloc &); 
# 237
void _M_dispose(const _Alloc &__a) 
# 238
{ 
# 240
if (__builtin_expect(this != (&(_S_empty_rep)()), false)) 
# 242
{ 
# 244
; 
# 245
if (::__gnu_cxx::__exchange_and_add_dispatch(&(this->_M_refcount), -1) <= 0) 
# 247
{ 
# 248
; 
# 249
_M_destroy(__a); 
# 250
}  
# 251
}  
# 252
} 
# 255
void _M_destroy(const _Alloc &) throw(); 
# 258
_CharT *_M_refcopy() throw() 
# 259
{ 
# 261
if (__builtin_expect(this != (&(_S_empty_rep)()), false)) { 
# 263
::__gnu_cxx::__atomic_add_dispatch(&(this->_M_refcount), 1); }  
# 264
return _M_refdata(); 
# 265
} 
# 268
_CharT *_M_clone(const _Alloc &, typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __res = 0); 
# 269
}; 
# 272
struct _Alloc_hider : public _Alloc { 
# 274
_Alloc_hider(_CharT *__dat, const _Alloc &__a) : _Alloc(__a), _M_p(__dat) 
# 275
{ } 
# 277
_CharT *_M_p; 
# 278
}; 
# 285
public: static const size_type npos = (static_cast< size_type>(-1)); 
# 289
private: mutable _Alloc_hider _M_dataplus; 
# 292
_CharT *_M_data() const 
# 293
{ return (_M_dataplus)._M_p; } 
# 296
_CharT *_M_data(_CharT *__p) 
# 297
{ return ((_M_dataplus)._M_p) = __p; } 
# 300
_Rep *_M_rep() const 
# 301
{ return &((reinterpret_cast< _Rep *>(_M_data()))[-1]); } 
# 306
iterator _M_ibegin() const 
# 307
{ return ((iterator)(_M_data())); } 
# 310
iterator _M_iend() const 
# 311
{ return ((iterator)(_M_data() + this->size())); } 
# 314
void _M_leak() 
# 315
{ 
# 316
if (!(_M_rep()->_M_is_leaked())) { 
# 317
_M_leak_hard(); }  
# 318
} 
# 321
size_type _M_check(size_type __pos, const char *__s) const 
# 322
{ 
# 323
if (__pos > this->size()) { 
# 324
__throw_out_of_range(__s); }  
# 325
return __pos; 
# 326
} 
# 329
void _M_check_length(size_type __n1, size_type __n2, const char *__s) const 
# 330
{ 
# 331
if ((this->max_size() - (this->size() - __n1)) < __n2) { 
# 332
__throw_length_error(__s); }  
# 333
} 
# 337
size_type _M_limit(size_type __pos, size_type __off) const 
# 338
{ 
# 339
const bool __testoff = __off < (this->size() - __pos); 
# 340
return __testoff ? __off : (this->size() - __pos); 
# 341
} 
# 345
bool _M_disjunct(const _CharT *__s) const 
# 346
{ 
# 347
return less< const _CharT *> ()(__s, _M_data()) || less< const _CharT *> ()(_M_data() + this->size(), __s); 
# 349
} 
# 354
static void _M_copy(_CharT *__d, const _CharT *__s, size_type __n) 
# 355
{ 
# 356
if (__n == 1) { 
# 357
traits_type::assign(*__d, *__s); } else { 
# 359
traits_type::copy(__d, __s, __n); }  
# 360
} 
# 363
static void _M_move(_CharT *__d, const _CharT *__s, size_type __n) 
# 364
{ 
# 365
if (__n == 1) { 
# 366
traits_type::assign(*__d, *__s); } else { 
# 368
traits_type::move(__d, __s, __n); }  
# 369
} 
# 372
static void _M_assign(_CharT *__d, size_type __n, _CharT __c) 
# 373
{ 
# 374
if (__n == 1) { 
# 375
traits_type::assign(*__d, __c); } else { 
# 377
traits_type::assign(__d, __n, __c); }  
# 378
} 
# 382
template< class _Iterator> static void 
# 384
_S_copy_chars(_CharT *__p, _Iterator __k1, _Iterator __k2) 
# 385
{ 
# 386
for (; __k1 != __k2; (++__k1), (++__p)) { 
# 387
traits_type::assign(*__p, *__k1); }  
# 388
} 
# 391
static void _S_copy_chars(_CharT *__p, iterator __k1, iterator __k2) 
# 392
{ _S_copy_chars(__p, (__k1.base()), (__k2.base())); } 
# 395
static void _S_copy_chars(_CharT *__p, const_iterator __k1, const_iterator __k2) 
# 396
{ _S_copy_chars(__p, (__k1.base()), (__k2.base())); } 
# 399
static void _S_copy_chars(_CharT *__p, _CharT *__k1, _CharT *__k2) 
# 400
{ (_M_copy)(__p, __k1, __k2 - __k1); } 
# 403
static void _S_copy_chars(_CharT *__p, const _CharT *__k1, const _CharT *__k2) 
# 404
{ (_M_copy)(__p, __k1, __k2 - __k1); } 
# 407
static int _S_compare(size_type __n1, size_type __n2) 
# 408
{ 
# 409
const difference_type __d = (difference_type)(__n1 - __n2); 
# 411
if (__d > __gnu_cxx::__numeric_traits< int> ::__max) { 
# 412
return __gnu_cxx::__numeric_traits_integer< int> ::__max; } else { 
# 413
if (__d < __gnu_cxx::__numeric_traits< int> ::__min) { 
# 414
return __gnu_cxx::__numeric_traits_integer< int> ::__min; } else { 
# 416
return (int)__d; }  }  
# 417
} 
# 420
void _M_mutate(size_type __pos, size_type __len1, size_type __len2); 
# 423
void _M_leak_hard(); 
# 426
static _Rep &_S_empty_rep() 
# 427
{ return _Rep::_S_empty_rep(); } 
# 437
public: basic_string() : _M_dataplus(((_S_empty_rep)()._M_refdata()), _Alloc()) 
# 439
{ } 
# 448
explicit basic_string(const _Alloc & __a); 
# 455
basic_string(const basic_string & __str); 
# 462
basic_string(const basic_string & __str, size_type __pos, size_type __n = npos); 
# 471
basic_string(const basic_string & __str, size_type __pos, size_type __n, const _Alloc & __a); 
# 483
basic_string(const _CharT * __s, size_type __n, const _Alloc & __a = _Alloc()); 
# 490
basic_string(const _CharT * __s, const _Alloc & __a = _Alloc()); 
# 497
basic_string(size_type __n, _CharT __c, const _Alloc & __a = _Alloc()); 
# 507
basic_string(basic_string &&__str) noexcept : _M_dataplus(__str._M_dataplus) 
# 509
{ 
# 511
(__str._M_data(((_S_empty_rep)()._M_refdata()))); 
# 515
} 
# 522
basic_string(initializer_list< _CharT>  __l, const _Alloc & __a = _Alloc()); 
# 531
template< class _InputIterator> basic_string(_InputIterator __beg, _InputIterator __end, const _Alloc & __a = _Alloc()); 
# 538
~basic_string() noexcept 
# 539
{ (_M_rep()->_M_dispose(this->get_allocator())); } 
# 546
basic_string &operator=(const basic_string &__str) 
# 547
{ return (this->assign(__str)); } 
# 554
basic_string &operator=(const _CharT *__s) 
# 555
{ return (this->assign(__s)); } 
# 565
basic_string &operator=(_CharT __c) 
# 566
{ 
# 567
(this->assign(1, __c)); 
# 568
return *this; 
# 569
} 
# 580
basic_string &operator=(basic_string &&__str) 
# 581
{ 
# 583
this->swap(__str); 
# 584
return *this; 
# 585
} 
# 592
basic_string &operator=(initializer_list< _CharT>  __l) 
# 593
{ 
# 594
(this->assign((__l.begin()), (__l.size()))); 
# 595
return *this; 
# 596
} 
# 605
iterator begin() noexcept 
# 606
{ 
# 607
_M_leak(); 
# 608
return ((iterator)(_M_data())); 
# 609
} 
# 616
const_iterator begin() const noexcept 
# 617
{ return ((const_iterator)(_M_data())); } 
# 624
iterator end() noexcept 
# 625
{ 
# 626
_M_leak(); 
# 627
return ((iterator)(_M_data() + this->size())); 
# 628
} 
# 635
const_iterator end() const noexcept 
# 636
{ return ((const_iterator)(_M_data() + this->size())); } 
# 644
reverse_iterator rbegin() noexcept 
# 645
{ return ((reverse_iterator)(this->end())); } 
# 653
const_reverse_iterator rbegin() const noexcept 
# 654
{ return ((const_reverse_iterator)(this->end())); } 
# 662
reverse_iterator rend() noexcept 
# 663
{ return ((reverse_iterator)(this->begin())); } 
# 671
const_reverse_iterator rend() const noexcept 
# 672
{ return ((const_reverse_iterator)(this->begin())); } 
# 680
const_iterator cbegin() const noexcept 
# 681
{ return ((const_iterator)((this->_M_data()))); } 
# 688
const_iterator cend() const noexcept 
# 689
{ return ((const_iterator)((this->_M_data()) + this->size())); } 
# 697
const_reverse_iterator crbegin() const noexcept 
# 698
{ return ((const_reverse_iterator)(this->end())); } 
# 706
const_reverse_iterator crend() const noexcept 
# 707
{ return ((const_reverse_iterator)(this->begin())); } 
# 715
size_type size() const noexcept 
# 716
{ return _M_rep()->_M_length; } 
# 721
size_type length() const noexcept 
# 722
{ return _M_rep()->_M_length; } 
# 726
size_type max_size() const noexcept 
# 727
{ return _Rep::_S_max_size; } 
# 740
void resize(size_type __n, _CharT __c); 
# 753
void resize(size_type __n) 
# 754
{ (this->resize(__n, _CharT())); } 
# 759
void shrink_to_fit() 
# 760
{ 
# 761
if (capacity() > size()) 
# 762
{ 
# 763
try 
# 764
{ reserve(0); } 
# 765
catch (...) 
# 766
{ }  
# 767
}  
# 768
} 
# 776
size_type capacity() const noexcept 
# 777
{ return _M_rep()->_M_capacity; } 
# 797
void reserve(size_type __res_arg = 0); 
# 803
void clear() noexcept 
# 804
{ _M_mutate(0, this->size(), 0); } 
# 811
bool empty() const noexcept 
# 812
{ return this->size() == 0; } 
# 826
const_reference operator[](size_type __pos) const 
# 827
{ 
# 828
; 
# 829
return _M_data()[__pos]; 
# 830
} 
# 843
reference operator[](size_type __pos) 
# 844
{ 
# 846
; 
# 848
; 
# 849
_M_leak(); 
# 850
return _M_data()[__pos]; 
# 851
} 
# 864
const_reference at(size_type __n) const 
# 865
{ 
# 866
if (__n >= this->size()) { 
# 867
__throw_out_of_range("basic_string::at"); }  
# 868
return _M_data()[__n]; 
# 869
} 
# 883
reference at(size_type __n) 
# 884
{ 
# 885
if (__n >= size()) { 
# 886
__throw_out_of_range("basic_string::at"); }  
# 887
_M_leak(); 
# 888
return _M_data()[__n]; 
# 889
} 
# 897
reference front() 
# 898
{ return operator[](0); } 
# 905
const_reference front() const 
# 906
{ return operator[](0); } 
# 913
reference back() 
# 914
{ return operator[](this->size() - 1); } 
# 921
const_reference back() const 
# 922
{ return operator[](this->size() - 1); } 
# 932
basic_string &operator+=(const basic_string &__str) 
# 933
{ return (this->append(__str)); } 
# 941
basic_string &operator+=(const _CharT *__s) 
# 942
{ return (this->append(__s)); } 
# 950
basic_string &operator+=(_CharT __c) 
# 951
{ 
# 952
this->push_back(__c); 
# 953
return *this; 
# 954
} 
# 963
basic_string &operator+=(initializer_list< _CharT>  __l) 
# 964
{ return (this->append((__l.begin()), (__l.size()))); } 
# 973
basic_string &append(const basic_string & __str); 
# 989
basic_string &append(const basic_string & __str, size_type __pos, size_type __n); 
# 998
basic_string &append(const _CharT * __s, size_type __n); 
# 1006
basic_string &append(const _CharT *__s) 
# 1007
{ 
# 1008
; 
# 1009
return (this->append(__s, traits_type::length(__s))); 
# 1010
} 
# 1021
basic_string &append(size_type __n, _CharT __c); 
# 1030
basic_string &append(initializer_list< _CharT>  __l) 
# 1031
{ return (this->append((__l.begin()), (__l.size()))); } 
# 1042
template< class _InputIterator> basic_string &
# 1044
append(_InputIterator __first, _InputIterator __last) 
# 1045
{ return (this->replace(_M_iend(), _M_iend(), __first, __last)); } 
# 1052
void push_back(_CharT __c) 
# 1053
{ 
# 1054
const size_type __len = 1 + this->size(); 
# 1055
if ((__len > this->capacity()) || (_M_rep()->_M_is_shared())) { 
# 1056
this->reserve(__len); }  
# 1057
traits_type::assign(_M_data()[this->size()], __c); 
# 1058
(_M_rep()->_M_set_length_and_sharable(__len)); 
# 1059
} 
# 1067
basic_string &assign(const basic_string & __str); 
# 1079
basic_string &assign(basic_string &&__str) 
# 1080
{ 
# 1081
this->swap(__str); 
# 1082
return *this; 
# 1083
} 
# 1100
basic_string &assign(const basic_string &__str, size_type __pos, size_type __n) 
# 1101
{ return (this->assign((__str._M_data()) + __str._M_check(__pos, "basic_string::assign"), __str._M_limit(__pos, __n))); 
# 1103
} 
# 1116
basic_string &assign(const _CharT * __s, size_type __n); 
# 1128
basic_string &assign(const _CharT *__s) 
# 1129
{ 
# 1130
; 
# 1131
return (this->assign(__s, traits_type::length(__s))); 
# 1132
} 
# 1144
basic_string &assign(size_type __n, _CharT __c) 
# 1145
{ return _M_replace_aux((size_type)0, this->size(), __n, __c); } 
# 1155
template< class _InputIterator> basic_string &
# 1157
assign(_InputIterator __first, _InputIterator __last) 
# 1158
{ return (this->replace(_M_ibegin(), _M_iend(), __first, __last)); } 
# 1167
basic_string &assign(initializer_list< _CharT>  __l) 
# 1168
{ return (this->assign((__l.begin()), (__l.size()))); } 
# 1185
void insert(iterator __p, size_type __n, _CharT __c) 
# 1186
{ (this->replace(__p, __p, __n, __c)); } 
# 1200
template< class _InputIterator> void 
# 1202
insert(iterator __p, _InputIterator __beg, _InputIterator __end) 
# 1203
{ (this->replace(__p, __p, __beg, __end)); } 
# 1213
void insert(iterator __p, initializer_list< _CharT>  __l) 
# 1214
{ 
# 1215
; 
# 1216
(this->insert(__p - _M_ibegin(), (__l.begin()), (__l.size()))); 
# 1217
} 
# 1233
basic_string &insert(size_type __pos1, const basic_string &__str) 
# 1234
{ return (this->insert(__pos1, __str, (size_type)0, __str.size())); } 
# 1255
basic_string &insert(size_type __pos1, const basic_string &__str, size_type 
# 1256
__pos2, size_type __n) 
# 1257
{ return (this->insert(__pos1, (__str._M_data()) + __str._M_check(__pos2, "basic_string::insert"), __str._M_limit(__pos2, __n))); 
# 1259
} 
# 1278
basic_string &insert(size_type __pos, const _CharT * __s, size_type __n); 
# 1296
basic_string &insert(size_type __pos, const _CharT *__s) 
# 1297
{ 
# 1298
; 
# 1299
return (this->insert(__pos, __s, traits_type::length(__s))); 
# 1300
} 
# 1319
basic_string &insert(size_type __pos, size_type __n, _CharT __c) 
# 1320
{ return _M_replace_aux(_M_check(__pos, "basic_string::insert"), (size_type)0, __n, __c); 
# 1321
} 
# 1337
iterator insert(iterator __p, _CharT __c) 
# 1338
{ 
# 1339
; 
# 1340
const size_type __pos = __p - _M_ibegin(); 
# 1341
_M_replace_aux(__pos, (size_type)0, (size_type)1, __c); 
# 1342
(_M_rep()->_M_set_leaked()); 
# 1343
return ((iterator)(_M_data() + __pos)); 
# 1344
} 
# 1362
basic_string &erase(size_type __pos = 0, size_type __n = npos) 
# 1363
{ 
# 1364
_M_mutate(_M_check(__pos, "basic_string::erase"), _M_limit(__pos, __n), (size_type)0); 
# 1366
return *this; 
# 1367
} 
# 1378
iterator erase(iterator __position) 
# 1379
{ 
# 1381
; 
# 1382
const size_type __pos = __position - _M_ibegin(); 
# 1383
_M_mutate(__pos, (size_type)1, (size_type)0); 
# 1384
(_M_rep()->_M_set_leaked()); 
# 1385
return ((iterator)(_M_data() + __pos)); 
# 1386
} 
# 1398
iterator erase(iterator __first, iterator __last); 
# 1407
void pop_back() 
# 1408
{ erase(size() - 1, 1); } 
# 1429
basic_string &replace(size_type __pos, size_type __n, const basic_string &__str) 
# 1430
{ return (this->replace(__pos, __n, (__str._M_data()), __str.size())); } 
# 1451
basic_string &replace(size_type __pos1, size_type __n1, const basic_string &__str, size_type 
# 1452
__pos2, size_type __n2) 
# 1453
{ return (this->replace(__pos1, __n1, (__str._M_data()) + __str._M_check(__pos2, "basic_string::replace"), __str._M_limit(__pos2, __n2))); 
# 1455
} 
# 1476
basic_string &replace(size_type __pos, size_type __n1, const _CharT * __s, size_type __n2); 
# 1496
basic_string &replace(size_type __pos, size_type __n1, const _CharT *__s) 
# 1497
{ 
# 1498
; 
# 1499
return (this->replace(__pos, __n1, __s, traits_type::length(__s))); 
# 1500
} 
# 1520
basic_string &replace(size_type __pos, size_type __n1, size_type __n2, _CharT __c) 
# 1521
{ return _M_replace_aux(_M_check(__pos, "basic_string::replace"), _M_limit(__pos, __n1), __n2, __c); 
# 1522
} 
# 1538
basic_string &replace(iterator __i1, iterator __i2, const basic_string &__str) 
# 1539
{ return (this->replace(__i1, __i2, (__str._M_data()), __str.size())); } 
# 1557
basic_string &replace(iterator __i1, iterator __i2, const _CharT *__s, size_type __n) 
# 1558
{ 
# 1560
; 
# 1561
return (this->replace(__i1 - _M_ibegin(), __i2 - __i1, __s, __n)); 
# 1562
} 
# 1578
basic_string &replace(iterator __i1, iterator __i2, const _CharT *__s) 
# 1579
{ 
# 1580
; 
# 1581
return (this->replace(__i1, __i2, __s, traits_type::length(__s))); 
# 1582
} 
# 1599
basic_string &replace(iterator __i1, iterator __i2, size_type __n, _CharT __c) 
# 1600
{ 
# 1602
; 
# 1603
return _M_replace_aux(__i1 - _M_ibegin(), __i2 - __i1, __n, __c); 
# 1604
} 
# 1621
template< class _InputIterator> basic_string &
# 1623
replace(iterator __i1, iterator __i2, _InputIterator 
# 1624
__k1, _InputIterator __k2) 
# 1625
{ 
# 1627
; 
# 1628
; 
# 1629
typedef typename __is_integer< _InputIterator> ::__type _Integral; 
# 1630
return _M_replace_dispatch(__i1, __i2, __k1, __k2, _Integral()); 
# 1631
} 
# 1636
basic_string &replace(iterator __i1, iterator __i2, _CharT *__k1, _CharT *__k2) 
# 1637
{ 
# 1639
; 
# 1640
; 
# 1641
return (this->replace(__i1 - _M_ibegin(), __i2 - __i1, __k1, __k2 - __k1)); 
# 1643
} 
# 1646
basic_string &replace(iterator __i1, iterator __i2, const _CharT *
# 1647
__k1, const _CharT *__k2) 
# 1648
{ 
# 1650
; 
# 1651
; 
# 1652
return (this->replace(__i1 - _M_ibegin(), __i2 - __i1, __k1, __k2 - __k1)); 
# 1654
} 
# 1657
basic_string &replace(iterator __i1, iterator __i2, iterator __k1, iterator __k2) 
# 1658
{ 
# 1660
; 
# 1661
; 
# 1662
return (this->replace(__i1 - _M_ibegin(), __i2 - __i1, (__k1.base()), __k2 - __k1)); 
# 1664
} 
# 1667
basic_string &replace(iterator __i1, iterator __i2, const_iterator 
# 1668
__k1, const_iterator __k2) 
# 1669
{ 
# 1671
; 
# 1672
; 
# 1673
return (this->replace(__i1 - _M_ibegin(), __i2 - __i1, (__k1.base()), __k2 - __k1)); 
# 1675
} 
# 1692
basic_string &replace(iterator __i1, iterator __i2, initializer_list< _CharT>  
# 1693
__l) 
# 1694
{ return (this->replace(__i1, __i2, (__l.begin()), (__l.end()))); } 
# 1700
private: 
# 1698
template< class _Integer> basic_string &
# 1700
_M_replace_dispatch(iterator __i1, iterator __i2, _Integer __n, _Integer 
# 1701
__val, __true_type) 
# 1702
{ return _M_replace_aux(__i1 - _M_ibegin(), __i2 - __i1, __n, __val); } 
# 1704
template< class _InputIterator> basic_string &_M_replace_dispatch(iterator __i1, iterator __i2, _InputIterator __k1, _InputIterator __k2, __false_type); 
# 1710
basic_string &_M_replace_aux(size_type __pos1, size_type __n1, size_type __n2, _CharT __c); 
# 1714
basic_string &_M_replace_safe(size_type __pos1, size_type __n1, const _CharT * __s, size_type __n2); 
# 1719
template< class _InIterator> static _CharT *
# 1721
_S_construct_aux(_InIterator __beg, _InIterator __end, const _Alloc &
# 1722
__a, __false_type) 
# 1723
{ 
# 1724
typedef typename iterator_traits< _InIterator> ::iterator_category _Tag; 
# 1725
return _S_construct(__beg, __end, __a, _Tag()); 
# 1726
} 
# 1730
template< class _Integer> static _CharT *
# 1732
_S_construct_aux(_Integer __beg, _Integer __end, const _Alloc &
# 1733
__a, __true_type) 
# 1734
{ return (_S_construct_aux_2)(static_cast< size_type>(__beg), __end, __a); 
# 1735
} 
# 1738
static _CharT *_S_construct_aux_2(size_type __req, _CharT __c, const _Alloc &__a) 
# 1739
{ return _S_construct(__req, __c, __a); } 
# 1741
template< class _InIterator> static _CharT *
# 1743
_S_construct(_InIterator __beg, _InIterator __end, const _Alloc &__a) 
# 1744
{ 
# 1745
typedef typename __is_integer< _InIterator> ::__type _Integral; 
# 1746
return _S_construct_aux(__beg, __end, __a, _Integral()); 
# 1747
} 
# 1750
template< class _InIterator> static _CharT *_S_construct(_InIterator __beg, _InIterator __end, const _Alloc & __a, input_iterator_tag); 
# 1757
template< class _FwdIterator> static _CharT *_S_construct(_FwdIterator __beg, _FwdIterator __end, const _Alloc & __a, forward_iterator_tag); 
# 1763
static _CharT *_S_construct(size_type __req, _CharT __c, const _Alloc & __a); 
# 1780
public: size_type copy(_CharT * __s, size_type __n, size_type __pos = 0) const; 
# 1790
void swap(basic_string & __s); 
# 1800
const _CharT *c_str() const noexcept 
# 1801
{ return _M_data(); } 
# 1810
const _CharT *data() const noexcept 
# 1811
{ return _M_data(); } 
# 1817
allocator_type get_allocator() const noexcept 
# 1818
{ return _M_dataplus; } 
# 1833
size_type find(const _CharT * __s, size_type __pos, size_type __n) const; 
# 1846
size_type find(const basic_string &__str, size_type __pos = 0) const noexcept 
# 1848
{ return (this->find(__str.data(), __pos, __str.size())); } 
# 1861
size_type find(const _CharT *__s, size_type __pos = 0) const 
# 1862
{ 
# 1863
; 
# 1864
return (this->find(__s, __pos, traits_type::length(__s))); 
# 1865
} 
# 1878
size_type find(_CharT __c, size_type __pos = 0) const noexcept; 
# 1891
size_type rfind(const basic_string &__str, size_type __pos = npos) const noexcept 
# 1893
{ return (this->rfind(__str.data(), __pos, __str.size())); } 
# 1908
size_type rfind(const _CharT * __s, size_type __pos, size_type __n) const; 
# 1921
size_type rfind(const _CharT *__s, size_type __pos = npos) const 
# 1922
{ 
# 1923
; 
# 1924
return (this->rfind(__s, __pos, traits_type::length(__s))); 
# 1925
} 
# 1938
size_type rfind(_CharT __c, size_type __pos = npos) const noexcept; 
# 1952
size_type find_first_of(const basic_string &__str, size_type __pos = 0) const noexcept 
# 1954
{ return (this->find_first_of(__str.data(), __pos, __str.size())); } 
# 1969
size_type find_first_of(const _CharT * __s, size_type __pos, size_type __n) const; 
# 1982
size_type find_first_of(const _CharT *__s, size_type __pos = 0) const 
# 1983
{ 
# 1984
; 
# 1985
return (this->find_first_of(__s, __pos, traits_type::length(__s))); 
# 1986
} 
# 2001
size_type find_first_of(_CharT __c, size_type __pos = 0) const noexcept 
# 2002
{ return (this->find(__c, __pos)); } 
# 2016
size_type find_last_of(const basic_string &__str, size_type __pos = npos) const noexcept 
# 2018
{ return (this->find_last_of(__str.data(), __pos, __str.size())); } 
# 2033
size_type find_last_of(const _CharT * __s, size_type __pos, size_type __n) const; 
# 2046
size_type find_last_of(const _CharT *__s, size_type __pos = npos) const 
# 2047
{ 
# 2048
; 
# 2049
return (this->find_last_of(__s, __pos, traits_type::length(__s))); 
# 2050
} 
# 2065
size_type find_last_of(_CharT __c, size_type __pos = npos) const noexcept 
# 2066
{ return (this->rfind(__c, __pos)); } 
# 2079
size_type find_first_not_of(const basic_string &__str, size_type __pos = 0) const noexcept 
# 2081
{ return (this->find_first_not_of(__str.data(), __pos, __str.size())); } 
# 2096
size_type find_first_not_of(const _CharT * __s, size_type __pos, size_type __n) const; 
# 2110
size_type find_first_not_of(const _CharT *__s, size_type __pos = 0) const 
# 2111
{ 
# 2112
; 
# 2113
return (this->find_first_not_of(__s, __pos, traits_type::length(__s))); 
# 2114
} 
# 2127
size_type find_first_not_of(_CharT __c, size_type __pos = 0) const noexcept; 
# 2142
size_type find_last_not_of(const basic_string &__str, size_type __pos = npos) const noexcept 
# 2144
{ return (this->find_last_not_of(__str.data(), __pos, __str.size())); } 
# 2159
size_type find_last_not_of(const _CharT * __s, size_type __pos, size_type __n) const; 
# 2173
size_type find_last_not_of(const _CharT *__s, size_type __pos = npos) const 
# 2174
{ 
# 2175
; 
# 2176
return (this->find_last_not_of(__s, __pos, traits_type::length(__s))); 
# 2177
} 
# 2190
size_type find_last_not_of(_CharT __c, size_type __pos = npos) const noexcept; 
# 2206
basic_string substr(size_type __pos = 0, size_type __n = npos) const 
# 2207
{ return basic_string(*this, _M_check(__pos, "basic_string::substr"), __n); 
# 2208
} 
# 2225
int compare(const basic_string &__str) const 
# 2226
{ 
# 2227
const size_type __size = this->size(); 
# 2228
const size_type __osize = __str.size(); 
# 2229
const size_type __len = std::min(__size, __osize); 
# 2231
int __r = traits_type::compare(_M_data(), __str.data(), __len); 
# 2232
if (!__r) { 
# 2233
__r = (_S_compare)(__size, __osize); }  
# 2234
return __r; 
# 2235
} 
# 2257
int compare(size_type __pos, size_type __n, const basic_string & __str) const; 
# 2283
int compare(size_type __pos1, size_type __n1, const basic_string & __str, size_type __pos2, size_type __n2) const; 
# 2301
int compare(const _CharT * __s) const; 
# 2325
int compare(size_type __pos, size_type __n1, const _CharT * __s) const; 
# 2352
int compare(size_type __pos, size_type __n1, const _CharT * __s, size_type __n2) const; 
# 2354
}; 
# 2363
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  
# 2365
operator+(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2366
__rhs) 
# 2367
{ 
# 2368
basic_string< _CharT, _Traits, _Alloc>  __str(__lhs); 
# 2369
(__str.append(__rhs)); 
# 2370
return __str; 
# 2371
} 
# 2379
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  operator+(const _CharT * __lhs, const basic_string< _CharT, _Traits, _Alloc>  & __rhs); 
# 2390
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  operator+(_CharT __lhs, const basic_string< _CharT, _Traits, _Alloc>  & __rhs); 
# 2400
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2402
operator+(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const _CharT *
# 2403
__rhs) 
# 2404
{ 
# 2405
basic_string< _CharT, _Traits, _Alloc>  __str(__lhs); 
# 2406
(__str.append(__rhs)); 
# 2407
return __str; 
# 2408
} 
# 2416
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2418
operator+(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, _CharT __rhs) 
# 2419
{ 
# 2420
typedef basic_string< _CharT, _Traits, _Alloc>  __string_type; 
# 2421
typedef typename basic_string< _CharT, _Traits, _Alloc> ::size_type __size_type; 
# 2422
__string_type __str(__lhs); 
# 2423
(__str.append((__size_type)1, __rhs)); 
# 2424
return __str; 
# 2425
} 
# 2428
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2430
operator+(basic_string< _CharT, _Traits, _Alloc>  &&__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2431
__rhs) 
# 2432
{ return std::move((__lhs.append(__rhs))); } 
# 2434
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2436
operator+(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, basic_string< _CharT, _Traits, _Alloc>  &&
# 2437
__rhs) 
# 2438
{ return std::move((__rhs.insert(0, __lhs))); } 
# 2440
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2442
operator+(basic_string< _CharT, _Traits, _Alloc>  &&__lhs, basic_string< _CharT, _Traits, _Alloc>  &&
# 2443
__rhs) 
# 2444
{ 
# 2445
const auto __size = (__lhs.size()) + (__rhs.size()); 
# 2446
const bool __cond = (__size > (__lhs.capacity())) && (__size <= (__rhs.capacity())); 
# 2448
return __cond ? std::move((__rhs.insert(0, __lhs))) : std::move((__lhs.append(__rhs))); 
# 2450
} 
# 2452
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2454
operator+(const _CharT *__lhs, basic_string< _CharT, _Traits, _Alloc>  &&
# 2455
__rhs) 
# 2456
{ return std::move((__rhs.insert(0, __lhs))); } 
# 2458
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2460
operator+(_CharT __lhs, basic_string< _CharT, _Traits, _Alloc>  &&
# 2461
__rhs) 
# 2462
{ return std::move((__rhs.insert(0, 1, __lhs))); } 
# 2464
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2466
operator+(basic_string< _CharT, _Traits, _Alloc>  &&__lhs, const _CharT *
# 2467
__rhs) 
# 2468
{ return std::move((__lhs.append(__rhs))); } 
# 2470
template< class _CharT, class _Traits, class _Alloc> inline basic_string< _CharT, _Traits, _Alloc>  
# 2472
operator+(basic_string< _CharT, _Traits, _Alloc>  &&__lhs, _CharT 
# 2473
__rhs) 
# 2474
{ return std::move((__lhs.append(1, __rhs))); } 
# 2484
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2486
operator==(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2487
__rhs) 
# 2488
{ return (__lhs.compare(__rhs)) == 0; } 
# 2490
template< class _CharT> inline typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, bool> ::__type 
# 2493
operator==(const basic_string< _CharT, char_traits< _CharT> , allocator< _CharT> >  &__lhs, const basic_string< _CharT, char_traits< _CharT> , allocator< _CharT> >  &
# 2494
__rhs) 
# 2495
{ return ((__lhs.size()) == (__rhs.size())) && (!std::char_traits< _CharT> ::compare((__lhs.data()), (__rhs.data()), (__lhs.size()))); 
# 2497
} 
# 2505
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2507
operator==(const _CharT *__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2508
__rhs) 
# 2509
{ return (__rhs.compare(__lhs)) == 0; } 
# 2517
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2519
operator==(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const _CharT *
# 2520
__rhs) 
# 2521
{ return (__lhs.compare(__rhs)) == 0; } 
# 2530
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2532
operator!=(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2533
__rhs) 
# 2534
{ return !(__lhs == __rhs); } 
# 2542
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2544
operator!=(const _CharT *__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2545
__rhs) 
# 2546
{ return !(__lhs == __rhs); } 
# 2554
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2556
operator!=(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const _CharT *
# 2557
__rhs) 
# 2558
{ return !(__lhs == __rhs); } 
# 2567
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2569
operator<(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2570
__rhs) 
# 2571
{ return (__lhs.compare(__rhs)) < 0; } 
# 2579
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2581
operator<(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const _CharT *
# 2582
__rhs) 
# 2583
{ return (__lhs.compare(__rhs)) < 0; } 
# 2591
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2593
operator<(const _CharT *__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2594
__rhs) 
# 2595
{ return (__rhs.compare(__lhs)) > 0; } 
# 2604
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2606
operator>(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2607
__rhs) 
# 2608
{ return (__lhs.compare(__rhs)) > 0; } 
# 2616
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2618
operator>(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const _CharT *
# 2619
__rhs) 
# 2620
{ return (__lhs.compare(__rhs)) > 0; } 
# 2628
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2630
operator>(const _CharT *__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2631
__rhs) 
# 2632
{ return (__rhs.compare(__lhs)) < 0; } 
# 2641
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2643
operator<=(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2644
__rhs) 
# 2645
{ return (__lhs.compare(__rhs)) <= 0; } 
# 2653
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2655
operator<=(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const _CharT *
# 2656
__rhs) 
# 2657
{ return (__lhs.compare(__rhs)) <= 0; } 
# 2665
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2667
operator<=(const _CharT *__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2668
__rhs) 
# 2669
{ return (__rhs.compare(__lhs)) >= 0; } 
# 2678
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2680
operator>=(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2681
__rhs) 
# 2682
{ return (__lhs.compare(__rhs)) >= 0; } 
# 2690
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2692
operator>=(const basic_string< _CharT, _Traits, _Alloc>  &__lhs, const _CharT *
# 2693
__rhs) 
# 2694
{ return (__lhs.compare(__rhs)) >= 0; } 
# 2702
template< class _CharT, class _Traits, class _Alloc> inline bool 
# 2704
operator>=(const _CharT *__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 2705
__rhs) 
# 2706
{ return (__rhs.compare(__lhs)) <= 0; } 
# 2715
template< class _CharT, class _Traits, class _Alloc> inline void 
# 2717
swap(basic_string< _CharT, _Traits, _Alloc>  &__lhs, basic_string< _CharT, _Traits, _Alloc>  &
# 2718
__rhs) 
# 2719
{ (__lhs.swap(__rhs)); } 
# 2733
template< class _CharT, class _Traits, class _Alloc> basic_istream< _CharT, _Traits>  &operator>>(basic_istream< _CharT, _Traits>  & __is, basic_string< _CharT, _Traits, _Alloc>  & __str); 
# 2740
template<> basic_istream< char>  &operator>>(basic_istream< char>  & __is, basic_string< char, char_traits< char> , allocator< char> >  & __str); 
# 2751
template< class _CharT, class _Traits, class _Alloc> inline basic_ostream< _CharT, _Traits>  &
# 2753
operator<<(basic_ostream< _CharT, _Traits>  &__os, const basic_string< _CharT, _Traits, _Alloc>  &
# 2754
__str) 
# 2755
{ 
# 2758
return __ostream_insert(__os, (__str.data()), (__str.size())); 
# 2759
} 
# 2774
template< class _CharT, class _Traits, class _Alloc> basic_istream< _CharT, _Traits>  &getline(basic_istream< _CharT, _Traits>  & __is, basic_string< _CharT, _Traits, _Alloc>  & __str, _CharT __delim); 
# 2791
template< class _CharT, class _Traits, class _Alloc> inline basic_istream< _CharT, _Traits>  &
# 2793
getline(basic_istream< _CharT, _Traits>  &__is, basic_string< _CharT, _Traits, _Alloc>  &
# 2794
__str) 
# 2795
{ return getline(__is, __str, (__is.widen('\n'))); } 
# 2799
template<> basic_istream< char>  &getline(basic_istream< char>  & __in, basic_string< char, char_traits< char> , allocator< char> >  & __str, char __delim); 
# 2805
template<> basic_istream< wchar_t>  &getline(basic_istream< wchar_t>  & __in, basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> >  & __str, wchar_t __delim); 
# 2810
}
# 29 "/usr/include/stdio.h" 3
extern "C" {
# 25 "/usr/include/_G_config.h" 3
typedef 
# 22
struct { 
# 23
__off_t __pos; 
# 24
__mbstate_t __state; 
# 25
} _G_fpos_t; 
# 30
typedef 
# 27
struct { 
# 28
__off64_t __pos; 
# 29
__mbstate_t __state; 
# 30
} _G_fpos64_t; 
# 144 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE; 
# 150
typedef void _IO_lock_t; 
# 156
struct _IO_marker { 
# 157
_IO_marker *_next; 
# 158
_IO_FILE *_sbuf; 
# 162
int _pos; 
# 173
}; 
# 176
enum __codecvt_result { 
# 178
__codecvt_ok, 
# 179
__codecvt_partial, 
# 180
__codecvt_error, 
# 181
__codecvt_noconv
# 182
}; 
# 241
struct _IO_FILE { 
# 242
int _flags; 
# 247
char *_IO_read_ptr; 
# 248
char *_IO_read_end; 
# 249
char *_IO_read_base; 
# 250
char *_IO_write_base; 
# 251
char *_IO_write_ptr; 
# 252
char *_IO_write_end; 
# 253
char *_IO_buf_base; 
# 254
char *_IO_buf_end; 
# 256
char *_IO_save_base; 
# 257
char *_IO_backup_base; 
# 258
char *_IO_save_end; 
# 260
_IO_marker *_markers; 
# 262
_IO_FILE *_chain; 
# 264
int _fileno; 
# 268
int _flags2; 
# 270
__off_t _old_offset; 
# 274
unsigned short _cur_column; 
# 275
signed char _vtable_offset; 
# 276
char _shortbuf[1]; 
# 280
_IO_lock_t *_lock; 
# 289
__off64_t _offset; 
# 297
void *__pad1; 
# 298
void *__pad2; 
# 299
void *__pad3; 
# 300
void *__pad4; 
# 302
size_t __pad5; 
# 303
int _mode; 
# 305
char _unused2[(((15) * sizeof(int)) - ((4) * sizeof(void *))) - sizeof(size_t)]; 
# 307
}; 
# 313
struct _IO_FILE_plus; 
# 315
extern _IO_FILE_plus _IO_2_1_stdin_; 
# 316
extern _IO_FILE_plus _IO_2_1_stdout_; 
# 317
extern _IO_FILE_plus _IO_2_1_stderr_; 
# 333
typedef __ssize_t __io_read_fn(void * __cookie, char * __buf, size_t __nbytes); 
# 341
typedef __ssize_t __io_write_fn(void * __cookie, const char * __buf, size_t __n); 
# 350
typedef int __io_seek_fn(void * __cookie, __off64_t * __pos, int __w); 
# 353
typedef int __io_close_fn(void * __cookie); 
# 358
typedef __io_read_fn cookie_read_function_t; 
# 359
typedef __io_write_fn cookie_write_function_t; 
# 360
typedef __io_seek_fn cookie_seek_function_t; 
# 361
typedef __io_close_fn cookie_close_function_t; 
# 370
typedef 
# 365
struct { 
# 366
__io_read_fn *read; 
# 367
__io_write_fn *write; 
# 368
__io_seek_fn *seek; 
# 369
__io_close_fn *close; 
# 370
} _IO_cookie_io_functions_t; 
# 371
typedef _IO_cookie_io_functions_t cookie_io_functions_t; 
# 373
struct _IO_cookie_file; 
# 376
extern void _IO_cookie_init(_IO_cookie_file * __cfile, int __read_write, void * __cookie, _IO_cookie_io_functions_t __fns); 
# 382
extern "C" {
# 385
extern int __underflow(_IO_FILE *); 
# 386
extern int __uflow(_IO_FILE *); 
# 387
extern int __overflow(_IO_FILE *, int); 
# 429
extern int _IO_getc(_IO_FILE * __fp); 
# 430
extern int _IO_putc(int __c, _IO_FILE * __fp); 
# 431
extern int _IO_feof(_IO_FILE * __fp) throw(); 
# 432
extern int _IO_ferror(_IO_FILE * __fp) throw(); 
# 434
extern int _IO_peekc_locked(_IO_FILE * __fp); 
# 440
extern void _IO_flockfile(_IO_FILE *) throw(); 
# 441
extern void _IO_funlockfile(_IO_FILE *) throw(); 
# 442
extern int _IO_ftrylockfile(_IO_FILE *) throw(); 
# 459
extern int _IO_vfscanf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list, int *__restrict__); 
# 461
extern int _IO_vfprintf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list); 
# 463
extern __ssize_t _IO_padn(_IO_FILE *, int, __ssize_t); 
# 464
extern size_t _IO_sgetn(_IO_FILE *, void *, size_t); 
# 466
extern __off64_t _IO_seekoff(_IO_FILE *, __off64_t, int, int); 
# 467
extern __off64_t _IO_seekpos(_IO_FILE *, __off64_t, int); 
# 469
extern void _IO_free_backup_area(_IO_FILE *) throw(); 
# 521
}
# 79 "/usr/include/stdio.h" 3
typedef __gnuc_va_list va_list; 
# 110
typedef _G_fpos_t fpos_t; 
# 116
typedef _G_fpos64_t fpos64_t; 
# 168
extern _IO_FILE *stdin; 
# 169
extern _IO_FILE *stdout; 
# 170
extern _IO_FILE *stderr; 
# 178
extern int remove(const char * __filename) throw(); 
# 180
extern int rename(const char * __old, const char * __new) throw(); 
# 185
extern int renameat(int __oldfd, const char * __old, int __newfd, const char * __new) throw(); 
# 195
extern FILE *tmpfile(); 
# 205
extern FILE *tmpfile64(); 
# 209
extern char *tmpnam(char * __s) throw(); 
# 215
extern char *tmpnam_r(char * __s) throw(); 
# 227
extern char *tempnam(const char * __dir, const char * __pfx) throw()
# 228
 __attribute((__malloc__)); 
# 237
extern int fclose(FILE * __stream); 
# 242
extern int fflush(FILE * __stream); 
# 252
extern int fflush_unlocked(FILE * __stream); 
# 262
extern int fcloseall(); 
# 272
extern FILE *fopen(const char *__restrict__ __filename, const char *__restrict__ __modes); 
# 278
extern FILE *freopen(const char *__restrict__ __filename, const char *__restrict__ __modes, FILE *__restrict__ __stream); 
# 297
extern FILE *fopen64(const char *__restrict__ __filename, const char *__restrict__ __modes); 
# 299
extern FILE *freopen64(const char *__restrict__ __filename, const char *__restrict__ __modes, FILE *__restrict__ __stream); 
# 306
extern FILE *fdopen(int __fd, const char * __modes) throw(); 
# 312
extern FILE *fopencookie(void *__restrict__ __magic_cookie, const char *__restrict__ __modes, _IO_cookie_io_functions_t __io_funcs) throw(); 
# 319
extern FILE *fmemopen(void * __s, size_t __len, const char * __modes) throw(); 
# 325
extern FILE *open_memstream(char ** __bufloc, size_t * __sizeloc) throw(); 
# 332
extern void setbuf(FILE *__restrict__ __stream, char *__restrict__ __buf) throw(); 
# 336
extern int setvbuf(FILE *__restrict__ __stream, char *__restrict__ __buf, int __modes, size_t __n) throw(); 
# 343
extern void setbuffer(FILE *__restrict__ __stream, char *__restrict__ __buf, size_t __size) throw(); 
# 347
extern void setlinebuf(FILE * __stream) throw(); 
# 356
extern int fprintf(FILE *__restrict__ __stream, const char *__restrict__ __format, ...); 
# 362
extern int printf(const char *__restrict__ __format, ...); 
# 364
extern int sprintf(char *__restrict__ __s, const char *__restrict__ __format, ...) throw(); 
# 371
extern int vfprintf(FILE *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg); 
# 377
extern int vprintf(const char *__restrict__ __format, __gnuc_va_list __arg); 
# 379
extern int vsprintf(char *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg) throw(); 
# 386
extern int snprintf(char *__restrict__ __s, size_t __maxlen, const char *__restrict__ __format, ...) throw()
# 388
 __attribute((__format__(__printf__, 3, 4))); 
# 390
extern int vsnprintf(char *__restrict__ __s, size_t __maxlen, const char *__restrict__ __format, __gnuc_va_list __arg) throw()
# 392
 __attribute((__format__(__printf__, 3, 0))); 
# 399
extern int vasprintf(char **__restrict__ __ptr, const char *__restrict__ __f, __gnuc_va_list __arg) throw()
# 401
 __attribute((__format__(__printf__, 2, 0))); 
# 402
extern int __asprintf(char **__restrict__ __ptr, const char *__restrict__ __fmt, ...) throw()
# 404
 __attribute((__format__(__printf__, 2, 3))); 
# 405
extern int asprintf(char **__restrict__ __ptr, const char *__restrict__ __fmt, ...) throw()
# 407
 __attribute((__format__(__printf__, 2, 3))); 
# 412
extern int vdprintf(int __fd, const char *__restrict__ __fmt, __gnuc_va_list __arg)
# 414
 __attribute((__format__(__printf__, 2, 0))); 
# 415
extern int dprintf(int __fd, const char *__restrict__ __fmt, ...)
# 416
 __attribute((__format__(__printf__, 2, 3))); 
# 425
extern int fscanf(FILE *__restrict__ __stream, const char *__restrict__ __format, ...); 
# 431
extern int scanf(const char *__restrict__ __format, ...); 
# 433
extern int sscanf(const char *__restrict__ __s, const char *__restrict__ __format, ...) throw(); 
# 471
extern int vfscanf(FILE *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg)
# 473
 __attribute((__format__(__scanf__, 2, 0))); 
# 479
extern int vscanf(const char *__restrict__ __format, __gnuc_va_list __arg)
# 480
 __attribute((__format__(__scanf__, 1, 0))); 
# 483
extern int vsscanf(const char *__restrict__ __s, const char *__restrict__ __format, __gnuc_va_list __arg) throw()
# 485
 __attribute((__format__(__scanf__, 2, 0))); 
# 531
extern int fgetc(FILE * __stream); 
# 532
extern int getc(FILE * __stream); 
# 538
extern int getchar(); 
# 550
extern int getc_unlocked(FILE * __stream); 
# 551
extern int getchar_unlocked(); 
# 561
extern int fgetc_unlocked(FILE * __stream); 
# 573
extern int fputc(int __c, FILE * __stream); 
# 574
extern int putc(int __c, FILE * __stream); 
# 580
extern int putchar(int __c); 
# 594
extern int fputc_unlocked(int __c, FILE * __stream); 
# 602
extern int putc_unlocked(int __c, FILE * __stream); 
# 603
extern int putchar_unlocked(int __c); 
# 610
extern int getw(FILE * __stream); 
# 613
extern int putw(int __w, FILE * __stream); 
# 622
extern char *fgets(char *__restrict__ __s, int __n, FILE *__restrict__ __stream); 
# 638
extern char *gets(char * __s) __attribute((__deprecated__)); 
# 649
extern char *fgets_unlocked(char *__restrict__ __s, int __n, FILE *__restrict__ __stream); 
# 665
extern __ssize_t __getdelim(char **__restrict__ __lineptr, size_t *__restrict__ __n, int __delimiter, FILE *__restrict__ __stream); 
# 668
extern __ssize_t getdelim(char **__restrict__ __lineptr, size_t *__restrict__ __n, int __delimiter, FILE *__restrict__ __stream); 
# 678
extern __ssize_t getline(char **__restrict__ __lineptr, size_t *__restrict__ __n, FILE *__restrict__ __stream); 
# 689
extern int fputs(const char *__restrict__ __s, FILE *__restrict__ __stream); 
# 695
extern int puts(const char * __s); 
# 702
extern int ungetc(int __c, FILE * __stream); 
# 709
extern size_t fread(void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __stream); 
# 715
extern size_t fwrite(const void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __s); 
# 726
extern int fputs_unlocked(const char *__restrict__ __s, FILE *__restrict__ __stream); 
# 737
extern size_t fread_unlocked(void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __stream); 
# 739
extern size_t fwrite_unlocked(const void *__restrict__ __ptr, size_t __size, size_t __n, FILE *__restrict__ __stream); 
# 749
extern int fseek(FILE * __stream, long __off, int __whence); 
# 754
extern long ftell(FILE * __stream); 
# 759
extern void rewind(FILE * __stream); 
# 773
extern int fseeko(FILE * __stream, __off_t __off, int __whence); 
# 778
extern __off_t ftello(FILE * __stream); 
# 798
extern int fgetpos(FILE *__restrict__ __stream, fpos_t *__restrict__ __pos); 
# 803
extern int fsetpos(FILE * __stream, const fpos_t * __pos); 
# 818
extern int fseeko64(FILE * __stream, __off64_t __off, int __whence); 
# 819
extern __off64_t ftello64(FILE * __stream); 
# 820
extern int fgetpos64(FILE *__restrict__ __stream, fpos64_t *__restrict__ __pos); 
# 821
extern int fsetpos64(FILE * __stream, const fpos64_t * __pos); 
# 826
extern void clearerr(FILE * __stream) throw(); 
# 828
extern int feof(FILE * __stream) throw(); 
# 830
extern int ferror(FILE * __stream) throw(); 
# 835
extern void clearerr_unlocked(FILE * __stream) throw(); 
# 836
extern int feof_unlocked(FILE * __stream) throw(); 
# 837
extern int ferror_unlocked(FILE * __stream) throw(); 
# 846
extern void perror(const char * __s); 
# 26 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3
extern int sys_nerr; 
# 27
extern const char *const sys_errlist[]; 
# 30
extern int _sys_nerr; 
# 31
extern const char *const _sys_errlist[]; 
# 858 "/usr/include/stdio.h" 3
extern int fileno(FILE * __stream) throw(); 
# 863
extern int fileno_unlocked(FILE * __stream) throw(); 
# 872
extern FILE *popen(const char * __command, const char * __modes); 
# 878
extern int pclose(FILE * __stream); 
# 884
extern char *ctermid(char * __s) throw(); 
# 890
extern char *cuserid(char * __s); 
# 895
struct obstack; 
# 898
extern int obstack_printf(obstack *__restrict__ __obstack, const char *__restrict__ __format, ...) throw()
# 900
 __attribute((__format__(__printf__, 2, 3))); 
# 901
extern int obstack_vprintf(obstack *__restrict__ __obstack, const char *__restrict__ __format, __gnuc_va_list __args) throw()
# 904
 __attribute((__format__(__printf__, 2, 0))); 
# 912
extern void flockfile(FILE * __stream) throw(); 
# 916
extern int ftrylockfile(FILE * __stream) throw(); 
# 919
extern void funlockfile(FILE * __stream) throw(); 
# 942
}
# 94 "/usr/include/c++/4.8/cstdio" 3
namespace std { 
# 96
using ::FILE;
# 97
using ::fpos_t;
# 99
using ::clearerr;
# 100
using ::fclose;
# 101
using ::feof;
# 102
using ::ferror;
# 103
using ::fflush;
# 104
using ::fgetc;
# 105
using ::fgetpos;
# 106
using ::fgets;
# 107
using ::fopen;
# 108
using ::fprintf;
# 109
using ::fputc;
# 110
using ::fputs;
# 111
using ::fread;
# 112
using ::freopen;
# 113
using ::fscanf;
# 114
using ::fseek;
# 115
using ::fsetpos;
# 116
using ::ftell;
# 117
using ::fwrite;
# 118
using ::getc;
# 119
using ::getchar;
# 120
using ::gets;
# 121
using ::perror;
# 122
using ::printf;
# 123
using ::putc;
# 124
using ::putchar;
# 125
using ::puts;
# 126
using ::remove;
# 127
using ::rename;
# 128
using ::rewind;
# 129
using ::scanf;
# 130
using ::setbuf;
# 131
using ::setvbuf;
# 132
using ::sprintf;
# 133
using ::sscanf;
# 134
using ::tmpfile;
# 135
using ::tmpnam;
# 136
using ::ungetc;
# 137
using ::vfprintf;
# 138
using ::vprintf;
# 139
using ::vsprintf;
# 140
}
# 150
namespace __gnu_cxx { 
# 168
using ::snprintf;
# 169
using ::vfscanf;
# 170
using ::vscanf;
# 171
using ::vsnprintf;
# 172
using ::vsscanf;
# 174
}
# 176
namespace std { 
# 178
using __gnu_cxx::snprintf;
# 179
using __gnu_cxx::vfscanf;
# 180
using __gnu_cxx::vscanf;
# 181
using __gnu_cxx::vsnprintf;
# 182
using __gnu_cxx::vsscanf;
# 183
}
# 31 "/usr/include/errno.h" 3
extern "C" {
# 50 "/usr/include/x86_64-linux-gnu/bits/errno.h" 3
extern int *__errno_location() throw() __attribute((const)); 
# 54 "/usr/include/errno.h" 3
extern char *program_invocation_name, *program_invocation_short_name; 
# 58
}
# 68
typedef int error_t; 
# 46 "/usr/include/c++/4.8/ext/string_conversions.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 51
template< class _TRet, class _Ret = _TRet, class _CharT, class ...
# 52
_Base> _Ret 
# 54
__stoa(_TRet (*__convf)(const _CharT *, _CharT **, _Base ...), const char *
# 55
__name, const _CharT *__str, std::size_t *__idx, _Base ...
# 56
__base) 
# 57
{ 
# 58
_Ret __ret; 
# 60
_CharT *__endptr; 
# 61
(*__errno_location()) = 0; 
# 62
const _TRet __tmp = __convf(__str, &__endptr, __base...); 
# 64
if (__endptr == __str) { 
# 65
std::__throw_invalid_argument(__name); } else { 
# 66
if (((*__errno_location()) == 34) || (std::__are_same< _Ret, int> ::__value && ((__tmp < __numeric_traits< int> ::__min) || (__tmp > __numeric_traits< int> ::__max)))) { 
# 70
std::__throw_out_of_range(__name); } else { 
# 72
__ret = __tmp; }  }  
# 74
if (__idx) { 
# 75
(*__idx) = (__endptr - __str); }  
# 77
return __ret; 
# 78
} 
# 81
template< class _String, class _CharT = typename _String::value_type> _String 
# 83
__to_xstring(int (*__convf)(_CharT *, std::size_t, const _CharT *, __builtin_va_list), std::size_t 
# 84
__n, const _CharT *
# 85
__fmt, ...) 
# 86
{ 
# 89
_CharT *__s = static_cast< _CharT *>(__builtin_alloca(sizeof(_CharT) * __n)); 
# 92
__builtin_va_list __args; 
# 93
__builtin_va_start((__args),__fmt); 
# 95
const int __len = __convf(__s, __n, __fmt, __args); 
# 97
__builtin_va_end(__args); 
# 99
return _String(__s, __s + __len); 
# 100
} 
# 103
}
# 2817 "/usr/include/c++/4.8/bits/basic_string.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 2823
inline int stoi(const string &__str, size_t *__idx = 0, int __base = 10) 
# 2824
{ return __gnu_cxx::__stoa< long, int> (&std::strtol, "stoi", __str.c_str(), __idx, __base); 
# 2825
} 
# 2828
inline long stol(const string &__str, size_t *__idx = 0, int __base = 10) 
# 2829
{ return __gnu_cxx::__stoa(&std::strtol, "stol", __str.c_str(), __idx, __base); 
# 2830
} 
# 2833
inline unsigned long stoul(const string &__str, size_t *__idx = 0, int __base = 10) 
# 2834
{ return __gnu_cxx::__stoa(&std::strtoul, "stoul", __str.c_str(), __idx, __base); 
# 2835
} 
# 2838
inline long long stoll(const string &__str, size_t *__idx = 0, int __base = 10) 
# 2839
{ return __gnu_cxx::__stoa(&std::strtoll, "stoll", __str.c_str(), __idx, __base); 
# 2840
} 
# 2843
inline unsigned long long stoull(const string &__str, size_t *__idx = 0, int __base = 10) 
# 2844
{ return __gnu_cxx::__stoa(&std::strtoull, "stoull", __str.c_str(), __idx, __base); 
# 2845
} 
# 2849
inline float stof(const string &__str, size_t *__idx = 0) 
# 2850
{ return __gnu_cxx::__stoa(&std::strtof, "stof", __str.c_str(), __idx); } 
# 2853
inline double stod(const string &__str, size_t *__idx = 0) 
# 2854
{ return __gnu_cxx::__stoa(&std::strtod, "stod", __str.c_str(), __idx); } 
# 2857
inline long double stold(const string &__str, size_t *__idx = 0) 
# 2858
{ return __gnu_cxx::__stoa(&std::strtold, "stold", __str.c_str(), __idx); } 
# 2864
inline string to_string(int __val) 
# 2865
{ return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, (4) * sizeof(int), "%d", __val); 
# 2866
} 
# 2869
inline string to_string(unsigned __val) 
# 2870
{ return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, (4) * sizeof(unsigned), "%u", __val); 
# 2872
} 
# 2875
inline string to_string(long __val) 
# 2876
{ return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, (4) * sizeof(long), "%ld", __val); 
# 2877
} 
# 2880
inline string to_string(unsigned long __val) 
# 2881
{ return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, (4) * sizeof(unsigned long), "%lu", __val); 
# 2883
} 
# 2886
inline string to_string(long long __val) 
# 2887
{ return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, (4) * sizeof(long long), "%lld", __val); 
# 2889
} 
# 2892
inline string to_string(unsigned long long __val) 
# 2893
{ return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, (4) * sizeof(unsigned long long), "%llu", __val); 
# 2895
} 
# 2898
inline string to_string(float __val) 
# 2899
{ 
# 2900
const int __n = (__gnu_cxx::__numeric_traits_floating< float> ::__max_exponent10 + 20); 
# 2902
return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, __n, "%f", __val); 
# 2904
} 
# 2907
inline string to_string(double __val) 
# 2908
{ 
# 2909
const int __n = (__gnu_cxx::__numeric_traits_floating< double> ::__max_exponent10 + 20); 
# 2911
return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, __n, "%f", __val); 
# 2913
} 
# 2916
inline string to_string(long double __val) 
# 2917
{ 
# 2918
const int __n = (__gnu_cxx::__numeric_traits_floating< long double> ::__max_exponent10 + 20); 
# 2920
return __gnu_cxx::__to_xstring< basic_string< char, char_traits< char> , allocator< char> > > (&std::vsnprintf, __n, "%Lf", __val); 
# 2922
} 
# 2926
inline int stoi(const wstring &__str, size_t *__idx = 0, int __base = 10) 
# 2927
{ return __gnu_cxx::__stoa< long, int> (&std::wcstol, "stoi", __str.c_str(), __idx, __base); 
# 2928
} 
# 2931
inline long stol(const wstring &__str, size_t *__idx = 0, int __base = 10) 
# 2932
{ return __gnu_cxx::__stoa(&std::wcstol, "stol", __str.c_str(), __idx, __base); 
# 2933
} 
# 2936
inline unsigned long stoul(const wstring &__str, size_t *__idx = 0, int __base = 10) 
# 2937
{ return __gnu_cxx::__stoa(&std::wcstoul, "stoul", __str.c_str(), __idx, __base); 
# 2938
} 
# 2941
inline long long stoll(const wstring &__str, size_t *__idx = 0, int __base = 10) 
# 2942
{ return __gnu_cxx::__stoa(&std::wcstoll, "stoll", __str.c_str(), __idx, __base); 
# 2943
} 
# 2946
inline unsigned long long stoull(const wstring &__str, size_t *__idx = 0, int __base = 10) 
# 2947
{ return __gnu_cxx::__stoa(&std::wcstoull, "stoull", __str.c_str(), __idx, __base); 
# 2948
} 
# 2952
inline float stof(const wstring &__str, size_t *__idx = 0) 
# 2953
{ return __gnu_cxx::__stoa(&std::wcstof, "stof", __str.c_str(), __idx); } 
# 2956
inline double stod(const wstring &__str, size_t *__idx = 0) 
# 2957
{ return __gnu_cxx::__stoa(&std::wcstod, "stod", __str.c_str(), __idx); } 
# 2960
inline long double stold(const wstring &__str, size_t *__idx = 0) 
# 2961
{ return __gnu_cxx::__stoa(&std::wcstold, "stold", __str.c_str(), __idx); } 
# 2965
inline wstring to_wstring(int __val) 
# 2966
{ return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, (4) * sizeof(int), L"\x25\x64", __val); 
# 2967
} 
# 2970
inline wstring to_wstring(unsigned __val) 
# 2971
{ return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, (4) * sizeof(unsigned), L"\x25\x75", __val); 
# 2973
} 
# 2976
inline wstring to_wstring(long __val) 
# 2977
{ return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, (4) * sizeof(long), L"\x25\x6c\x64", __val); 
# 2978
} 
# 2981
inline wstring to_wstring(unsigned long __val) 
# 2982
{ return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, (4) * sizeof(unsigned long), L"\x25\x6c\x75", __val); 
# 2984
} 
# 2987
inline wstring to_wstring(long long __val) 
# 2988
{ return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, (4) * sizeof(long long), L"\x25\x6c\x6c\x64", __val); 
# 2990
} 
# 2993
inline wstring to_wstring(unsigned long long __val) 
# 2994
{ return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, (4) * sizeof(unsigned long long), L"\x25\x6c\x6c\x75", __val); 
# 2996
} 
# 2999
inline wstring to_wstring(float __val) 
# 3000
{ 
# 3001
const int __n = (__gnu_cxx::__numeric_traits_floating< float> ::__max_exponent10 + 20); 
# 3003
return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, __n, L"\x25\x66", __val); 
# 3005
} 
# 3008
inline wstring to_wstring(double __val) 
# 3009
{ 
# 3010
const int __n = (__gnu_cxx::__numeric_traits_floating< double> ::__max_exponent10 + 20); 
# 3012
return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, __n, L"\x25\x66", __val); 
# 3014
} 
# 3017
inline wstring to_wstring(long double __val) 
# 3018
{ 
# 3019
const int __n = (__gnu_cxx::__numeric_traits_floating< long double> ::__max_exponent10 + 20); 
# 3021
return __gnu_cxx::__to_xstring< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > (&std::vswprintf, __n, L"\x25\x4c\x66", __val); 
# 3023
} 
# 3027
}
# 37 "/usr/include/c++/4.8/bits/hash_bytes.h" 3
namespace std { 
# 47
size_t _Hash_bytes(const void * __ptr, size_t __len, size_t __seed); 
# 54
size_t _Fnv_hash_bytes(const void * __ptr, size_t __len, size_t __seed); 
# 57
}
# 37 "/usr/include/c++/4.8/bits/functional_hash.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 49
template< class _Result, class _Arg> 
# 50
struct __hash_base { 
# 52
typedef _Result result_type; 
# 53
typedef _Arg argument_type; 
# 54
}; 
# 57
template< class _Tp> struct hash; 
# 61
template< class _Tp> 
# 62
struct hash< _Tp *>  : public __hash_base< unsigned long, _Tp *>  { 
# 65
::std::size_t operator()(_Tp *__p) const noexcept 
# 66
{ return reinterpret_cast< ::std::size_t>(__p); } 
# 67
}; 
# 80
template<> struct hash< bool>  : public __hash_base< unsigned long, bool>  { size_t operator()(bool __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 83
template<> struct hash< char>  : public __hash_base< unsigned long, char>  { size_t operator()(char __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 86
template<> struct hash< signed char>  : public __hash_base< unsigned long, signed char>  { size_t operator()(signed char __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 89
template<> struct hash< unsigned char>  : public __hash_base< unsigned long, unsigned char>  { size_t operator()(unsigned char __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 92
template<> struct hash< wchar_t>  : public __hash_base< unsigned long, wchar_t>  { size_t operator()(wchar_t __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 95
template<> struct hash< char16_t>  : public __hash_base< unsigned long, char16_t>  { size_t operator()(char16_t __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 98
template<> struct hash< char32_t>  : public __hash_base< unsigned long, char32_t>  { size_t operator()(char32_t __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 101
template<> struct hash< short>  : public __hash_base< unsigned long, short>  { size_t operator()(short __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 104
template<> struct hash< int>  : public __hash_base< unsigned long, int>  { size_t operator()(int __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 107
template<> struct hash< long>  : public __hash_base< unsigned long, long>  { size_t operator()(long __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 110
template<> struct hash< long long>  : public __hash_base< unsigned long, long long>  { size_t operator()(long long __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 113
template<> struct hash< unsigned short>  : public __hash_base< unsigned long, unsigned short>  { size_t operator()(unsigned short __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 116
template<> struct hash< unsigned>  : public __hash_base< unsigned long, unsigned>  { size_t operator()(unsigned __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 119
template<> struct hash< unsigned long>  : public __hash_base< unsigned long, unsigned long>  { size_t operator()(unsigned long __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 122
template<> struct hash< unsigned long long>  : public __hash_base< unsigned long, unsigned long long>  { size_t operator()(unsigned long long __val) const noexcept { return static_cast< size_t>(__val); } }; 
# 126
struct _Hash_impl { 
# 129
static size_t hash(const void *__ptr, size_t __clength, size_t 
# 130
__seed = static_cast< size_t>(3339675911UL)) 
# 131
{ return _Hash_bytes(__ptr, __clength, __seed); } 
# 133
template< class _Tp> static size_t 
# 135
hash(const _Tp &__val) 
# 136
{ return hash(&__val, sizeof(__val)); } 
# 138
template< class _Tp> static size_t 
# 140
__hash_combine(const _Tp &__val, size_t __hash) 
# 141
{ return hash(&__val, sizeof(__val), __hash); } 
# 142
}; 
# 144
struct _Fnv_hash_impl { 
# 147
static size_t hash(const void *__ptr, size_t __clength, size_t 
# 148
__seed = static_cast< size_t>(2166136261UL)) 
# 149
{ return _Fnv_hash_bytes(__ptr, __clength, __seed); } 
# 151
template< class _Tp> static size_t 
# 153
hash(const _Tp &__val) 
# 154
{ return hash(&__val, sizeof(__val)); } 
# 156
template< class _Tp> static size_t 
# 158
__hash_combine(const _Tp &__val, size_t __hash) 
# 159
{ return hash(&__val, sizeof(__val), __hash); } 
# 160
}; 
# 164
template<> struct hash< float>  : public __hash_base< unsigned long, float>  { 
# 167
size_t operator()(float __val) const noexcept 
# 168
{ 
# 170
return (__val != (0.0F)) ? std::_Hash_impl::hash(__val) : (0); 
# 171
} 
# 172
}; 
# 176
template<> struct hash< double>  : public __hash_base< unsigned long, double>  { 
# 179
size_t operator()(double __val) const noexcept 
# 180
{ 
# 182
return (__val != (0.0)) ? std::_Hash_impl::hash(__val) : (0); 
# 183
} 
# 184
}; 
# 188
template<> struct hash< long double>  : public __hash_base< unsigned long, long double>  { 
# 191
__attribute((__pure__)) size_t 
# 192
operator()(long double __val) const noexcept; 
# 193
}; 
# 201
template< class _Hash> 
# 202
struct __is_fast_hash : public true_type { 
# 203
}; 
# 206
template<> struct __is_fast_hash< hash< long double> >  : public false_type { 
# 207
}; 
# 210
}
# 3035 "/usr/include/c++/4.8/bits/basic_string.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 3044
template<> struct hash< basic_string< char, char_traits< char> , allocator< char> > >  : public __hash_base< unsigned long, basic_string< char, char_traits< char> , allocator< char> > >  { 
# 3048
size_t operator()(const string &__s) const noexcept 
# 3049
{ return std::_Hash_impl::hash(__s.data(), __s.length()); } 
# 3050
}; 
# 3053
template<> struct __is_fast_hash< hash< basic_string< char, char_traits< char> , allocator< char> > > >  : public false_type { 
# 3054
}; 
# 3059
template<> struct hash< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > >  : public __hash_base< unsigned long, basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > >  { 
# 3063
size_t operator()(const wstring &__s) const noexcept 
# 3064
{ return std::_Hash_impl::hash(__s.data(), __s.length() * sizeof(wchar_t)); 
# 3065
} 
# 3066
}; 
# 3069
template<> struct __is_fast_hash< hash< basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > > >  : public false_type { 
# 3070
}; 
# 3077
template<> struct hash< basic_string< char16_t, char_traits< char16_t> , allocator< char16_t> > >  : public __hash_base< unsigned long, basic_string< char16_t, char_traits< char16_t> , allocator< char16_t> > >  { 
# 3081
size_t operator()(const u16string &__s) const noexcept 
# 3082
{ return std::_Hash_impl::hash(__s.data(), __s.length() * sizeof(char16_t)); 
# 3083
} 
# 3084
}; 
# 3087
template<> struct __is_fast_hash< hash< basic_string< char16_t, char_traits< char16_t> , allocator< char16_t> > > >  : public false_type { 
# 3088
}; 
# 3092
template<> struct hash< basic_string< char32_t, char_traits< char32_t> , allocator< char32_t> > >  : public __hash_base< unsigned long, basic_string< char32_t, char_traits< char32_t> , allocator< char32_t> > >  { 
# 3096
size_t operator()(const u32string &__s) const noexcept 
# 3097
{ return std::_Hash_impl::hash(__s.data(), __s.length() * sizeof(char32_t)); 
# 3098
} 
# 3099
}; 
# 3102
template<> struct __is_fast_hash< hash< basic_string< char32_t, char_traits< char32_t> , allocator< char32_t> > > >  : public false_type { 
# 3103
}; 
# 3107
}
# 44 "/usr/include/c++/4.8/bits/basic_string.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 48
template< class _CharT, class _Traits, class _Alloc> const typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 51
basic_string< _CharT, _Traits, _Alloc> ::_Rep::_S_max_size = (((npos - sizeof(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::_Rep_base)) / sizeof(_CharT)) - 1) / 4; 
# 53
template< class _CharT, class _Traits, class _Alloc> const _CharT 
# 56
basic_string< _CharT, _Traits, _Alloc> ::_Rep::_S_terminal = (_CharT()); 
# 58
template< class _CharT, class _Traits, class _Alloc> const typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 60
basic_string< _CharT, _Traits, _Alloc> ::npos; 
# 64
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 66
basic_string< _CharT, _Traits, _Alloc> ::_Rep::_S_empty_rep_storage[(((sizeof(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::_Rep_base) + sizeof(_CharT)) + sizeof(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type)) - (1)) / sizeof(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type)]; 
# 74
template< class _CharT, class _Traits, class _Alloc> 
# 75
template< class _InIterator> _CharT *
# 78
basic_string< _CharT, _Traits, _Alloc> ::_S_construct(_InIterator __beg, _InIterator __end, const _Alloc &__a, input_iterator_tag) 
# 80
{ 
# 82
if ((__beg == __end) && (__a == _Alloc())) { 
# 83
return ((_S_empty_rep)()._M_refdata()); }  
# 86
_CharT __buf[128]; 
# 87
size_type __len = (0); 
# 88
while ((__beg != __end) && (__len < (sizeof(__buf) / sizeof(_CharT)))) 
# 89
{ 
# 90
(__buf[__len++]) = (*__beg); 
# 91
++__beg; 
# 92
}  
# 93
_Rep *__r = _Rep::_S_create(__len, (size_type)0, __a); 
# 94
(_M_copy)((__r->_M_refdata()), __buf, __len); 
# 95
try 
# 96
{ 
# 97
while (__beg != __end) 
# 98
{ 
# 99
if (__len == (__r->_M_capacity)) 
# 100
{ 
# 102
_Rep *__another = _Rep::_S_create(__len + 1, __len, __a); 
# 103
(_M_copy)((__another->_M_refdata()), (__r->_M_refdata()), __len); 
# 104
(__r->_M_destroy(__a)); 
# 105
__r = __another; 
# 106
}  
# 107
((__r->_M_refdata())[__len++]) = (*__beg); 
# 108
++__beg; 
# 109
}  
# 110
} 
# 111
catch (...) 
# 112
{ 
# 113
(__r->_M_destroy(__a)); 
# 114
throw; 
# 115
}  
# 116
(__r->_M_set_length_and_sharable(__len)); 
# 117
return (__r->_M_refdata()); 
# 118
} 
# 120
template< class _CharT, class _Traits, class _Alloc> 
# 121
template< class _InIterator> _CharT *
# 124
basic_string< _CharT, _Traits, _Alloc> ::_S_construct(_InIterator __beg, _InIterator __end, const _Alloc &__a, forward_iterator_tag) 
# 126
{ 
# 128
if ((__beg == __end) && (__a == _Alloc())) { 
# 129
return ((_S_empty_rep)()._M_refdata()); }  
# 132
if (__gnu_cxx::__is_null_pointer(__beg) && (__beg != __end)) { 
# 133
__throw_logic_error("basic_string::_S_construct null not valid"); }  
# 135
const size_type __dnew = static_cast< size_type>(std::distance(__beg, __end)); 
# 138
_Rep *__r = _Rep::_S_create(__dnew, (size_type)0, __a); 
# 139
try 
# 140
{ _S_copy_chars((__r->_M_refdata()), __beg, __end); } 
# 141
catch (...) 
# 142
{ 
# 143
(__r->_M_destroy(__a)); 
# 144
throw; 
# 145
}  
# 146
(__r->_M_set_length_and_sharable(__dnew)); 
# 147
return (__r->_M_refdata()); 
# 148
} 
# 150
template< class _CharT, class _Traits, class _Alloc> _CharT *
# 153
basic_string< _CharT, _Traits, _Alloc> ::_S_construct(size_type __n, _CharT __c, const _Alloc &__a) 
# 154
{ 
# 156
if ((__n == 0) && (__a == _Alloc())) { 
# 157
return ((_S_empty_rep)()._M_refdata()); }  
# 160
_Rep *__r = _Rep::_S_create(__n, (size_type)0, __a); 
# 161
if (__n) { 
# 162
(_M_assign)((__r->_M_refdata()), __n, __c); }  
# 164
(__r->_M_set_length_and_sharable(__n)); 
# 165
return (__r->_M_refdata()); 
# 166
} 
# 168
template< class _CharT, class _Traits, class _Alloc> 
# 170
basic_string< _CharT, _Traits, _Alloc> ::basic_string(const basic_string &__str) : _M_dataplus((__str._M_rep()->_M_grab((_Alloc)__str.get_allocator(), __str.get_allocator())), __str.get_allocator()) 
# 174
{ } 
# 176
template< class _CharT, class _Traits, class _Alloc> 
# 178
basic_string< _CharT, _Traits, _Alloc> ::basic_string(const _Alloc &__a) : _M_dataplus(_S_construct(size_type(), _CharT(), __a), __a) 
# 180
{ } 
# 182
template< class _CharT, class _Traits, class _Alloc> 
# 184
basic_string< _CharT, _Traits, _Alloc> ::basic_string(const basic_string &__str, size_type __pos, size_type __n) : _M_dataplus(_S_construct((__str._M_data()) + __str._M_check(__pos, "basic_string::basic_string"), ((__str._M_data()) + __str._M_limit(__pos, __n)) + __pos, _Alloc()), _Alloc()) 
# 190
{ } 
# 192
template< class _CharT, class _Traits, class _Alloc> 
# 194
basic_string< _CharT, _Traits, _Alloc> ::basic_string(const basic_string &__str, size_type __pos, size_type 
# 195
__n, const _Alloc &__a) : _M_dataplus(_S_construct((__str._M_data()) + __str._M_check(__pos, "basic_string::basic_string"), ((__str._M_data()) + __str._M_limit(__pos, __n)) + __pos, __a), __a) 
# 201
{ } 
# 204
template< class _CharT, class _Traits, class _Alloc> 
# 206
basic_string< _CharT, _Traits, _Alloc> ::basic_string(const _CharT *__s, size_type __n, const _Alloc &__a) : _M_dataplus(_S_construct(__s, __s + __n, __a), __a) 
# 208
{ } 
# 211
template< class _CharT, class _Traits, class _Alloc> 
# 213
basic_string< _CharT, _Traits, _Alloc> ::basic_string(const _CharT *__s, const _Alloc &__a) : _M_dataplus(_S_construct(__s, (__s) ? __s + traits_type::length(__s) : (__s + npos), __a), __a) 
# 216
{ } 
# 218
template< class _CharT, class _Traits, class _Alloc> 
# 220
basic_string< _CharT, _Traits, _Alloc> ::basic_string(size_type __n, _CharT __c, const _Alloc &__a) : _M_dataplus(_S_construct(__n, __c, __a), __a) 
# 222
{ } 
# 225
template< class _CharT, class _Traits, class _Alloc> 
# 226
template< class _InputIterator> 
# 228
basic_string< _CharT, _Traits, _Alloc> ::basic_string(_InputIterator __beg, _InputIterator __end, const _Alloc &__a) : _M_dataplus(_S_construct(__beg, __end, __a), __a) 
# 230
{ } 
# 233
template< class _CharT, class _Traits, class _Alloc> 
# 235
basic_string< _CharT, _Traits, _Alloc> ::basic_string(initializer_list< _CharT>  __l, const _Alloc &__a) : _M_dataplus(_S_construct((__l.begin()), (__l.end()), __a), __a) 
# 237
{ } 
# 240
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 243
basic_string< _CharT, _Traits, _Alloc> ::assign(const basic_string &__str) 
# 244
{ 
# 245
if (_M_rep() != __str._M_rep()) 
# 246
{ 
# 248
const allocator_type __a = this->get_allocator(); 
# 249
_CharT *__tmp = (__str._M_rep()->_M_grab(__a, __str.get_allocator())); 
# 250
(_M_rep()->_M_dispose(__a)); 
# 251
_M_data(__tmp); 
# 252
}  
# 253
return *this; 
# 254
} 
# 256
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 259
basic_string< _CharT, _Traits, _Alloc> ::assign(const _CharT *__s, size_type __n) 
# 260
{ 
# 261
; 
# 262
_M_check_length(this->size(), __n, "basic_string::assign"); 
# 263
if (_M_disjunct(__s) || (_M_rep()->_M_is_shared())) { 
# 264
return _M_replace_safe((size_type)0, this->size(), __s, __n); } else 
# 266
{ 
# 268
const size_type __pos = __s - _M_data(); 
# 269
if (__pos >= __n) { 
# 270
(_M_copy)(_M_data(), __s, __n); } else { 
# 271
if (__pos) { 
# 272
(_M_move)(_M_data(), __s, __n); }  }  
# 273
(_M_rep()->_M_set_length_and_sharable(__n)); 
# 274
return *this; 
# 275
}  
# 276
} 
# 278
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 281
basic_string< _CharT, _Traits, _Alloc> ::append(size_type __n, _CharT __c) 
# 282
{ 
# 283
if (__n) 
# 284
{ 
# 285
_M_check_length((size_type)0, __n, "basic_string::append"); 
# 286
const size_type __len = __n + this->size(); 
# 287
if ((__len > this->capacity()) || (_M_rep()->_M_is_shared())) { 
# 288
this->reserve(__len); }  
# 289
(_M_assign)(_M_data() + this->size(), __n, __c); 
# 290
(_M_rep()->_M_set_length_and_sharable(__len)); 
# 291
}  
# 292
return *this; 
# 293
} 
# 295
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 298
basic_string< _CharT, _Traits, _Alloc> ::append(const _CharT *__s, size_type __n) 
# 299
{ 
# 300
; 
# 301
if (__n) 
# 302
{ 
# 303
_M_check_length((size_type)0, __n, "basic_string::append"); 
# 304
const size_type __len = __n + this->size(); 
# 305
if ((__len > this->capacity()) || (_M_rep()->_M_is_shared())) 
# 306
{ 
# 307
if (_M_disjunct(__s)) { 
# 308
this->reserve(__len); } else 
# 310
{ 
# 311
const size_type __off = __s - _M_data(); 
# 312
this->reserve(__len); 
# 313
__s = (_M_data() + __off); 
# 314
}  
# 315
}  
# 316
(_M_copy)(_M_data() + this->size(), __s, __n); 
# 317
(_M_rep()->_M_set_length_and_sharable(__len)); 
# 318
}  
# 319
return *this; 
# 320
} 
# 322
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 325
basic_string< _CharT, _Traits, _Alloc> ::append(const basic_string &__str) 
# 326
{ 
# 327
const size_type __size = __str.size(); 
# 328
if (__size) 
# 329
{ 
# 330
const size_type __len = __size + this->size(); 
# 331
if ((__len > this->capacity()) || (_M_rep()->_M_is_shared())) { 
# 332
this->reserve(__len); }  
# 333
(_M_copy)(_M_data() + this->size(), (__str._M_data()), __size); 
# 334
(_M_rep()->_M_set_length_and_sharable(__len)); 
# 335
}  
# 336
return *this; 
# 337
} 
# 339
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 342
basic_string< _CharT, _Traits, _Alloc> ::append(const basic_string &__str, size_type __pos, size_type __n) 
# 343
{ 
# 344
__str._M_check(__pos, "basic_string::append"); 
# 345
__n = __str._M_limit(__pos, __n); 
# 346
if (__n) 
# 347
{ 
# 348
const size_type __len = __n + this->size(); 
# 349
if ((__len > this->capacity()) || (_M_rep()->_M_is_shared())) { 
# 350
this->reserve(__len); }  
# 351
(_M_copy)(_M_data() + this->size(), (__str._M_data()) + __pos, __n); 
# 352
(_M_rep()->_M_set_length_and_sharable(__len)); 
# 353
}  
# 354
return *this; 
# 355
} 
# 357
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 360
basic_string< _CharT, _Traits, _Alloc> ::insert(size_type __pos, const _CharT *__s, size_type __n) 
# 361
{ 
# 362
; 
# 363
_M_check(__pos, "basic_string::insert"); 
# 364
_M_check_length((size_type)0, __n, "basic_string::insert"); 
# 365
if (_M_disjunct(__s) || (_M_rep()->_M_is_shared())) { 
# 366
return _M_replace_safe(__pos, (size_type)0, __s, __n); } else 
# 368
{ 
# 370
const size_type __off = __s - _M_data(); 
# 371
_M_mutate(__pos, 0, __n); 
# 372
__s = (_M_data() + __off); 
# 373
_CharT *__p = _M_data() + __pos; 
# 374
if ((__s + __n) <= __p) { 
# 375
(_M_copy)(__p, __s, __n); } else { 
# 376
if (__s >= __p) { 
# 377
(_M_copy)(__p, __s + __n, __n); } else 
# 379
{ 
# 380
const size_type __nleft = __p - __s; 
# 381
(_M_copy)(__p, __s, __nleft); 
# 382
(_M_copy)(__p + __nleft, __p + __n, __n - __nleft); 
# 383
}  }  
# 384
return *this; 
# 385
}  
# 386
} 
# 388
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::iterator 
# 391
basic_string< _CharT, _Traits, _Alloc> ::erase(iterator __first, iterator __last) 
# 392
{ 
# 394
; 
# 399
const size_type __size = __last - __first; 
# 400
if (__size) 
# 401
{ 
# 402
const size_type __pos = __first - _M_ibegin(); 
# 403
_M_mutate(__pos, __size, (size_type)0); 
# 404
(_M_rep()->_M_set_leaked()); 
# 405
return ((iterator)(_M_data() + __pos)); 
# 406
} else { 
# 408
return __first; }  
# 409
} 
# 411
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 414
basic_string< _CharT, _Traits, _Alloc> ::replace(size_type __pos, size_type __n1, const _CharT *__s, size_type 
# 415
__n2) 
# 416
{ 
# 417
; 
# 418
_M_check(__pos, "basic_string::replace"); 
# 419
__n1 = _M_limit(__pos, __n1); 
# 420
_M_check_length(__n1, __n2, "basic_string::replace"); 
# 421
bool __left; 
# 422
if (_M_disjunct(__s) || (_M_rep()->_M_is_shared())) { 
# 423
return _M_replace_safe(__pos, __n1, __s, __n2); } else { 
# 424
if ((__left = ((__s + __n2) <= (_M_data() + __pos))) || (((_M_data() + __pos) + __n1) <= __s)) 
# 426
{ 
# 428
size_type __off = __s - _M_data(); 
# 429
__left ? __off : (__off += (__n2 - __n1)); 
# 430
_M_mutate(__pos, __n1, __n2); 
# 431
(_M_copy)(_M_data() + __pos, _M_data() + __off, __n2); 
# 432
return *this; 
# 433
} else 
# 435
{ 
# 437
const basic_string __tmp(__s, __n2); 
# 438
return _M_replace_safe(__pos, __n1, (__tmp._M_data()), __n2); 
# 439
}  }  
# 440
} 
# 442
template< class _CharT, class _Traits, class _Alloc> void 
# 445
basic_string< _CharT, _Traits, _Alloc> ::_Rep::_M_destroy(const _Alloc &__a) throw() 
# 446
{ 
# 447
const typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __size = sizeof(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::_Rep_base) + (((this->_M_capacity) + 1) * sizeof(_CharT)); 
# 449
(((_Raw_bytes_alloc)__a).deallocate(reinterpret_cast< char *>(this), __size)); 
# 450
} 
# 452
template< class _CharT, class _Traits, class _Alloc> void 
# 455
basic_string< _CharT, _Traits, _Alloc> ::_M_leak_hard() 
# 456
{ 
# 458
if (_M_rep() == (&(_S_empty_rep)())) { 
# 459
return; }  
# 461
if ((_M_rep()->_M_is_shared())) { 
# 462
_M_mutate(0, 0, 0); }  
# 463
(_M_rep()->_M_set_leaked()); 
# 464
} 
# 466
template< class _CharT, class _Traits, class _Alloc> void 
# 469
basic_string< _CharT, _Traits, _Alloc> ::_M_mutate(size_type __pos, size_type __len1, size_type __len2) 
# 470
{ 
# 471
const size_type __old_size = this->size(); 
# 472
const size_type __new_size = (__old_size + __len2) - __len1; 
# 473
const size_type __how_much = (__old_size - __pos) - __len1; 
# 475
if ((__new_size > this->capacity()) || (_M_rep()->_M_is_shared())) 
# 476
{ 
# 478
const allocator_type __a = get_allocator(); 
# 479
_Rep *__r = _Rep::_S_create(__new_size, this->capacity(), __a); 
# 481
if (__pos) { 
# 482
(_M_copy)((__r->_M_refdata()), _M_data(), __pos); }  
# 483
if (__how_much) { 
# 484
(_M_copy)(((__r->_M_refdata()) + __pos) + __len2, (_M_data() + __pos) + __len1, __how_much); }  
# 487
(_M_rep()->_M_dispose(__a)); 
# 488
_M_data((__r->_M_refdata())); 
# 489
} else { 
# 490
if (__how_much && (__len1 != __len2)) 
# 491
{ 
# 493
(_M_move)((_M_data() + __pos) + __len2, (_M_data() + __pos) + __len1, __how_much); 
# 495
}  }  
# 496
(_M_rep()->_M_set_length_and_sharable(__new_size)); 
# 497
} 
# 499
template< class _CharT, class _Traits, class _Alloc> void 
# 502
basic_string< _CharT, _Traits, _Alloc> ::reserve(size_type __res) 
# 503
{ 
# 504
if ((__res != this->capacity()) || (_M_rep()->_M_is_shared())) 
# 505
{ 
# 507
if (__res < this->size()) { 
# 508
__res = this->size(); }  
# 509
const allocator_type __a = get_allocator(); 
# 510
_CharT *__tmp = (_M_rep()->_M_clone(__a, __res - this->size())); 
# 511
(_M_rep()->_M_dispose(__a)); 
# 512
_M_data(__tmp); 
# 513
}  
# 514
} 
# 516
template< class _CharT, class _Traits, class _Alloc> void 
# 519
basic_string< _CharT, _Traits, _Alloc> ::swap(basic_string &__s) 
# 520
{ 
# 521
if ((_M_rep()->_M_is_leaked())) { 
# 522
(_M_rep()->_M_set_sharable()); }  
# 523
if ((__s._M_rep()->_M_is_leaked())) { 
# 524
(__s._M_rep()->_M_set_sharable()); }  
# 525
if (this->get_allocator() == __s.get_allocator()) 
# 526
{ 
# 527
_CharT *__tmp = _M_data(); 
# 528
_M_data((__s._M_data())); 
# 529
(__s._M_data(__tmp)); 
# 530
} else 
# 533
{ 
# 534
const basic_string __tmp1(_M_ibegin(), _M_iend(), __s.get_allocator()); 
# 536
const basic_string __tmp2(__s._M_ibegin(), __s._M_iend(), this->get_allocator()); 
# 538
(*this) = __tmp2; 
# 539
__s = __tmp1; 
# 540
}  
# 541
} 
# 543
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::_Rep *
# 546
basic_string< _CharT, _Traits, _Alloc> ::_Rep::_S_create(typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __capacity, typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __old_capacity, const _Alloc &
# 547
__alloc) 
# 548
{ 
# 551
if (__capacity > _S_max_size) { 
# 552
__throw_length_error("basic_string::_S_create"); }  
# 577
const typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __pagesize = (4096); 
# 578
const typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __malloc_header_size = ((4) * sizeof(void *)); 
# 586
if ((__capacity > __old_capacity) && (__capacity < (2 * __old_capacity))) { 
# 587
__capacity = (2 * __old_capacity); }  
# 592
typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __size = ((__capacity + 1) * sizeof(_CharT)) + sizeof(_Rep); 
# 594
const typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __adj_size = __size + __malloc_header_size; 
# 595
if ((__adj_size > __pagesize) && (__capacity > __old_capacity)) 
# 596
{ 
# 597
const typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __extra = __pagesize - (__adj_size % __pagesize); 
# 598
__capacity += (__extra / sizeof(_CharT)); 
# 600
if (__capacity > _S_max_size) { 
# 601
__capacity = _S_max_size; }  
# 602
__size = (((__capacity + 1) * sizeof(_CharT)) + sizeof(_Rep)); 
# 603
}  
# 607
void *__place = (((_Raw_bytes_alloc)__alloc).allocate(__size)); 
# 608
_Rep *__p = new (__place) _Rep; 
# 609
(__p->_M_capacity) = __capacity; 
# 617
__p->_M_set_sharable(); 
# 618
return __p; 
# 619
} 
# 621
template< class _CharT, class _Traits, class _Alloc> _CharT *
# 624
basic_string< _CharT, _Traits, _Alloc> ::_Rep::_M_clone(const _Alloc &__alloc, typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __res) 
# 625
{ 
# 627
const typename ::std::basic_string< _CharT, _Traits, _Alloc> ::size_type __requested_cap = (this->_M_length) + __res; 
# 628
_Rep *__r = (_S_create)(__requested_cap, (this->_M_capacity), __alloc); 
# 630
if (this->_M_length) { 
# 631
(::std::basic_string< _CharT, _Traits, _Alloc> ::_M_copy)(__r->_M_refdata(), _M_refdata(), (this->_M_length)); }  
# 633
__r->_M_set_length_and_sharable((this->_M_length)); 
# 634
return __r->_M_refdata(); 
# 635
} 
# 637
template< class _CharT, class _Traits, class _Alloc> void 
# 640
basic_string< _CharT, _Traits, _Alloc> ::resize(size_type __n, _CharT __c) 
# 641
{ 
# 642
const size_type __size = this->size(); 
# 643
_M_check_length(__size, __n, "basic_string::resize"); 
# 644
if (__size < __n) { 
# 645
(this->append(__n - __size, __c)); } else { 
# 646
if (__n < __size) { 
# 647
(this->erase(__n)); }  }  
# 649
} 
# 651
template< class _CharT, class _Traits, class _Alloc> 
# 652
template< class _InputIterator> basic_string< _CharT, _Traits, _Alloc>  &
# 655
basic_string< _CharT, _Traits, _Alloc> ::_M_replace_dispatch(iterator __i1, iterator __i2, _InputIterator __k1, _InputIterator 
# 656
__k2, __false_type) 
# 657
{ 
# 658
const basic_string __s(__k1, __k2); 
# 659
const size_type __n1 = __i2 - __i1; 
# 660
_M_check_length(__n1, __s.size(), "basic_string::_M_replace_dispatch"); 
# 661
return _M_replace_safe(__i1 - _M_ibegin(), __n1, (__s._M_data()), __s.size()); 
# 663
} 
# 665
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 668
basic_string< _CharT, _Traits, _Alloc> ::_M_replace_aux(size_type __pos1, size_type __n1, size_type __n2, _CharT 
# 669
__c) 
# 670
{ 
# 671
_M_check_length(__n1, __n2, "basic_string::_M_replace_aux"); 
# 672
_M_mutate(__pos1, __n1, __n2); 
# 673
if (__n2) { 
# 674
(_M_assign)(_M_data() + __pos1, __n2, __c); }  
# 675
return *this; 
# 676
} 
# 678
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  &
# 681
basic_string< _CharT, _Traits, _Alloc> ::_M_replace_safe(size_type __pos1, size_type __n1, const _CharT *__s, size_type 
# 682
__n2) 
# 683
{ 
# 684
_M_mutate(__pos1, __n1, __n2); 
# 685
if (__n2) { 
# 686
(_M_copy)(_M_data() + __pos1, __s, __n2); }  
# 687
return *this; 
# 688
} 
# 690
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  
# 692
operator+(const _CharT *__lhs, const basic_string< _CharT, _Traits, _Alloc>  &
# 693
__rhs) 
# 694
{ 
# 695
; 
# 696
typedef basic_string< _CharT, _Traits, _Alloc>  __string_type; 
# 697
typedef typename basic_string< _CharT, _Traits, _Alloc> ::size_type __size_type; 
# 698
const __size_type __len = _Traits::length(__lhs); 
# 699
__string_type __str; 
# 700
(__str.reserve(__len + (__rhs.size()))); 
# 701
(__str.append(__lhs, __len)); 
# 702
(__str.append(__rhs)); 
# 703
return __str; 
# 704
} 
# 706
template< class _CharT, class _Traits, class _Alloc> basic_string< _CharT, _Traits, _Alloc>  
# 708
operator+(_CharT __lhs, const basic_string< _CharT, _Traits, _Alloc>  &__rhs) 
# 709
{ 
# 710
typedef basic_string< _CharT, _Traits, _Alloc>  __string_type; 
# 711
typedef typename basic_string< _CharT, _Traits, _Alloc> ::size_type __size_type; 
# 712
__string_type __str; 
# 713
const __size_type __len = (__rhs.size()); 
# 714
(__str.reserve(__len + 1)); 
# 715
(__str.append((__size_type)1, __lhs)); 
# 716
(__str.append(__rhs)); 
# 717
return __str; 
# 718
} 
# 720
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 723
basic_string< _CharT, _Traits, _Alloc> ::copy(_CharT *__s, size_type __n, size_type __pos) const 
# 724
{ 
# 725
_M_check(__pos, "basic_string::copy"); 
# 726
__n = _M_limit(__pos, __n); 
# 727
; 
# 728
if (__n) { 
# 729
(_M_copy)(__s, _M_data() + __pos, __n); }  
# 731
return __n; 
# 732
} 
# 734
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 737
basic_string< _CharT, _Traits, _Alloc> ::find(const _CharT *__s, size_type __pos, size_type __n) const 
# 738
{ 
# 739
; 
# 740
const size_type __size = this->size(); 
# 741
const _CharT *__data = _M_data(); 
# 743
if (__n == 0) { 
# 744
return (__pos <= __size) ? __pos : npos; }  
# 746
if (__n <= __size) 
# 747
{ 
# 748
for (; __pos <= (__size - __n); ++__pos) { 
# 749
if (traits_type::eq(__data[__pos], __s[0]) && (traits_type::compare((__data + __pos) + 1, __s + 1, __n - 1) == 0)) { 
# 752
return __pos; }  }  
# 753
}  
# 754
return npos; 
# 755
} 
# 757
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 760
basic_string< _CharT, _Traits, _Alloc> ::find(_CharT __c, size_type __pos) const noexcept 
# 761
{ 
# 762
size_type __ret = npos; 
# 763
const size_type __size = this->size(); 
# 764
if (__pos < __size) 
# 765
{ 
# 766
const _CharT *__data = _M_data(); 
# 767
const size_type __n = __size - __pos; 
# 768
const _CharT *__p = traits_type::find(__data + __pos, __n, __c); 
# 769
if (__p) { 
# 770
__ret = (__p - __data); }  
# 771
}  
# 772
return __ret; 
# 773
} 
# 775
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 778
basic_string< _CharT, _Traits, _Alloc> ::rfind(const _CharT *__s, size_type __pos, size_type __n) const 
# 779
{ 
# 780
; 
# 781
const size_type __size = this->size(); 
# 782
if (__n <= __size) 
# 783
{ 
# 784
__pos = std::min((size_type)(__size - __n), __pos); 
# 785
const _CharT *__data = _M_data(); 
# 786
do 
# 787
{ 
# 788
if (traits_type::compare(__data + __pos, __s, __n) == 0) { 
# 789
return __pos; }  
# 790
} 
# 791
while ((__pos--) > 0); 
# 792
}  
# 793
return npos; 
# 794
} 
# 796
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 799
basic_string< _CharT, _Traits, _Alloc> ::rfind(_CharT __c, size_type __pos) const noexcept 
# 800
{ 
# 801
size_type __size = this->size(); 
# 802
if (__size) 
# 803
{ 
# 804
if ((--__size) > __pos) { 
# 805
__size = __pos; }  
# 806
for (++__size; (__size--) > 0;) { 
# 807
if (traits_type::eq(_M_data()[__size], __c)) { 
# 808
return __size; }  }  
# 809
}  
# 810
return npos; 
# 811
} 
# 813
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 816
basic_string< _CharT, _Traits, _Alloc> ::find_first_of(const _CharT *__s, size_type __pos, size_type __n) const 
# 817
{ 
# 818
; 
# 819
for (; __n && (__pos < this->size()); ++__pos) 
# 820
{ 
# 821
const _CharT *__p = traits_type::find(__s, __n, _M_data()[__pos]); 
# 822
if (__p) { 
# 823
return __pos; }  
# 824
}  
# 825
return npos; 
# 826
} 
# 828
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 831
basic_string< _CharT, _Traits, _Alloc> ::find_last_of(const _CharT *__s, size_type __pos, size_type __n) const 
# 832
{ 
# 833
; 
# 834
size_type __size = this->size(); 
# 835
if (__size && __n) 
# 836
{ 
# 837
if ((--__size) > __pos) { 
# 838
__size = __pos; }  
# 839
do 
# 840
{ 
# 841
if (traits_type::find(__s, __n, _M_data()[__size])) { 
# 842
return __size; }  
# 843
} 
# 844
while ((__size--) != 0); 
# 845
}  
# 846
return npos; 
# 847
} 
# 849
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 852
basic_string< _CharT, _Traits, _Alloc> ::find_first_not_of(const _CharT *__s, size_type __pos, size_type __n) const 
# 853
{ 
# 854
; 
# 855
for (; __pos < this->size(); ++__pos) { 
# 856
if (!traits_type::find(__s, __n, _M_data()[__pos])) { 
# 857
return __pos; }  }  
# 858
return npos; 
# 859
} 
# 861
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 864
basic_string< _CharT, _Traits, _Alloc> ::find_first_not_of(_CharT __c, size_type __pos) const noexcept 
# 865
{ 
# 866
for (; __pos < this->size(); ++__pos) { 
# 867
if (!traits_type::eq(_M_data()[__pos], __c)) { 
# 868
return __pos; }  }  
# 869
return npos; 
# 870
} 
# 872
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 875
basic_string< _CharT, _Traits, _Alloc> ::find_last_not_of(const _CharT *__s, size_type __pos, size_type __n) const 
# 876
{ 
# 877
; 
# 878
size_type __size = this->size(); 
# 879
if (__size) 
# 880
{ 
# 881
if ((--__size) > __pos) { 
# 882
__size = __pos; }  
# 883
do 
# 884
{ 
# 885
if (!traits_type::find(__s, __n, _M_data()[__size])) { 
# 886
return __size; }  
# 887
} 
# 888
while (__size--); 
# 889
}  
# 890
return npos; 
# 891
} 
# 893
template< class _CharT, class _Traits, class _Alloc> typename basic_string< _CharT, _Traits, _Alloc> ::size_type 
# 896
basic_string< _CharT, _Traits, _Alloc> ::find_last_not_of(_CharT __c, size_type __pos) const noexcept 
# 897
{ 
# 898
size_type __size = this->size(); 
# 899
if (__size) 
# 900
{ 
# 901
if ((--__size) > __pos) { 
# 902
__size = __pos; }  
# 903
do 
# 904
{ 
# 905
if (!traits_type::eq(_M_data()[__size], __c)) { 
# 906
return __size; }  
# 907
} 
# 908
while (__size--); 
# 909
}  
# 910
return npos; 
# 911
} 
# 913
template< class _CharT, class _Traits, class _Alloc> int 
# 916
basic_string< _CharT, _Traits, _Alloc> ::compare(size_type __pos, size_type __n, const basic_string &__str) const 
# 917
{ 
# 918
_M_check(__pos, "basic_string::compare"); 
# 919
__n = _M_limit(__pos, __n); 
# 920
const size_type __osize = __str.size(); 
# 921
const size_type __len = std::min(__n, __osize); 
# 922
int __r = traits_type::compare(_M_data() + __pos, __str.data(), __len); 
# 923
if (!__r) { 
# 924
__r = (_S_compare)(__n, __osize); }  
# 925
return __r; 
# 926
} 
# 928
template< class _CharT, class _Traits, class _Alloc> int 
# 931
basic_string< _CharT, _Traits, _Alloc> ::compare(size_type __pos1, size_type __n1, const basic_string &__str, size_type 
# 932
__pos2, size_type __n2) const 
# 933
{ 
# 934
_M_check(__pos1, "basic_string::compare"); 
# 935
__str._M_check(__pos2, "basic_string::compare"); 
# 936
__n1 = _M_limit(__pos1, __n1); 
# 937
__n2 = __str._M_limit(__pos2, __n2); 
# 938
const size_type __len = std::min(__n1, __n2); 
# 939
int __r = traits_type::compare(_M_data() + __pos1, __str.data() + __pos2, __len); 
# 941
if (!__r) { 
# 942
__r = (_S_compare)(__n1, __n2); }  
# 943
return __r; 
# 944
} 
# 946
template< class _CharT, class _Traits, class _Alloc> int 
# 949
basic_string< _CharT, _Traits, _Alloc> ::compare(const _CharT *__s) const 
# 950
{ 
# 951
; 
# 952
const size_type __size = this->size(); 
# 953
const size_type __osize = traits_type::length(__s); 
# 954
const size_type __len = std::min(__size, __osize); 
# 955
int __r = traits_type::compare(_M_data(), __s, __len); 
# 956
if (!__r) { 
# 957
__r = (_S_compare)(__size, __osize); }  
# 958
return __r; 
# 959
} 
# 961
template< class _CharT, class _Traits, class _Alloc> int 
# 964
basic_string< _CharT, _Traits, _Alloc> ::compare(size_type __pos, size_type __n1, const _CharT *__s) const 
# 965
{ 
# 966
; 
# 967
_M_check(__pos, "basic_string::compare"); 
# 968
__n1 = _M_limit(__pos, __n1); 
# 969
const size_type __osize = traits_type::length(__s); 
# 970
const size_type __len = std::min(__n1, __osize); 
# 971
int __r = traits_type::compare(_M_data() + __pos, __s, __len); 
# 972
if (!__r) { 
# 973
__r = (_S_compare)(__n1, __osize); }  
# 974
return __r; 
# 975
} 
# 977
template< class _CharT, class _Traits, class _Alloc> int 
# 980
basic_string< _CharT, _Traits, _Alloc> ::compare(size_type __pos, size_type __n1, const _CharT *__s, size_type 
# 981
__n2) const 
# 982
{ 
# 983
; 
# 984
_M_check(__pos, "basic_string::compare"); 
# 985
__n1 = _M_limit(__pos, __n1); 
# 986
const size_type __len = std::min(__n1, __n2); 
# 987
int __r = traits_type::compare(_M_data() + __pos, __s, __len); 
# 988
if (!__r) { 
# 989
__r = (_S_compare)(__n1, __n2); }  
# 990
return __r; 
# 991
} 
# 994
template< class _CharT, class _Traits, class _Alloc> basic_istream< _CharT, _Traits>  &
# 996
operator>>(basic_istream< _CharT, _Traits>  &__in, basic_string< _CharT, _Traits, _Alloc>  &
# 997
__str) 
# 998
{ 
# 999
typedef basic_istream< _CharT, _Traits>  __istream_type; 
# 1000
typedef basic_string< _CharT, _Traits, _Alloc>  __string_type; 
# 1001
typedef typename basic_istream< _CharT, _Traits> ::ios_base __ios_base; 
# 1002
typedef typename basic_istream< _CharT, _Traits> ::int_type __int_type; 
# 1003
typedef typename basic_string< _CharT, _Traits, _Alloc> ::size_type __size_type; 
# 1004
typedef ctype< _CharT>  __ctype_type; 
# 1005
typedef typename ctype< _CharT> ::ctype_base __ctype_base; 
# 1007
__size_type __extracted = (0); 
# 1008
typename basic_istream< _CharT, _Traits> ::ios_base::iostate __err = (__ios_base::goodbit); 
# 1009
typename basic_istream< _CharT, _Traits> ::sentry __cerb(__in, false); 
# 1010
if (__cerb) 
# 1011
{ 
# 1012
try 
# 1013
{ 
# 1015
(__str.erase()); 
# 1016
_CharT __buf[128]; 
# 1017
__size_type __len = (0); 
# 1018
const streamsize __w = (__in.width()); 
# 1019
const __size_type __n = (__w > (0)) ? static_cast< __size_type>(__w) : (__str.max_size()); 
# 1021
const __ctype_type &__ct = use_facet< ctype< _CharT> > ((__in.getloc())); 
# 1022
const __int_type __eof = _Traits::eof(); 
# 1023
__int_type __c = ((__in.rdbuf())->sgetc()); 
# 1025
while ((__extracted < __n) && (!_Traits::eq_int_type(__c, __eof)) && (!(__ct.is(__ctype_base::space, _Traits::to_char_type(__c))))) 
# 1029
{ 
# 1030
if (__len == (sizeof(__buf) / sizeof(_CharT))) 
# 1031
{ 
# 1032
(__str.append(__buf, sizeof(__buf) / sizeof(_CharT))); 
# 1033
__len = 0; 
# 1034
}  
# 1035
(__buf[__len++]) = _Traits::to_char_type(__c); 
# 1036
++__extracted; 
# 1037
__c = ((__in.rdbuf())->snextc()); 
# 1038
}  
# 1039
(__str.append(__buf, __len)); 
# 1041
if (_Traits::eq_int_type(__c, __eof)) { 
# 1042
__err |= __ios_base::eofbit; }  
# 1043
(__in.width(0)); 
# 1044
} 
# 1045
catch (__cxxabiv1::__forced_unwind &) 
# 1046
{ 
# 1047
(__in._M_setstate(__ios_base::badbit)); 
# 1048
throw; 
# 1049
} 
# 1050
catch (...) 
# 1051
{ 
# 1055
(__in._M_setstate(__ios_base::badbit)); 
# 1056
}  
# 1057
}  
# 1059
if (!__extracted) { 
# 1060
__err |= __ios_base::failbit; }  
# 1061
if (__err) { 
# 1062
(__in.setstate(__err)); }  
# 1063
return __in; 
# 1064
} 
# 1066
template< class _CharT, class _Traits, class _Alloc> basic_istream< _CharT, _Traits>  &
# 1068
getline(basic_istream< _CharT, _Traits>  &__in, basic_string< _CharT, _Traits, _Alloc>  &
# 1069
__str, _CharT __delim) 
# 1070
{ 
# 1071
typedef basic_istream< _CharT, _Traits>  __istream_type; 
# 1072
typedef basic_string< _CharT, _Traits, _Alloc>  __string_type; 
# 1073
typedef typename basic_istream< _CharT, _Traits> ::ios_base __ios_base; 
# 1074
typedef typename basic_istream< _CharT, _Traits> ::int_type __int_type; 
# 1075
typedef typename basic_string< _CharT, _Traits, _Alloc> ::size_type __size_type; 
# 1077
__size_type __extracted = (0); 
# 1078
const __size_type __n = (__str.max_size()); 
# 1079
typename basic_istream< _CharT, _Traits> ::ios_base::iostate __err = (__ios_base::goodbit); 
# 1080
typename basic_istream< _CharT, _Traits> ::sentry __cerb(__in, true); 
# 1081
if (__cerb) 
# 1082
{ 
# 1083
try 
# 1084
{ 
# 1085
(__str.erase()); 
# 1086
const __int_type __idelim = _Traits::to_int_type(__delim); 
# 1087
const __int_type __eof = _Traits::eof(); 
# 1088
__int_type __c = ((__in.rdbuf())->sgetc()); 
# 1090
while ((__extracted < __n) && (!_Traits::eq_int_type(__c, __eof)) && (!_Traits::eq_int_type(__c, __idelim))) 
# 1093
{ 
# 1094
__str += _Traits::to_char_type(__c); 
# 1095
++__extracted; 
# 1096
__c = ((__in.rdbuf())->snextc()); 
# 1097
}  
# 1099
if (_Traits::eq_int_type(__c, __eof)) { 
# 1100
__err |= __ios_base::eofbit; } else { 
# 1101
if (_Traits::eq_int_type(__c, __idelim)) 
# 1102
{ 
# 1103
++__extracted; 
# 1104
((__in.rdbuf())->sbumpc()); 
# 1105
} else { 
# 1107
__err |= __ios_base::failbit; }  }  
# 1108
} 
# 1109
catch (__cxxabiv1::__forced_unwind &) 
# 1110
{ 
# 1111
(__in._M_setstate(__ios_base::badbit)); 
# 1112
throw; 
# 1113
} 
# 1114
catch (...) 
# 1115
{ 
# 1119
(__in._M_setstate(__ios_base::badbit)); 
# 1120
}  
# 1121
}  
# 1122
if (!__extracted) { 
# 1123
__err |= __ios_base::failbit; }  
# 1124
if (__err) { 
# 1125
(__in.setstate(__err)); }  
# 1126
return __in; 
# 1127
} 
# 1132
extern template class basic_string< char, char_traits< char> , allocator< char> > ;
# 1133
extern template basic_istream< char>  &operator>>(basic_istream< char>  & __is, basic_string< char, char_traits< char> , allocator< char> >  & __str);
# 1136
extern template basic_ostream< char>  &operator<<(basic_ostream< char>  & __os, const basic_string< char, char_traits< char> , allocator< char> >  & __str);
# 1139
extern template basic_istream< char>  &getline(basic_istream< char>  & __is, basic_string< char, char_traits< char> , allocator< char> >  & __str, char __delim);
# 1142
extern template basic_istream< char>  &getline(basic_istream< char>  & __is, basic_string< char, char_traits< char> , allocator< char> >  & __str);
# 1147
extern template class basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> > ;
# 1148
extern template basic_istream< wchar_t>  &operator>>(basic_istream< wchar_t>  & __is, basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> >  & __str);
# 1151
extern template basic_ostream< wchar_t>  &operator<<(basic_ostream< wchar_t>  & __os, const basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> >  & __str);
# 1154
extern template basic_istream< wchar_t>  &getline(basic_istream< wchar_t>  & __is, basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> >  & __str, wchar_t __delim);
# 1157
extern template basic_istream< wchar_t>  &getline(basic_istream< wchar_t>  & __is, basic_string< wchar_t, char_traits< wchar_t> , allocator< wchar_t> >  & __str);
# 1164
}
# 43 "/usr/include/c++/4.8/bits/locale_classes.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 62
class locale { 
# 67
public: typedef int category; 
# 70
class facet; 
# 71
class id; 
# 72
class _Impl; 
# 74
friend class facet; 
# 75
friend class _Impl; 
# 77
template< class _Facet> friend bool has_facet(const locale &) throw(); 
# 81
template< class _Facet> friend const _Facet &use_facet(const locale &); 
# 85
template< class _Cache> friend struct __use_cache; 
# 98
static const category none = 0; 
# 99
static const category ctype = (1L << 0); 
# 100
static const category numeric = (1L << 1); 
# 101
static const category collate = (1L << 2); 
# 102
static const category time = (1L << 3); 
# 103
static const category monetary = (1L << 4); 
# 104
static const category messages = (1L << 5); 
# 105
static const category all = (((((ctype | numeric) | collate) | time) | monetary) | messages); 
# 117
locale() throw(); 
# 126
locale(const locale & __other) throw(); 
# 137
explicit locale(const char * __s); 
# 151
locale(const locale & __base, const char * __s, category __cat); 
# 164
locale(const locale & __base, const locale & __add, category __cat); 
# 177
template< class _Facet> locale(const locale & __other, _Facet * __f); 
# 181
~locale() throw(); 
# 192
const locale &operator=(const locale & __other) throw(); 
# 206
template< class _Facet> locale combine(const locale & __other) const; 
# 216
string name() const; 
# 226
bool operator==(const locale & __other) const throw(); 
# 235
bool operator!=(const locale &__other) const throw() 
# 236
{ return !this->operator==(__other); } 
# 253
template< class _Char, class _Traits, class _Alloc> bool operator()(const basic_string< _Char, _Traits, _Alloc>  & __s1, const basic_string< _Char, _Traits, _Alloc>  & __s2) const; 
# 270
static locale global(const locale & __loc); 
# 276
static const locale &classic(); 
# 280
private: _Impl *_M_impl; 
# 283
static _Impl *_S_classic; 
# 286
static _Impl *_S_global; 
# 292
static const char *const *const _S_categories; 
# 304
enum { _S_categories_size = 12}; 
# 307
static __gthread_once_t _S_once; 
# 311
explicit locale(_Impl *) throw(); 
# 314
static void _S_initialize(); 
# 317
static void _S_initialize_once() throw(); 
# 320
static category _S_normalize_category(category); 
# 323
void _M_coalesce(const locale & __base, const locale & __add, category __cat); 
# 324
}; 
# 338
class locale::facet { 
# 341
friend class locale; 
# 342
friend class _Impl; 
# 344
mutable _Atomic_word _M_refcount; 
# 347
static __c_locale _S_c_locale; 
# 350
static const char _S_c_name[2]; 
# 353
static __gthread_once_t _S_once; 
# 357
static void _S_initialize_once(); 
# 370
protected: explicit facet(size_t __refs = 0) throw() : _M_refcount((__refs) ? 1 : 0) 
# 371
{ } 
# 375
virtual ~facet(); 
# 378
static void _S_create_c_locale(__c_locale & __cloc, const char * __s, __c_locale __old = 0); 
# 382
static __c_locale _S_clone_c_locale(__c_locale & __cloc) throw(); 
# 385
static void _S_destroy_c_locale(__c_locale & __cloc); 
# 388
static __c_locale _S_lc_ctype_c_locale(__c_locale __cloc, const char * __s); 
# 393
static __c_locale _S_get_c_locale(); 
# 395
__attribute((const)) static const char *
# 396
_S_get_c_name() throw(); 
# 400
private: void _M_add_reference() const throw() 
# 401
{ __gnu_cxx::__atomic_add_dispatch(&(_M_refcount), 1); } 
# 404
void _M_remove_reference() const throw() 
# 405
{ 
# 407
; 
# 408
if (__gnu_cxx::__exchange_and_add_dispatch(&(_M_refcount), -1) == 1) 
# 409
{ 
# 410
; 
# 411
try 
# 412
{ delete this; } 
# 413
catch (...) 
# 414
{ }  
# 415
}  
# 416
} 
# 418
facet(const facet &); 
# 421
facet &operator=(const facet &); 
# 422
}; 
# 436
class locale::id { 
# 439
friend class locale; 
# 440
friend class _Impl; 
# 442
template< class _Facet> friend const _Facet &use_facet(const locale &); 
# 446
template< class _Facet> friend bool has_facet(const locale &) throw(); 
# 453
mutable size_t _M_index; 
# 456
static _Atomic_word _S_refcount; 
# 459
void operator=(const id &); 
# 461
id(const id &); 
# 467
public: id() { } 
# 470
size_t _M_id() const throw(); 
# 471
}; 
# 475
class locale::_Impl { 
# 479
friend class locale; 
# 480
friend class facet; 
# 482
template< class _Facet> friend bool has_facet(const locale &) throw(); 
# 486
template< class _Facet> friend const _Facet &use_facet(const locale &); 
# 490
template< class _Cache> friend struct __use_cache; 
# 495
_Atomic_word _M_refcount; 
# 496
const facet **_M_facets; 
# 497
size_t _M_facets_size; 
# 498
const facet **_M_caches; 
# 499
char **_M_names; 
# 500
static const id *const _S_id_ctype[]; 
# 501
static const id *const _S_id_numeric[]; 
# 502
static const id *const _S_id_collate[]; 
# 503
static const id *const _S_id_time[]; 
# 504
static const id *const _S_id_monetary[]; 
# 505
static const id *const _S_id_messages[]; 
# 506
static const id *const *const _S_facet_categories[]; 
# 509
void _M_add_reference() throw() 
# 510
{ __gnu_cxx::__atomic_add_dispatch(&(_M_refcount), 1); } 
# 513
void _M_remove_reference() throw() 
# 514
{ 
# 516
; 
# 517
if (__gnu_cxx::__exchange_and_add_dispatch(&(_M_refcount), -1) == 1) 
# 518
{ 
# 519
; 
# 520
try 
# 521
{ delete this; } 
# 522
catch (...) 
# 523
{ }  
# 524
}  
# 525
} 
# 527
_Impl(const _Impl &, size_t); 
# 528
_Impl(const char *, size_t); 
# 529
_Impl(size_t) throw(); 
# 531
~_Impl() throw(); 
# 533
_Impl(const _Impl &); 
# 536
void operator=(const _Impl &); 
# 539
bool _M_check_same_name() 
# 540
{ 
# 541
bool __ret = true; 
# 542
if ((_M_names)[1]) { 
# 544
for (size_t __i = (0); __ret && (__i < ((_S_categories_size) - 1)); ++__i) { 
# 545
__ret = (__builtin_strcmp((_M_names)[__i], (_M_names)[__i + (1)]) == 0); }  }  
# 546
return __ret; 
# 547
} 
# 550
void _M_replace_categories(const _Impl *, category); 
# 553
void _M_replace_category(const _Impl *, const id *const *); 
# 556
void _M_replace_facet(const _Impl *, const id *); 
# 559
void _M_install_facet(const id *, const facet *); 
# 561
template< class _Facet> void 
# 563
_M_init_facet(_Facet *__facet) 
# 564
{ this->_M_install_facet(&_Facet::id, __facet); } 
# 567
void _M_install_cache(const facet *, size_t); 
# 568
}; 
# 583
template< class _CharT> 
# 584
class collate : public locale::facet { 
# 590
public: typedef _CharT char_type; 
# 591
typedef basic_string< _CharT, char_traits< _CharT> , allocator< _CharT> >  string_type; 
# 597
protected: __c_locale _M_c_locale_collate; 
# 601
public: static locale::id id; 
# 611
explicit collate(size_t __refs = 0) : locale::facet(__refs), _M_c_locale_collate(_S_get_c_locale()) 
# 613
{ } 
# 625
explicit collate(__c_locale __cloc, size_t __refs = 0) : locale::facet(__refs), _M_c_locale_collate(_S_clone_c_locale(__cloc)) 
# 627
{ } 
# 642
int compare(const _CharT *__lo1, const _CharT *__hi1, const _CharT *
# 643
__lo2, const _CharT *__hi2) const 
# 644
{ return this->do_compare(__lo1, __hi1, __lo2, __hi2); } 
# 661
string_type transform(const _CharT *__lo, const _CharT *__hi) const 
# 662
{ return this->do_transform(__lo, __hi); } 
# 675
long hash(const _CharT *__lo, const _CharT *__hi) const 
# 676
{ return this->do_hash(__lo, __hi); } 
# 680
int _M_compare(const _CharT *, const _CharT *) const throw(); 
# 683
size_t _M_transform(_CharT *, const _CharT *, size_t) const throw(); 
# 688
protected: virtual ~collate() 
# 689
{ _S_destroy_c_locale(_M_c_locale_collate); } 
# 704
virtual int do_compare(const _CharT * __lo1, const _CharT * __hi1, const _CharT * __lo2, const _CharT * __hi2) const; 
# 718
virtual string_type do_transform(const _CharT * __lo, const _CharT * __hi) const; 
# 731
virtual long do_hash(const _CharT * __lo, const _CharT * __hi) const; 
# 732
}; 
# 734
template< class _CharT> locale::id 
# 735
collate< _CharT> ::id; 
# 740
template<> int collate< char> ::_M_compare(const char *, const char *) const throw(); 
# 744
template<> size_t collate< char> ::_M_transform(char *, const char *, size_t) const throw(); 
# 749
template<> int collate< wchar_t> ::_M_compare(const wchar_t *, const wchar_t *) const throw(); 
# 753
template<> size_t collate< wchar_t> ::_M_transform(wchar_t *, const wchar_t *, size_t) const throw(); 
# 757
template< class _CharT> 
# 758
class collate_byname : public collate< _CharT>  { 
# 763
public: typedef _CharT char_type; 
# 764
typedef basic_string< _CharT, char_traits< _CharT> , allocator< _CharT> >  string_type; 
# 768
explicit collate_byname(const char *__s, ::std::size_t __refs = 0) : ::std::collate< _CharT> (__refs) 
# 770
{ 
# 771
if ((__builtin_strcmp(__s, "C") != 0) && (__builtin_strcmp(__s, "POSIX") != 0)) 
# 773
{ 
# 774
(this->_S_destroy_c_locale((this->_M_c_locale_collate))); 
# 775
(this->_S_create_c_locale((this->_M_c_locale_collate), __s)); 
# 776
}  
# 777
} 
# 781
protected: virtual ~collate_byname() { } 
# 782
}; 
# 785
}
# 39 "/usr/include/c++/4.8/bits/locale_classes.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 43
template< class _Facet> 
# 45
locale::locale(const locale &__other, _Facet *__f) 
# 46
{ 
# 47
(_M_impl) = (new _Impl(*(__other._M_impl), 1)); 
# 49
try 
# 50
{ (_M_impl)->_M_install_facet(&_Facet::id, __f); } 
# 51
catch (...) 
# 52
{ 
# 53
(_M_impl)->_M_remove_reference(); 
# 54
throw; 
# 55
}  
# 56
delete [] (((_M_impl)->_M_names)[0]); 
# 57
(((_M_impl)->_M_names)[0]) = (0); 
# 58
} 
# 60
template< class _Facet> locale 
# 63
locale::combine(const locale &__other) const 
# 64
{ 
# 65
_Impl *__tmp = new _Impl(*(_M_impl), 1); 
# 66
try 
# 67
{ 
# 68
__tmp->_M_replace_facet(__other._M_impl, &_Facet::id); 
# 69
} 
# 70
catch (...) 
# 71
{ 
# 72
__tmp->_M_remove_reference(); 
# 73
throw; 
# 74
}  
# 75
return ((locale)(__tmp)); 
# 76
} 
# 78
template< class _CharT, class _Traits, class _Alloc> bool 
# 81
locale::operator()(const basic_string< _CharT, _Traits, _Alloc>  &__s1, const basic_string< _CharT, _Traits, _Alloc>  &
# 82
__s2) const 
# 83
{ 
# 84
typedef std::collate< _CharT>  __collate_type; 
# 85
const __collate_type &__collate = use_facet< std::collate< _CharT> > (*this); 
# 86
return (__collate.compare((__s1.data()), (__s1.data()) + (__s1.length()), (__s2.data()), (__s2.data()) + (__s2.length()))) < 0; 
# 88
} 
# 102
template< class _Facet> bool 
# 104
has_facet(const locale &__loc) throw() 
# 105
{ 
# 106
const size_t __i = (_Facet::id._M_id)(); 
# 107
const locale::facet **__facets = (__loc._M_impl)->_M_facets; 
# 108
return (__i < ((__loc._M_impl)->_M_facets_size)) && (dynamic_cast< const _Facet *>(__facets[__i])); 
# 114
} 
# 130
template< class _Facet> const _Facet &
# 132
use_facet(const locale &__loc) 
# 133
{ 
# 134
const size_t __i = (_Facet::id._M_id)(); 
# 135
const locale::facet **__facets = (__loc._M_impl)->_M_facets; 
# 136
if ((__i >= ((__loc._M_impl)->_M_facets_size)) || (!(__facets[__i]))) { 
# 137
__throw_bad_cast(); }  
# 139
return dynamic_cast< const _Facet &>(*(__facets[__i])); 
# 143
} 
# 147
template< class _CharT> int 
# 149
collate< _CharT> ::_M_compare(const _CharT *, const _CharT *) const throw() 
# 150
{ return 0; } 
# 153
template< class _CharT> size_t 
# 155
collate< _CharT> ::_M_transform(_CharT *, const _CharT *, size_t) const throw() 
# 156
{ return 0; } 
# 158
template< class _CharT> int 
# 161
collate< _CharT> ::do_compare(const _CharT *__lo1, const _CharT *__hi1, const _CharT *
# 162
__lo2, const _CharT *__hi2) const 
# 163
{ 
# 166
const string_type __one(__lo1, __hi1); 
# 167
const string_type __two(__lo2, __hi2); 
# 169
const _CharT *__p = (__one.c_str()); 
# 170
const _CharT *__pend = (__one.data()) + (__one.length()); 
# 171
const _CharT *__q = (__two.c_str()); 
# 172
const _CharT *__qend = (__two.data()) + (__two.length()); 
# 177
for (; ;) 
# 178
{ 
# 179
const int __res = _M_compare(__p, __q); 
# 180
if (__res) { 
# 181
return __res; }  
# 183
__p += char_traits< _CharT> ::length(__p); 
# 184
__q += char_traits< _CharT> ::length(__q); 
# 185
if ((__p == __pend) && (__q == __qend)) { 
# 186
return 0; } else { 
# 187
if (__p == __pend) { 
# 188
return -1; } else { 
# 189
if (__q == __qend) { 
# 190
return 1; }  }  }  
# 192
__p++; 
# 193
__q++; 
# 194
}  
# 195
} 
# 197
template< class _CharT> typename collate< _CharT> ::string_type 
# 200
collate< _CharT> ::do_transform(const _CharT *__lo, const _CharT *__hi) const 
# 201
{ 
# 202
string_type __ret; 
# 205
const string_type __str(__lo, __hi); 
# 207
const _CharT *__p = (__str.c_str()); 
# 208
const _CharT *__pend = (__str.data()) + (__str.length()); 
# 210
size_t __len = (__hi - __lo) * 2; 
# 212
_CharT *__c = new _CharT [__len]; 
# 214
try 
# 215
{ 
# 219
for (; ;) 
# 220
{ 
# 222
size_t __res = _M_transform(__c, __p, __len); 
# 225
if (__res >= __len) 
# 226
{ 
# 227
__len = (__res + (1)); 
# 228
(delete [] __c), (__c = 0); 
# 229
__c = (new _CharT [__len]); 
# 230
__res = _M_transform(__c, __p, __len); 
# 231
}  
# 233
(__ret.append(__c, __res)); 
# 234
__p += char_traits< _CharT> ::length(__p); 
# 235
if (__p == __pend) { 
# 236
break; }  
# 238
__p++; 
# 239
(__ret.push_back(_CharT())); 
# 240
}  
# 241
} 
# 242
catch (...) 
# 243
{ 
# 244
delete [] __c; 
# 245
throw; 
# 246
}  
# 248
delete [] __c; 
# 250
return __ret; 
# 251
} 
# 253
template< class _CharT> long 
# 256
collate< _CharT> ::do_hash(const _CharT *__lo, const _CharT *__hi) const 
# 257
{ 
# 258
unsigned long __val = (0); 
# 259
for (; __lo < __hi; ++__lo) { 
# 260
__val = ((*__lo) + ((__val << 7) | (__val >> (__gnu_cxx::__numeric_traits_integer< unsigned long> ::__digits - 7)))); }  
# 264
return static_cast< long>(__val); 
# 265
} 
# 270
extern template class collate< char> ;
# 271
extern template class collate_byname< char> ;
# 273
extern template const collate< char>  &use_facet< collate< char> > (const locale &);
# 277
extern template bool has_facet< collate< char> > (const locale &) throw();
# 282
extern template class collate< wchar_t> ;
# 283
extern template class collate_byname< wchar_t> ;
# 285
extern template const collate< wchar_t>  &use_facet< collate< wchar_t> > (const locale &);
# 289
extern template bool has_facet< collate< wchar_t> > (const locale &) throw();
# 296
}
# 43 "/usr/include/c++/4.8/bits/ios_base.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 51
enum _Ios_Fmtflags { 
# 53
_S_boolalpha = 1, 
# 54
_S_dec, 
# 55
_S_fixed = 4, 
# 56
_S_hex = 8, 
# 57
_S_internal = 16, 
# 58
_S_left = 32, 
# 59
_S_oct = 64, 
# 60
_S_right = 128, 
# 61
_S_scientific = 256, 
# 62
_S_showbase = 512, 
# 63
_S_showpoint = 1024, 
# 64
_S_showpos = 2048, 
# 65
_S_skipws = 4096, 
# 66
_S_unitbuf = 8192, 
# 67
_S_uppercase = 16384, 
# 68
_S_adjustfield = 176, 
# 69
_S_basefield = 74, 
# 70
_S_floatfield = 260, 
# 71
_S_ios_fmtflags_end = 65536, 
# 72
_S_ios_fmtflags_max = 2147483647, 
# 73
_S_ios_fmtflags_min = (-2147483647-1)
# 74
}; 
# 77
constexpr _Ios_Fmtflags operator&(_Ios_Fmtflags __a, _Ios_Fmtflags __b) 
# 78
{ return (_Ios_Fmtflags)((static_cast< int>(__a)) & (static_cast< int>(__b))); } 
# 81
constexpr _Ios_Fmtflags operator|(_Ios_Fmtflags __a, _Ios_Fmtflags __b) 
# 82
{ return (_Ios_Fmtflags)((static_cast< int>(__a)) | (static_cast< int>(__b))); } 
# 85
constexpr _Ios_Fmtflags operator^(_Ios_Fmtflags __a, _Ios_Fmtflags __b) 
# 86
{ return (_Ios_Fmtflags)((static_cast< int>(__a)) ^ (static_cast< int>(__b))); } 
# 89
constexpr _Ios_Fmtflags operator~(_Ios_Fmtflags __a) 
# 90
{ return (_Ios_Fmtflags)(~(static_cast< int>(__a))); } 
# 93
inline const _Ios_Fmtflags &operator|=(_Ios_Fmtflags &__a, _Ios_Fmtflags __b) 
# 94
{ return __a = ((__a | __b)); } 
# 97
inline const _Ios_Fmtflags &operator&=(_Ios_Fmtflags &__a, _Ios_Fmtflags __b) 
# 98
{ return __a = ((__a & __b)); } 
# 101
inline const _Ios_Fmtflags &operator^=(_Ios_Fmtflags &__a, _Ios_Fmtflags __b) 
# 102
{ return __a = ((__a ^ __b)); } 
# 105
enum _Ios_Openmode { 
# 107
_S_app = 1, 
# 108
_S_ate, 
# 109
_S_bin = 4, 
# 110
_S_in = 8, 
# 111
_S_out = 16, 
# 112
_S_trunc = 32, 
# 113
_S_ios_openmode_end = 65536, 
# 114
_S_ios_openmode_max = 2147483647, 
# 115
_S_ios_openmode_min = (-2147483647-1)
# 116
}; 
# 119
constexpr _Ios_Openmode operator&(_Ios_Openmode __a, _Ios_Openmode __b) 
# 120
{ return (_Ios_Openmode)((static_cast< int>(__a)) & (static_cast< int>(__b))); } 
# 123
constexpr _Ios_Openmode operator|(_Ios_Openmode __a, _Ios_Openmode __b) 
# 124
{ return (_Ios_Openmode)((static_cast< int>(__a)) | (static_cast< int>(__b))); } 
# 127
constexpr _Ios_Openmode operator^(_Ios_Openmode __a, _Ios_Openmode __b) 
# 128
{ return (_Ios_Openmode)((static_cast< int>(__a)) ^ (static_cast< int>(__b))); } 
# 131
constexpr _Ios_Openmode operator~(_Ios_Openmode __a) 
# 132
{ return (_Ios_Openmode)(~(static_cast< int>(__a))); } 
# 135
inline const _Ios_Openmode &operator|=(_Ios_Openmode &__a, _Ios_Openmode __b) 
# 136
{ return __a = ((__a | __b)); } 
# 139
inline const _Ios_Openmode &operator&=(_Ios_Openmode &__a, _Ios_Openmode __b) 
# 140
{ return __a = ((__a & __b)); } 
# 143
inline const _Ios_Openmode &operator^=(_Ios_Openmode &__a, _Ios_Openmode __b) 
# 144
{ return __a = ((__a ^ __b)); } 
# 147
enum _Ios_Iostate { 
# 149
_S_goodbit, 
# 150
_S_badbit, 
# 151
_S_eofbit, 
# 152
_S_failbit = 4, 
# 153
_S_ios_iostate_end = 65536, 
# 154
_S_ios_iostate_max = 2147483647, 
# 155
_S_ios_iostate_min = (-2147483647-1)
# 156
}; 
# 159
constexpr _Ios_Iostate operator&(_Ios_Iostate __a, _Ios_Iostate __b) 
# 160
{ return (_Ios_Iostate)((static_cast< int>(__a)) & (static_cast< int>(__b))); } 
# 163
constexpr _Ios_Iostate operator|(_Ios_Iostate __a, _Ios_Iostate __b) 
# 164
{ return (_Ios_Iostate)((static_cast< int>(__a)) | (static_cast< int>(__b))); } 
# 167
constexpr _Ios_Iostate operator^(_Ios_Iostate __a, _Ios_Iostate __b) 
# 168
{ return (_Ios_Iostate)((static_cast< int>(__a)) ^ (static_cast< int>(__b))); } 
# 171
constexpr _Ios_Iostate operator~(_Ios_Iostate __a) 
# 172
{ return (_Ios_Iostate)(~(static_cast< int>(__a))); } 
# 175
inline const _Ios_Iostate &operator|=(_Ios_Iostate &__a, _Ios_Iostate __b) 
# 176
{ return __a = ((__a | __b)); } 
# 179
inline const _Ios_Iostate &operator&=(_Ios_Iostate &__a, _Ios_Iostate __b) 
# 180
{ return __a = ((__a & __b)); } 
# 183
inline const _Ios_Iostate &operator^=(_Ios_Iostate &__a, _Ios_Iostate __b) 
# 184
{ return __a = ((__a ^ __b)); } 
# 187
enum _Ios_Seekdir { 
# 189
_S_beg, 
# 190
_S_cur, 
# 191
_S_end, 
# 192
_S_ios_seekdir_end = 65536
# 193
}; 
# 205
class ios_base { 
# 215
public: class failure : public exception { 
# 221
public: explicit failure(const string & __str) throw(); 
# 226
virtual ~failure() throw(); 
# 229
virtual const char *what() const throw(); 
# 232
private: string _M_msg; 
# 233
}; 
# 261
typedef _Ios_Fmtflags fmtflags; 
# 264
static const fmtflags boolalpha = _S_boolalpha; 
# 267
static const fmtflags dec = _S_dec; 
# 270
static const fmtflags fixed = _S_fixed; 
# 273
static const fmtflags hex = _S_hex; 
# 278
static const fmtflags internal = _S_internal; 
# 282
static const fmtflags left = _S_left; 
# 285
static const fmtflags oct = _S_oct; 
# 289
static const fmtflags right = _S_right; 
# 292
static const fmtflags scientific = _S_scientific; 
# 296
static const fmtflags showbase = _S_showbase; 
# 300
static const fmtflags showpoint = _S_showpoint; 
# 303
static const fmtflags showpos = _S_showpos; 
# 306
static const fmtflags skipws = _S_skipws; 
# 309
static const fmtflags unitbuf = _S_unitbuf; 
# 313
static const fmtflags uppercase = _S_uppercase; 
# 316
static const fmtflags adjustfield = _S_adjustfield; 
# 319
static const fmtflags basefield = _S_basefield; 
# 322
static const fmtflags floatfield = _S_floatfield; 
# 336
typedef _Ios_Iostate iostate; 
# 340
static const iostate badbit = _S_badbit; 
# 343
static const iostate eofbit = _S_eofbit; 
# 348
static const iostate failbit = _S_failbit; 
# 351
static const iostate goodbit = _S_goodbit; 
# 367
typedef _Ios_Openmode openmode; 
# 370
static const openmode app = _S_app; 
# 373
static const openmode ate = _S_ate; 
# 378
static const openmode binary = _S_bin; 
# 381
static const openmode in = _S_in; 
# 384
static const openmode out = _S_out; 
# 387
static const openmode trunc = _S_trunc; 
# 399
typedef _Ios_Seekdir seekdir; 
# 402
static const seekdir beg = _S_beg; 
# 405
static const seekdir cur = _S_cur; 
# 408
static const seekdir end = _S_end; 
# 411
typedef int io_state; 
# 412
typedef int open_mode; 
# 413
typedef int seek_dir; 
# 415
typedef std::streampos streampos; 
# 416
typedef std::streamoff streamoff; 
# 425
enum event { 
# 427
erase_event, 
# 428
imbue_event, 
# 429
copyfmt_event
# 430
}; 
# 442
typedef void (*event_callback)(event __e, ios_base & __b, int __i); 
# 455
void register_callback(event_callback __fn, int __index); 
# 458
protected: streamsize _M_precision; 
# 459
streamsize _M_width; 
# 460
fmtflags _M_flags; 
# 461
iostate _M_exception; 
# 462
iostate _M_streambuf_state; 
# 466
struct _Callback_list { 
# 469
_Callback_list *_M_next; 
# 470
event_callback _M_fn; 
# 471
int _M_index; 
# 472
_Atomic_word _M_refcount; 
# 474
_Callback_list(event_callback __fn, int __index, _Callback_list *
# 475
__cb) : _M_next(__cb), _M_fn(__fn), _M_index(__index), _M_refcount(0) 
# 476
{ } 
# 479
void _M_add_reference() { __gnu_cxx::__atomic_add_dispatch(&(_M_refcount), 1); } 
# 483
int _M_remove_reference() 
# 484
{ 
# 486
; 
# 487
int __res = __gnu_cxx::__exchange_and_add_dispatch(&(_M_refcount), -1); 
# 488
if (__res == 0) 
# 489
{ 
# 490
; 
# 491
}  
# 492
return __res; 
# 493
} 
# 494
}; 
# 496
_Callback_list *_M_callbacks; 
# 499
void _M_call_callbacks(event __ev) throw(); 
# 502
void _M_dispose_callbacks() throw(); 
# 505
struct _Words { 
# 507
void *_M_pword; 
# 508
long _M_iword; 
# 509
_Words() : _M_pword((0)), _M_iword((0)) { } 
# 510
}; 
# 513
_Words _M_word_zero; 
# 517
enum { _S_local_word_size = 8}; 
# 518
_Words _M_local_word[_S_local_word_size]; 
# 521
int _M_word_size; 
# 522
_Words *_M_word; 
# 525
_Words &_M_grow_words(int __index, bool __iword); 
# 528
locale _M_ios_locale; 
# 531
void _M_init() throw(); 
# 539
public: class Init { 
# 541
friend class ios_base; 
# 543
public: Init(); 
# 544
~Init(); 
# 547
private: static _Atomic_word _S_refcount; 
# 548
static bool _S_synced_with_stdio; 
# 549
}; 
# 557
fmtflags flags() const 
# 558
{ return _M_flags; } 
# 568
fmtflags flags(fmtflags __fmtfl) 
# 569
{ 
# 570
fmtflags __old = _M_flags; 
# 571
(_M_flags) = __fmtfl; 
# 572
return __old; 
# 573
} 
# 584
fmtflags setf(fmtflags __fmtfl) 
# 585
{ 
# 586
fmtflags __old = _M_flags; 
# 587
((_M_flags) |= __fmtfl); 
# 588
return __old; 
# 589
} 
# 601
fmtflags setf(fmtflags __fmtfl, fmtflags __mask) 
# 602
{ 
# 603
fmtflags __old = _M_flags; 
# 604
((_M_flags) &= ((~__mask))); 
# 605
((_M_flags) |= ((__fmtfl & __mask))); 
# 606
return __old; 
# 607
} 
# 616
void unsetf(fmtflags __mask) 
# 617
{ ((_M_flags) &= ((~__mask))); } 
# 627
streamsize precision() const 
# 628
{ return _M_precision; } 
# 636
streamsize precision(streamsize __prec) 
# 637
{ 
# 638
streamsize __old = _M_precision; 
# 639
(_M_precision) = __prec; 
# 640
return __old; 
# 641
} 
# 650
streamsize width() const 
# 651
{ return _M_width; } 
# 659
streamsize width(streamsize __wide) 
# 660
{ 
# 661
streamsize __old = _M_width; 
# 662
(_M_width) = __wide; 
# 663
return __old; 
# 664
} 
# 678
static bool sync_with_stdio(bool __sync = true); 
# 690
locale imbue(const locale & __loc) throw(); 
# 701
locale getloc() const 
# 702
{ return _M_ios_locale; } 
# 712
const locale &_M_getloc() const 
# 713
{ return _M_ios_locale; } 
# 731
static int xalloc() throw(); 
# 747
long &iword(int __ix) 
# 748
{ 
# 749
_Words &__word = (__ix < (_M_word_size)) ? (_M_word)[__ix] : this->_M_grow_words(__ix, true); 
# 751
return __word._M_iword; 
# 752
} 
# 768
void *&pword(int __ix) 
# 769
{ 
# 770
_Words &__word = (__ix < (_M_word_size)) ? (_M_word)[__ix] : this->_M_grow_words(__ix, false); 
# 772
return __word._M_pword; 
# 773
} 
# 784
virtual ~ios_base(); 
# 787
protected: ios_base() throw(); 
# 792
private: ios_base(const ios_base &); 
# 795
ios_base &operator=(const ios_base &); 
# 796
}; 
# 801
inline ios_base &boolalpha(ios_base &__base) 
# 802
{ 
# 803
__base.setf(ios_base::boolalpha); 
# 804
return __base; 
# 805
} 
# 809
inline ios_base &noboolalpha(ios_base &__base) 
# 810
{ 
# 811
__base.unsetf(ios_base::boolalpha); 
# 812
return __base; 
# 813
} 
# 817
inline ios_base &showbase(ios_base &__base) 
# 818
{ 
# 819
__base.setf(ios_base::showbase); 
# 820
return __base; 
# 821
} 
# 825
inline ios_base &noshowbase(ios_base &__base) 
# 826
{ 
# 827
__base.unsetf(ios_base::showbase); 
# 828
return __base; 
# 829
} 
# 833
inline ios_base &showpoint(ios_base &__base) 
# 834
{ 
# 835
__base.setf(ios_base::showpoint); 
# 836
return __base; 
# 837
} 
# 841
inline ios_base &noshowpoint(ios_base &__base) 
# 842
{ 
# 843
__base.unsetf(ios_base::showpoint); 
# 844
return __base; 
# 845
} 
# 849
inline ios_base &showpos(ios_base &__base) 
# 850
{ 
# 851
__base.setf(ios_base::showpos); 
# 852
return __base; 
# 853
} 
# 857
inline ios_base &noshowpos(ios_base &__base) 
# 858
{ 
# 859
__base.unsetf(ios_base::showpos); 
# 860
return __base; 
# 861
} 
# 865
inline ios_base &skipws(ios_base &__base) 
# 866
{ 
# 867
__base.setf(ios_base::skipws); 
# 868
return __base; 
# 869
} 
# 873
inline ios_base &noskipws(ios_base &__base) 
# 874
{ 
# 875
__base.unsetf(ios_base::skipws); 
# 876
return __base; 
# 877
} 
# 881
inline ios_base &uppercase(ios_base &__base) 
# 882
{ 
# 883
__base.setf(ios_base::uppercase); 
# 884
return __base; 
# 885
} 
# 889
inline ios_base &nouppercase(ios_base &__base) 
# 890
{ 
# 891
__base.unsetf(ios_base::uppercase); 
# 892
return __base; 
# 893
} 
# 897
inline ios_base &unitbuf(ios_base &__base) 
# 898
{ 
# 899
__base.setf(ios_base::unitbuf); 
# 900
return __base; 
# 901
} 
# 905
inline ios_base &nounitbuf(ios_base &__base) 
# 906
{ 
# 907
__base.unsetf(ios_base::unitbuf); 
# 908
return __base; 
# 909
} 
# 914
inline ios_base &internal(ios_base &__base) 
# 915
{ 
# 916
__base.setf(ios_base::internal, ios_base::adjustfield); 
# 917
return __base; 
# 918
} 
# 922
inline ios_base &left(ios_base &__base) 
# 923
{ 
# 924
__base.setf(ios_base::left, ios_base::adjustfield); 
# 925
return __base; 
# 926
} 
# 930
inline ios_base &right(ios_base &__base) 
# 931
{ 
# 932
__base.setf(ios_base::right, ios_base::adjustfield); 
# 933
return __base; 
# 934
} 
# 939
inline ios_base &dec(ios_base &__base) 
# 940
{ 
# 941
__base.setf(ios_base::dec, ios_base::basefield); 
# 942
return __base; 
# 943
} 
# 947
inline ios_base &hex(ios_base &__base) 
# 948
{ 
# 949
__base.setf(ios_base::hex, ios_base::basefield); 
# 950
return __base; 
# 951
} 
# 955
inline ios_base &oct(ios_base &__base) 
# 956
{ 
# 957
__base.setf(ios_base::oct, ios_base::basefield); 
# 958
return __base; 
# 959
} 
# 964
inline ios_base &fixed(ios_base &__base) 
# 965
{ 
# 966
__base.setf(ios_base::fixed, ios_base::floatfield); 
# 967
return __base; 
# 968
} 
# 972
inline ios_base &scientific(ios_base &__base) 
# 973
{ 
# 974
__base.setf(ios_base::scientific, ios_base::floatfield); 
# 975
return __base; 
# 976
} 
# 979
}
# 45 "/usr/include/c++/4.8/streambuf" 3
namespace std __attribute((__visibility__("default"))) { 
# 49
template< class _CharT, class _Traits> streamsize __copy_streambufs_eof(basic_streambuf< _CharT, _Traits>  *, basic_streambuf< _CharT, _Traits>  *, bool &); 
# 119
template< class _CharT, class _Traits> 
# 120
class basic_streambuf { 
# 129
public: typedef _CharT char_type; 
# 130
typedef _Traits traits_type; 
# 131
typedef typename _Traits::int_type int_type; 
# 132
typedef typename _Traits::pos_type pos_type; 
# 133
typedef typename _Traits::off_type off_type; 
# 138
typedef basic_streambuf __streambuf_type; 
# 141
friend class basic_ios< _CharT, _Traits> ; 
# 142
friend class basic_istream< _CharT, _Traits> ; 
# 143
friend class basic_ostream< _CharT, _Traits> ; 
# 144
friend class istreambuf_iterator< _CharT, _Traits> ; 
# 145
friend class ostreambuf_iterator< _CharT, _Traits> ; 
# 148
friend streamsize __copy_streambufs_eof<> (basic_streambuf *, basic_streambuf *, bool &); 
# 150
template< bool _IsMove, class _CharT2> friend typename __gnu_cxx::__enable_if< __is_char< _CharT2> ::__value, _CharT2 *> ::__type __copy_move_a2(istreambuf_iterator< _CharT2, char_traits< _CharT2> > , istreambuf_iterator< _CharT2, char_traits< _CharT2> > , _CharT2 *); 
# 156
template< class _CharT2> friend typename __gnu_cxx::__enable_if< __is_char< _CharT2> ::__value, istreambuf_iterator< _CharT2, char_traits< _CharT2> > > ::__type find(istreambuf_iterator< _CharT2, char_traits< _CharT2> > , istreambuf_iterator< _CharT2, char_traits< _CharT2> > , const _CharT2 &); 
# 162
template< class _CharT2, class _Traits2> friend basic_istream< _CharT2, _Traits2>  &operator>>(basic_istream< _CharT2, _Traits2>  &, _CharT2 *); 
# 166
template< class _CharT2, class _Traits2, class _Alloc> friend basic_istream< _CharT2, _Traits2>  &operator>>(basic_istream< _CharT2, _Traits2>  &, basic_string< _CharT2, _Traits2, _Alloc>  &); 
# 171
template< class _CharT2, class _Traits2, class _Alloc> friend basic_istream< _CharT2, _Traits2>  &getline(basic_istream< _CharT2, _Traits2>  &, basic_string< _CharT2, _Traits2, _Alloc>  &, _CharT2); 
# 184
protected: char_type *_M_in_beg; 
# 185
char_type *_M_in_cur; 
# 186
char_type *_M_in_end; 
# 187
char_type *_M_out_beg; 
# 188
char_type *_M_out_cur; 
# 189
char_type *_M_out_end; 
# 192
locale _M_buf_locale; 
# 197
public: virtual ~basic_streambuf() 
# 198
{ } 
# 209
locale pubimbue(const locale &__loc) 
# 210
{ 
# 211
locale __tmp(this->getloc()); 
# 212
this->imbue(__loc); 
# 213
((_M_buf_locale) = __loc); 
# 214
return __tmp; 
# 215
} 
# 226
locale getloc() const 
# 227
{ return _M_buf_locale; } 
# 239
basic_streambuf *pubsetbuf(char_type *__s, streamsize __n) 
# 240
{ return this->setbuf(__s, __n); } 
# 251
pos_type pubseekoff(off_type __off, ios_base::seekdir __way, ios_base::openmode 
# 252
__mode = (ios_base::in | ios_base::out)) 
# 253
{ return this->seekoff(__off, __way, __mode); } 
# 263
pos_type pubseekpos(pos_type __sp, ios_base::openmode 
# 264
__mode = (ios_base::in | ios_base::out)) 
# 265
{ return this->seekpos(__sp, __mode); } 
# 271
int pubsync() { return this->sync(); } 
# 284
streamsize in_avail() 
# 285
{ 
# 286
const streamsize __ret = this->egptr() - this->gptr(); 
# 287
return (__ret) ? __ret : this->showmanyc(); 
# 288
} 
# 298
int_type snextc() 
# 299
{ 
# 300
int_type __ret = traits_type::eof(); 
# 301
if (__builtin_expect(!traits_type::eq_int_type(this->sbumpc(), __ret), true)) { 
# 303
__ret = this->sgetc(); }  
# 304
return __ret; 
# 305
} 
# 316
int_type sbumpc() 
# 317
{ 
# 318
int_type __ret; 
# 319
if (__builtin_expect(this->gptr() < this->egptr(), true)) 
# 320
{ 
# 321
__ret = traits_type::to_int_type(*this->gptr()); 
# 322
this->gbump(1); 
# 323
} else { 
# 325
__ret = this->uflow(); }  
# 326
return __ret; 
# 327
} 
# 338
int_type sgetc() 
# 339
{ 
# 340
int_type __ret; 
# 341
if (__builtin_expect(this->gptr() < this->egptr(), true)) { 
# 342
__ret = traits_type::to_int_type(*this->gptr()); } else { 
# 344
__ret = this->underflow(); }  
# 345
return __ret; 
# 346
} 
# 357
streamsize sgetn(char_type *__s, streamsize __n) 
# 358
{ return this->xsgetn(__s, __n); } 
# 372
int_type sputbackc(char_type __c) 
# 373
{ 
# 374
int_type __ret; 
# 375
const bool __testpos = this->eback() < this->gptr(); 
# 376
if (__builtin_expect((!__testpos) || (!traits_type::eq(__c, this->gptr()[-1])), false)) { 
# 378
__ret = this->pbackfail(traits_type::to_int_type(__c)); } else 
# 380
{ 
# 381
this->gbump(-1); 
# 382
__ret = traits_type::to_int_type(*this->gptr()); 
# 383
}  
# 384
return __ret; 
# 385
} 
# 397
int_type sungetc() 
# 398
{ 
# 399
int_type __ret; 
# 400
if (__builtin_expect(this->eback() < this->gptr(), true)) 
# 401
{ 
# 402
this->gbump(-1); 
# 403
__ret = traits_type::to_int_type(*this->gptr()); 
# 404
} else { 
# 406
__ret = this->pbackfail(); }  
# 407
return __ret; 
# 408
} 
# 424
int_type sputc(char_type __c) 
# 425
{ 
# 426
int_type __ret; 
# 427
if (__builtin_expect(this->pptr() < this->epptr(), true)) 
# 428
{ 
# 429
(*this->pptr()) = __c; 
# 430
this->pbump(1); 
# 431
__ret = traits_type::to_int_type(__c); 
# 432
} else { 
# 434
__ret = this->overflow(traits_type::to_int_type(__c)); }  
# 435
return __ret; 
# 436
} 
# 450
streamsize sputn(const char_type *__s, streamsize __n) 
# 451
{ return this->xsputn(__s, __n); } 
# 463
protected: basic_streambuf() : _M_in_beg((0)), _M_in_cur((0)), _M_in_end((0)), _M_out_beg((0)), _M_out_cur((0)), _M_out_end((0)), _M_buf_locale(locale()) 
# 467
{ } 
# 482
char_type *eback() const { return _M_in_beg; } 
# 485
char_type *gptr() const { return _M_in_cur; } 
# 488
char_type *egptr() const { return _M_in_end; } 
# 498
void gbump(int __n) { (_M_in_cur) += __n; } 
# 509
void setg(char_type *__gbeg, char_type *__gnext, char_type *__gend) 
# 510
{ 
# 511
(_M_in_beg) = __gbeg; 
# 512
(_M_in_cur) = __gnext; 
# 513
(_M_in_end) = __gend; 
# 514
} 
# 529
char_type *pbase() const { return _M_out_beg; } 
# 532
char_type *pptr() const { return _M_out_cur; } 
# 535
char_type *epptr() const { return _M_out_end; } 
# 545
void pbump(int __n) { (_M_out_cur) += __n; } 
# 555
void setp(char_type *__pbeg, char_type *__pend) 
# 556
{ 
# 557
(_M_out_beg) = ((_M_out_cur) = __pbeg); 
# 558
(_M_out_end) = __pend; 
# 559
} 
# 576
virtual void imbue(const locale &__loc) 
# 577
{ } 
# 591
virtual basic_streambuf *setbuf(char_type *, streamsize) 
# 592
{ return this; } 
# 602
virtual pos_type seekoff(off_type, ios_base::seekdir, ios_base::openmode = (ios_base::in | ios_base::out)) 
# 604
{ return (pos_type)((off_type)(-1)); } 
# 614
virtual pos_type seekpos(pos_type, ios_base::openmode = (ios_base::in | ios_base::out)) 
# 616
{ return (pos_type)((off_type)(-1)); } 
# 627
virtual int sync() { return 0; } 
# 649
virtual streamsize showmanyc() { return 0; } 
# 665
virtual streamsize xsgetn(char_type * __s, streamsize __n); 
# 687
virtual int_type underflow() 
# 688
{ return traits_type::eof(); } 
# 700
virtual int_type uflow() 
# 701
{ 
# 702
int_type __ret = traits_type::eof(); 
# 703
const bool __testeof = traits_type::eq_int_type(this->underflow(), __ret); 
# 705
if (!__testeof) 
# 706
{ 
# 707
__ret = traits_type::to_int_type(*this->gptr()); 
# 708
this->gbump(1); 
# 709
}  
# 710
return __ret; 
# 711
} 
# 724
virtual int_type pbackfail(int_type __c = traits_type::eof()) 
# 725
{ return traits_type::eof(); } 
# 742
virtual streamsize xsputn(const char_type * __s, streamsize __n); 
# 768
virtual int_type overflow(int_type __c = traits_type::eof()) 
# 769
{ return traits_type::eof(); } 
# 783
public: void stossc() 
# 784
{ 
# 785
if (this->gptr() < this->egptr()) { 
# 786
this->gbump(1); } else { 
# 788
this->uflow(); }  
# 789
} 
# 794
void __safe_gbump(streamsize __n) { (_M_in_cur) += __n; } 
# 797
void __safe_pbump(streamsize __n) { (_M_out_cur) += __n; } 
# 802
private: basic_streambuf(const basic_streambuf &__sb) : _M_in_beg(__sb._M_in_beg), _M_in_cur(__sb._M_in_cur), _M_in_end(__sb._M_in_end), _M_out_beg(__sb._M_out_beg), _M_out_cur(__sb._M_out_cur), _M_out_end(__sb._M_out_cur), _M_buf_locale(__sb._M_buf_locale) 
# 807
{ } 
# 810
basic_streambuf &operator=(const basic_streambuf &) { return *this; } 
# 811
}; 
# 816
template<> streamsize __copy_streambufs_eof(basic_streambuf< char, char_traits< char> >  * __sbin, basic_streambuf< char, char_traits< char> >  * __sbout, bool & __ineof); 
# 821
template<> streamsize __copy_streambufs_eof(basic_streambuf< wchar_t, char_traits< wchar_t> >  * __sbin, basic_streambuf< wchar_t, char_traits< wchar_t> >  * __sbout, bool & __ineof); 
# 826
}
# 39 "/usr/include/c++/4.8/bits/streambuf.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 43
template< class _CharT, class _Traits> streamsize 
# 46
basic_streambuf< _CharT, _Traits> ::xsgetn(char_type *__s, streamsize __n) 
# 47
{ 
# 48
streamsize __ret = (0); 
# 49
while (__ret < __n) 
# 50
{ 
# 51
const streamsize __buf_len = this->egptr() - this->gptr(); 
# 52
if (__buf_len) 
# 53
{ 
# 54
const streamsize __remaining = __n - __ret; 
# 55
const streamsize __len = std::min(__buf_len, __remaining); 
# 56
traits_type::copy(__s, this->gptr(), __len); 
# 57
__ret += __len; 
# 58
__s += __len; 
# 59
this->__safe_gbump(__len); 
# 60
}  
# 62
if (__ret < __n) 
# 63
{ 
# 64
const int_type __c = this->uflow(); 
# 65
if (!traits_type::eq_int_type(__c, traits_type::eof())) 
# 66
{ 
# 67
traits_type::assign(*(__s++), traits_type::to_char_type(__c)); 
# 68
++__ret; 
# 69
} else { 
# 71
break; }  
# 72
}  
# 73
}  
# 74
return __ret; 
# 75
} 
# 77
template< class _CharT, class _Traits> streamsize 
# 80
basic_streambuf< _CharT, _Traits> ::xsputn(const char_type *__s, streamsize __n) 
# 81
{ 
# 82
streamsize __ret = (0); 
# 83
while (__ret < __n) 
# 84
{ 
# 85
const streamsize __buf_len = this->epptr() - this->pptr(); 
# 86
if (__buf_len) 
# 87
{ 
# 88
const streamsize __remaining = __n - __ret; 
# 89
const streamsize __len = std::min(__buf_len, __remaining); 
# 90
traits_type::copy(this->pptr(), __s, __len); 
# 91
__ret += __len; 
# 92
__s += __len; 
# 93
this->__safe_pbump(__len); 
# 94
}  
# 96
if (__ret < __n) 
# 97
{ 
# 98
int_type __c = this->overflow(traits_type::to_int_type(*__s)); 
# 99
if (!traits_type::eq_int_type(__c, traits_type::eof())) 
# 100
{ 
# 101
++__ret; 
# 102
++__s; 
# 103
} else { 
# 105
break; }  
# 106
}  
# 107
}  
# 108
return __ret; 
# 109
} 
# 114
template< class _CharT, class _Traits> streamsize 
# 116
__copy_streambufs_eof(basic_streambuf< _CharT, _Traits>  *__sbin, basic_streambuf< _CharT, _Traits>  *
# 117
__sbout, bool &
# 118
__ineof) 
# 119
{ 
# 120
streamsize __ret = (0); 
# 121
__ineof = true; 
# 122
typename _Traits::int_type __c = (__sbin->sgetc()); 
# 123
while (!_Traits::eq_int_type(__c, _Traits::eof())) 
# 124
{ 
# 125
__c = (__sbout->sputc(_Traits::to_char_type(__c))); 
# 126
if (_Traits::eq_int_type(__c, _Traits::eof())) 
# 127
{ 
# 128
__ineof = false; 
# 129
break; 
# 130
}  
# 131
++__ret; 
# 132
__c = (__sbin->snextc()); 
# 133
}  
# 134
return __ret; 
# 135
} 
# 137
template< class _CharT, class _Traits> inline streamsize 
# 139
__copy_streambufs(basic_streambuf< _CharT, _Traits>  *__sbin, basic_streambuf< _CharT, _Traits>  *
# 140
__sbout) 
# 141
{ 
# 142
bool __ineof; 
# 143
return __copy_streambufs_eof(__sbin, __sbout, __ineof); 
# 144
} 
# 149
extern template class basic_streambuf< char, char_traits< char> > ;
# 150
extern template streamsize __copy_streambufs(basic_streambuf< char, char_traits< char> >  * __sbin, basic_streambuf< char, char_traits< char> >  * __sbout);
# 154
extern template streamsize __copy_streambufs_eof< char, char_traits< char> > (basic_streambuf< char, char_traits< char> >  *, basic_streambuf< char, char_traits< char> >  *, bool &);
# 160
extern template class basic_streambuf< wchar_t, char_traits< wchar_t> > ;
# 161
extern template streamsize __copy_streambufs(basic_streambuf< wchar_t, char_traits< wchar_t> >  * __sbin, basic_streambuf< wchar_t, char_traits< wchar_t> >  * __sbout);
# 165
extern template streamsize __copy_streambufs_eof< wchar_t, char_traits< wchar_t> > (basic_streambuf< wchar_t, char_traits< wchar_t> >  *, basic_streambuf< wchar_t, char_traits< wchar_t> >  *, bool &);
# 173
}
# 52 "/usr/include/wctype.h" 3
typedef unsigned long wctype_t; 
# 72
enum { 
# 73
__ISwupper, 
# 74
__ISwlower, 
# 75
__ISwalpha, 
# 76
__ISwdigit, 
# 77
__ISwxdigit, 
# 78
__ISwspace, 
# 79
__ISwprint, 
# 80
__ISwgraph, 
# 81
__ISwblank, 
# 82
__ISwcntrl, 
# 83
__ISwpunct, 
# 84
__ISwalnum, 
# 86
_ISwupper = 16777216, 
# 87
_ISwlower = 33554432, 
# 88
_ISwalpha = 67108864, 
# 89
_ISwdigit = 134217728, 
# 90
_ISwxdigit = 268435456, 
# 91
_ISwspace = 536870912, 
# 92
_ISwprint = 1073741824, 
# 93
_ISwgraph = (-2147483647-1), 
# 94
_ISwblank = 65536, 
# 95
_ISwcntrl = 131072, 
# 96
_ISwpunct = 262144, 
# 97
_ISwalnum = 524288
# 98
}; 
# 102
extern "C" {
# 111
extern int iswalnum(wint_t __wc) throw(); 
# 117
extern int iswalpha(wint_t __wc) throw(); 
# 120
extern int iswcntrl(wint_t __wc) throw(); 
# 124
extern int iswdigit(wint_t __wc) throw(); 
# 128
extern int iswgraph(wint_t __wc) throw(); 
# 133
extern int iswlower(wint_t __wc) throw(); 
# 136
extern int iswprint(wint_t __wc) throw(); 
# 141
extern int iswpunct(wint_t __wc) throw(); 
# 146
extern int iswspace(wint_t __wc) throw(); 
# 151
extern int iswupper(wint_t __wc) throw(); 
# 156
extern int iswxdigit(wint_t __wc) throw(); 
# 162
extern int iswblank(wint_t __wc) throw(); 
# 171
extern wctype_t wctype(const char * __property) throw(); 
# 175
extern int iswctype(wint_t __wc, wctype_t __desc) throw(); 
# 186
typedef const __int32_t *wctrans_t; 
# 194
extern wint_t towlower(wint_t __wc) throw(); 
# 197
extern wint_t towupper(wint_t __wc) throw(); 
# 200
}
# 213
extern "C" {
# 218
extern wctrans_t wctrans(const char * __property) throw(); 
# 221
extern wint_t towctrans(wint_t __wc, wctrans_t __desc) throw(); 
# 230
extern int iswalnum_l(wint_t __wc, __locale_t __locale) throw(); 
# 236
extern int iswalpha_l(wint_t __wc, __locale_t __locale) throw(); 
# 239
extern int iswcntrl_l(wint_t __wc, __locale_t __locale) throw(); 
# 243
extern int iswdigit_l(wint_t __wc, __locale_t __locale) throw(); 
# 247
extern int iswgraph_l(wint_t __wc, __locale_t __locale) throw(); 
# 252
extern int iswlower_l(wint_t __wc, __locale_t __locale) throw(); 
# 255
extern int iswprint_l(wint_t __wc, __locale_t __locale) throw(); 
# 260
extern int iswpunct_l(wint_t __wc, __locale_t __locale) throw(); 
# 265
extern int iswspace_l(wint_t __wc, __locale_t __locale) throw(); 
# 270
extern int iswupper_l(wint_t __wc, __locale_t __locale) throw(); 
# 275
extern int iswxdigit_l(wint_t __wc, __locale_t __locale) throw(); 
# 280
extern int iswblank_l(wint_t __wc, __locale_t __locale) throw(); 
# 284
extern wctype_t wctype_l(const char * __property, __locale_t __locale) throw(); 
# 289
extern int iswctype_l(wint_t __wc, wctype_t __desc, __locale_t __locale) throw(); 
# 298
extern wint_t towlower_l(wint_t __wc, __locale_t __locale) throw(); 
# 301
extern wint_t towupper_l(wint_t __wc, __locale_t __locale) throw(); 
# 305
extern wctrans_t wctrans_l(const char * __property, __locale_t __locale) throw(); 
# 309
extern wint_t towctrans_l(wint_t __wc, wctrans_t __desc, __locale_t __locale) throw(); 
# 314
}
# 80 "/usr/include/c++/4.8/cwctype" 3
namespace std { 
# 82
using ::wctrans_t;
# 83
using ::wctype_t;
# 86
using ::iswalnum;
# 87
using ::iswalpha;
# 89
using ::iswblank;
# 91
using ::iswcntrl;
# 92
using ::iswctype;
# 93
using ::iswdigit;
# 94
using ::iswgraph;
# 95
using ::iswlower;
# 96
using ::iswprint;
# 97
using ::iswpunct;
# 98
using ::iswspace;
# 99
using ::iswupper;
# 100
using ::iswxdigit;
# 101
using ::towctrans;
# 102
using ::towlower;
# 103
using ::towupper;
# 104
using ::wctrans;
# 105
using ::wctype;
# 106
}
# 36 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/ctype_base.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 41
struct ctype_base { 
# 44
typedef const int *__to_type; 
# 48
typedef unsigned short mask; 
# 49
static const mask upper = (_ISupper); 
# 50
static const mask lower = (_ISlower); 
# 51
static const mask alpha = (_ISalpha); 
# 52
static const mask digit = (_ISdigit); 
# 53
static const mask xdigit = (_ISxdigit); 
# 54
static const mask space = (_ISspace); 
# 55
static const mask print = (_ISprint); 
# 56
static const mask graph = (((_ISalpha) | (_ISdigit)) | (_ISpunct)); 
# 57
static const mask cntrl = (_IScntrl); 
# 58
static const mask punct = (_ISpunct); 
# 59
static const mask alnum = ((_ISalpha) | (_ISdigit)); 
# 60
}; 
# 63
}
# 38 "/usr/include/c++/4.8/bits/streambuf_iterator.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 49
template< class _CharT, class _Traits> 
# 50
class istreambuf_iterator : public iterator< input_iterator_tag, _CharT, typename _Traits::off_type, _CharT *, _CharT>  { 
# 64
public: typedef _CharT char_type; 
# 65
typedef _Traits traits_type; 
# 66
typedef typename _Traits::int_type int_type; 
# 67
typedef basic_streambuf< _CharT, _Traits>  streambuf_type; 
# 68
typedef basic_istream< _CharT, _Traits>  istream_type; 
# 71
template< class _CharT2> friend typename ::__gnu_cxx::__enable_if< __is_char< _CharT2> ::__value, ostreambuf_iterator< _CharT2, char_traits< _CharT2> > > ::__type copy(::std::istreambuf_iterator< _CharT2, char_traits< _CharT2> > , ::std::istreambuf_iterator< _CharT2, char_traits< _CharT2> > , ostreambuf_iterator< _CharT2, char_traits< _CharT2> > ); 
# 77
template< bool _IsMove, class _CharT2> friend typename ::__gnu_cxx::__enable_if< __is_char< _CharT2> ::__value, _CharT2 *> ::__type __copy_move_a2(::std::istreambuf_iterator< _CharT2, char_traits< _CharT2> > , ::std::istreambuf_iterator< _CharT2, char_traits< _CharT2> > , _CharT2 *); 
# 83
template< class _CharT2> friend typename ::__gnu_cxx::__enable_if< __is_char< _CharT2> ::__value, ::std::istreambuf_iterator< _CharT2, char_traits< _CharT2> > > ::__type find(::std::istreambuf_iterator< _CharT2, char_traits< _CharT2> > , ::std::istreambuf_iterator< _CharT2, char_traits< _CharT2> > , const _CharT2 &); 
# 97
private: mutable streambuf_type *_M_sbuf; 
# 98
mutable int_type _M_c; 
# 102
public: constexpr istreambuf_iterator() noexcept : _M_sbuf((0)), _M_c(traits_type::eof()) 
# 103
{ } 
# 106
istreambuf_iterator(const istreambuf_iterator &) noexcept = default;
# 108
~istreambuf_iterator() = default;
# 112
istreambuf_iterator(istream_type &__s) noexcept : _M_sbuf((__s.rdbuf())), _M_c(traits_type::eof()) 
# 113
{ } 
# 116
istreambuf_iterator(streambuf_type *__s) noexcept : _M_sbuf(__s), _M_c(traits_type::eof()) 
# 117
{ } 
# 123
char_type operator*() const 
# 124
{ 
# 132
return traits_type::to_char_type(_M_get()); 
# 133
} 
# 137
istreambuf_iterator &operator++() 
# 138
{ 
# 141
; 
# 142
if (_M_sbuf) 
# 143
{ 
# 144
((_M_sbuf)->sbumpc()); 
# 145
(_M_c) = traits_type::eof(); 
# 146
}  
# 147
return *this; 
# 148
} 
# 152
istreambuf_iterator operator++(int) 
# 153
{ 
# 156
; 
# 158
istreambuf_iterator __old = *this; 
# 159
if (_M_sbuf) 
# 160
{ 
# 161
(__old._M_c) = ((_M_sbuf)->sbumpc()); 
# 162
(_M_c) = traits_type::eof(); 
# 163
}  
# 164
return __old; 
# 165
} 
# 172
bool equal(const istreambuf_iterator &__b) const 
# 173
{ return _M_at_eof() == __b._M_at_eof(); } 
# 177
private: int_type _M_get() const 
# 178
{ 
# 179
const int_type __eof = traits_type::eof(); 
# 180
int_type __ret = __eof; 
# 181
if (_M_sbuf) 
# 182
{ 
# 183
if (!traits_type::eq_int_type(_M_c, __eof)) { 
# 184
__ret = (_M_c); } else { 
# 185
if (!traits_type::eq_int_type(__ret = ((_M_sbuf)->sgetc()), __eof)) { 
# 187
(_M_c) = __ret; } else { 
# 189
(_M_sbuf) = 0; }  }  
# 190
}  
# 191
return __ret; 
# 192
} 
# 195
bool _M_at_eof() const 
# 196
{ 
# 197
const int_type __eof = traits_type::eof(); 
# 198
return traits_type::eq_int_type(_M_get(), __eof); 
# 199
} 
# 200
}; 
# 202
template< class _CharT, class _Traits> inline bool 
# 204
operator==(const istreambuf_iterator< _CharT, _Traits>  &__a, const istreambuf_iterator< _CharT, _Traits>  &
# 205
__b) 
# 206
{ return (__a.equal(__b)); } 
# 208
template< class _CharT, class _Traits> inline bool 
# 210
operator!=(const istreambuf_iterator< _CharT, _Traits>  &__a, const istreambuf_iterator< _CharT, _Traits>  &
# 211
__b) 
# 212
{ return !(__a.equal(__b)); } 
# 215
template< class _CharT, class _Traits> 
# 216
class ostreambuf_iterator : public iterator< output_iterator_tag, void, void, void, void>  { 
# 223
public: typedef _CharT char_type; 
# 224
typedef _Traits traits_type; 
# 225
typedef basic_streambuf< _CharT, _Traits>  streambuf_type; 
# 226
typedef basic_ostream< _CharT, _Traits>  ostream_type; 
# 229
template< class _CharT2> friend typename __gnu_cxx::__enable_if< __is_char< _CharT2> ::__value, std::ostreambuf_iterator< _CharT2, char_traits< _CharT2> > > ::__type copy(istreambuf_iterator< _CharT2, char_traits< _CharT2> > , istreambuf_iterator< _CharT2, char_traits< _CharT2> > , std::ostreambuf_iterator< _CharT2, char_traits< _CharT2> > ); 
# 236
private: streambuf_type *_M_sbuf; 
# 237
bool _M_failed; 
# 241
public: ostreambuf_iterator(ostream_type &__s) noexcept : _M_sbuf((__s.rdbuf())), _M_failed(!(_M_sbuf)) 
# 242
{ } 
# 245
ostreambuf_iterator(streambuf_type *__s) noexcept : _M_sbuf(__s), _M_failed(!(_M_sbuf)) 
# 246
{ } 
# 250
ostreambuf_iterator &operator=(_CharT __c) 
# 251
{ 
# 252
if ((!(_M_failed)) && _Traits::eq_int_type(((_M_sbuf)->sputc(__c)), _Traits::eof())) { 
# 254
(_M_failed) = true; }  
# 255
return *this; 
# 256
} 
# 260
ostreambuf_iterator &operator*() 
# 261
{ return *this; } 
# 265
ostreambuf_iterator &operator++(int) 
# 266
{ return *this; } 
# 270
ostreambuf_iterator &operator++() 
# 271
{ return *this; } 
# 275
bool failed() const noexcept 
# 276
{ return _M_failed; } 
# 279
ostreambuf_iterator &_M_put(const _CharT *__ws, streamsize __len) 
# 280
{ 
# 281
if ((__builtin_expect(!(_M_failed), true)) && (__builtin_expect(((this->_M_sbuf)->sputn(__ws, __len)) != __len, false))) { 
# 284
(_M_failed) = true; }  
# 285
return *this; 
# 286
} 
# 287
}; 
# 290
template< class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type 
# 293
copy(istreambuf_iterator< _CharT, char_traits< _CharT> >  __first, istreambuf_iterator< _CharT, char_traits< _CharT> >  
# 294
__last, ostreambuf_iterator< _CharT, char_traits< _CharT> >  
# 295
__result) 
# 296
{ 
# 297
if ((__first._M_sbuf) && (!(__last._M_sbuf)) && (!(__result._M_failed))) 
# 298
{ 
# 299
bool __ineof; 
# 300
__copy_streambufs_eof((__first._M_sbuf), (__result._M_sbuf), __ineof); 
# 301
if (!__ineof) { 
# 302
(__result._M_failed) = true; }  
# 303
}  
# 304
return __result; 
# 305
} 
# 307
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type 
# 310
__copy_move_a2(_CharT *__first, _CharT *__last, ostreambuf_iterator< _CharT, char_traits< _CharT> >  
# 311
__result) 
# 312
{ 
# 313
const streamsize __num = __last - __first; 
# 314
if (__num > (0)) { 
# 315
(__result._M_put(__first, __num)); }  
# 316
return __result; 
# 317
} 
# 319
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, ostreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type 
# 322
__copy_move_a2(const _CharT *__first, const _CharT *__last, ostreambuf_iterator< _CharT, char_traits< _CharT> >  
# 323
__result) 
# 324
{ 
# 325
const streamsize __num = __last - __first; 
# 326
if (__num > (0)) { 
# 327
(__result._M_put(__first, __num)); }  
# 328
return __result; 
# 329
} 
# 331
template< bool _IsMove, class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, _CharT *> ::__type 
# 334
__copy_move_a2(istreambuf_iterator< _CharT, char_traits< _CharT> >  __first, istreambuf_iterator< _CharT, char_traits< _CharT> >  
# 335
__last, _CharT *__result) 
# 336
{ 
# 337
typedef istreambuf_iterator< _CharT, char_traits< _CharT> >  __is_iterator_type; 
# 338
typedef typename istreambuf_iterator< _CharT, char_traits< _CharT> > ::traits_type traits_type; 
# 339
typedef typename istreambuf_iterator< _CharT, char_traits< _CharT> > ::streambuf_type streambuf_type; 
# 340
typedef typename istreambuf_iterator< _CharT, char_traits< _CharT> > ::traits_type::int_type int_type; 
# 342
if ((__first._M_sbuf) && (!(__last._M_sbuf))) 
# 343
{ 
# 344
streambuf_type *__sb = ((__first._M_sbuf)); 
# 345
int_type __c = (__sb->sgetc()); 
# 346
while (!traits_type::eq_int_type(__c, traits_type::eof())) 
# 347
{ 
# 348
const streamsize __n = (__sb->egptr()) - (__sb->gptr()); 
# 349
if (__n > (1)) 
# 350
{ 
# 351
traits_type::copy(__result, (__sb->gptr()), __n); 
# 352
(__sb->__safe_gbump(__n)); 
# 353
__result += __n; 
# 354
__c = (__sb->underflow()); 
# 355
} else 
# 357
{ 
# 358
(*(__result++)) = traits_type::to_char_type(__c); 
# 359
__c = (__sb->snextc()); 
# 360
}  
# 361
}  
# 362
}  
# 363
return __result; 
# 364
} 
# 366
template< class _CharT> typename __gnu_cxx::__enable_if< __is_char< _CharT> ::__value, istreambuf_iterator< _CharT, char_traits< _CharT> > > ::__type 
# 369
find(istreambuf_iterator< _CharT, char_traits< _CharT> >  __first, istreambuf_iterator< _CharT, char_traits< _CharT> >  
# 370
__last, const _CharT &__val) 
# 371
{ 
# 372
typedef istreambuf_iterator< _CharT, char_traits< _CharT> >  __is_iterator_type; 
# 373
typedef typename istreambuf_iterator< _CharT, char_traits< _CharT> > ::traits_type traits_type; 
# 374
typedef typename istreambuf_iterator< _CharT, char_traits< _CharT> > ::streambuf_type streambuf_type; 
# 375
typedef typename istreambuf_iterator< _CharT, char_traits< _CharT> > ::traits_type::int_type int_type; 
# 377
if ((__first._M_sbuf) && (!(__last._M_sbuf))) 
# 378
{ 
# 379
const int_type __ival = traits_type::to_int_type(__val); 
# 380
streambuf_type *__sb = ((__first._M_sbuf)); 
# 381
int_type __c = (__sb->sgetc()); 
# 382
while ((!traits_type::eq_int_type(__c, traits_type::eof())) && (!traits_type::eq_int_type(__c, __ival))) 
# 384
{ 
# 385
streamsize __n = (__sb->egptr()) - (__sb->gptr()); 
# 386
if (__n > (1)) 
# 387
{ 
# 388
const _CharT *__p = traits_type::find((__sb->gptr()), __n, __val); 
# 390
if (__p) { 
# 391
__n = (__p - (__sb->gptr())); }  
# 392
(__sb->__safe_gbump(__n)); 
# 393
__c = (__sb->sgetc()); 
# 394
} else { 
# 396
__c = (__sb->snextc()); }  
# 397
}  
# 399
if (!traits_type::eq_int_type(__c, traits_type::eof())) { 
# 400
(__first._M_c) = __c; } else { 
# 402
(__first._M_sbuf) = 0; }  
# 403
}  
# 404
return __first; 
# 405
} 
# 410
}
# 50 "/usr/include/c++/4.8/bits/locale_facets.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 64
template< class _Tp> void __convert_to_v(const char *, _Tp &, ios_base::iostate &, const __c_locale &) throw(); 
# 72
template<> void __convert_to_v(const char *, float &, ios_base::iostate &, const __c_locale &) throw(); 
# 77
template<> void __convert_to_v(const char *, double &, ios_base::iostate &, const __c_locale &) throw(); 
# 82
template<> void __convert_to_v(const char *, long double &, ios_base::iostate &, const __c_locale &) throw(); 
# 87
template< class _CharT, class _Traits> 
# 88
struct __pad { 
# 91
static void _S_pad(ios_base & __io, _CharT __fill, _CharT * __news, const _CharT * __olds, streamsize __newlen, streamsize __oldlen); 
# 93
}; 
# 100
template< class _CharT> _CharT *__add_grouping(_CharT * __s, _CharT __sep, const char * __gbeg, size_t __gsize, const _CharT * __first, const _CharT * __last); 
# 109
template< class _CharT> inline ostreambuf_iterator< _CharT, char_traits< _CharT> >  
# 112
__write(ostreambuf_iterator< _CharT, char_traits< _CharT> >  __s, const _CharT *__ws, int __len) 
# 113
{ 
# 114
(__s._M_put(__ws, __len)); 
# 115
return __s; 
# 116
} 
# 119
template< class _CharT, class _OutIter> inline _OutIter 
# 122
__write(_OutIter __s, const _CharT *__ws, int __len) 
# 123
{ 
# 124
for (int __j = 0; __j < __len; (__j++), (++__s)) { 
# 125
(*__s) = (__ws[__j]); }  
# 126
return __s; 
# 127
} 
# 142
template< class _CharT> 
# 143
class __ctype_abstract_base : public locale::facet, public ctype_base { 
# 148
public: typedef _CharT char_type; 
# 162
bool is(mask __m, char_type __c) const 
# 163
{ return (this->do_is(__m, __c)); } 
# 179
const char_type *is(const char_type *__lo, const char_type *__hi, mask *__vec) const 
# 180
{ return (this->do_is(__lo, __hi, __vec)); } 
# 195
const char_type *scan_is(mask __m, const char_type *__lo, const char_type *__hi) const 
# 196
{ return this->do_scan_is(__m, __lo, __hi); } 
# 211
const char_type *scan_not(mask __m, const char_type *__lo, const char_type *__hi) const 
# 212
{ return this->do_scan_not(__m, __lo, __hi); } 
# 225
char_type toupper(char_type __c) const 
# 226
{ return (this->do_toupper(__c)); } 
# 240
const char_type *toupper(char_type *__lo, const char_type *__hi) const 
# 241
{ return (this->do_toupper(__lo, __hi)); } 
# 254
char_type tolower(char_type __c) const 
# 255
{ return (this->do_tolower(__c)); } 
# 269
const char_type *tolower(char_type *__lo, const char_type *__hi) const 
# 270
{ return (this->do_tolower(__lo, __hi)); } 
# 286
char_type widen(char __c) const 
# 287
{ return (this->do_widen(__c)); } 
# 305
const char *widen(const char *__lo, const char *__hi, char_type *__to) const 
# 306
{ return (this->do_widen(__lo, __hi, __to)); } 
# 324
char narrow(char_type __c, char __dfault) const 
# 325
{ return (this->do_narrow(__c, __dfault)); } 
# 346
const char_type *narrow(const char_type *__lo, const char_type *__hi, char 
# 347
__dfault, char *__to) const 
# 348
{ return (this->do_narrow(__lo, __hi, __dfault, __to)); } 
# 352
protected: explicit __ctype_abstract_base(size_t __refs = 0) : locale::facet(__refs) { } 
# 355
virtual ~__ctype_abstract_base() { } 
# 371
virtual bool do_is(mask __m, char_type __c) const = 0; 
# 390
virtual const char_type *do_is(const char_type * __lo, const char_type * __hi, mask * __vec) const = 0; 
# 409
virtual const char_type *do_scan_is(mask __m, const char_type * __lo, const char_type * __hi) const = 0; 
# 428
virtual const char_type *do_scan_not(mask __m, const char_type * __lo, const char_type * __hi) const = 0; 
# 446
virtual char_type do_toupper(char_type __c) const = 0; 
# 463
virtual const char_type *do_toupper(char_type * __lo, const char_type * __hi) const = 0; 
# 479
virtual char_type do_tolower(char_type __c) const = 0; 
# 496
virtual const char_type *do_tolower(char_type * __lo, const char_type * __hi) const = 0; 
# 515
virtual char_type do_widen(char __c) const = 0; 
# 536
virtual const char *do_widen(const char * __lo, const char * __hi, char_type * __to) const = 0; 
# 557
virtual char do_narrow(char_type __c, char __dfault) const = 0; 
# 582
virtual const char_type *do_narrow(const char_type * __lo, const char_type * __hi, char __dfault, char * __to) const = 0; 
# 584
}; 
# 604
template< class _CharT> 
# 605
class ctype : public __ctype_abstract_base< _CharT>  { 
# 609
public: typedef _CharT char_type; 
# 610
typedef typename ::std::__ctype_abstract_base< _CharT> ::mask mask; 
# 613
static ::std::locale::id id; 
# 616
explicit ctype(::std::size_t __refs = 0) : ::std::__ctype_abstract_base< _CharT> (__refs) { } 
# 620
protected: virtual ~ctype(); 
# 623
virtual bool do_is(mask __m, char_type __c) const; 
# 626
virtual const char_type *do_is(const char_type * __lo, const char_type * __hi, mask * __vec) const; 
# 629
virtual const char_type *do_scan_is(mask __m, const char_type * __lo, const char_type * __hi) const; 
# 632
virtual const char_type *do_scan_not(mask __m, const char_type * __lo, const char_type * __hi) const; 
# 636
virtual char_type do_toupper(char_type __c) const; 
# 639
virtual const char_type *do_toupper(char_type * __lo, const char_type * __hi) const; 
# 642
virtual char_type do_tolower(char_type __c) const; 
# 645
virtual const char_type *do_tolower(char_type * __lo, const char_type * __hi) const; 
# 648
virtual char_type do_widen(char __c) const; 
# 651
virtual const char *do_widen(const char * __lo, const char * __hi, char_type * __dest) const; 
# 654
virtual char do_narrow(char_type, char __dfault) const; 
# 657
virtual const char_type *do_narrow(const char_type * __lo, const char_type * __hi, char __dfault, char * __to) const; 
# 659
}; 
# 661
template< class _CharT> locale::id 
# 662
ctype< _CharT> ::id; 
# 674
template<> class ctype< char>  : public locale::facet, public ctype_base { 
# 679
public: typedef char char_type; 
# 683
protected: __c_locale _M_c_locale_ctype; 
# 684
bool _M_del; 
# 685
__to_type _M_toupper; 
# 686
__to_type _M_tolower; 
# 687
const mask *_M_table; 
# 688
mutable char _M_widen_ok; 
# 689
mutable char _M_widen[1 + (static_cast< unsigned char>(-1))]; 
# 690
mutable char _M_narrow[1 + (static_cast< unsigned char>(-1))]; 
# 691
mutable char _M_narrow_ok; 
# 696
public: static locale::id id; 
# 698
static const size_t table_size = (1 + (static_cast< unsigned char>(-1))); 
# 711
explicit ctype(const mask * __table = 0, bool __del = false, size_t __refs = 0); 
# 724
explicit ctype(__c_locale __cloc, const mask * __table = 0, bool __del = false, size_t __refs = 0); 
# 737
inline bool is(mask __m, char __c) const; 
# 752
inline const char *is(const char * __lo, const char * __hi, mask * __vec) const; 
# 766
inline const char *scan_is(mask __m, const char * __lo, const char * __hi) const; 
# 780
inline const char *scan_not(mask __m, const char * __lo, const char * __hi) const; 
# 795
char_type toupper(char_type __c) const 
# 796
{ return this->do_toupper(__c); } 
# 812
const char_type *toupper(char_type *__lo, const char_type *__hi) const 
# 813
{ return this->do_toupper(__lo, __hi); } 
# 828
char_type tolower(char_type __c) const 
# 829
{ return this->do_tolower(__c); } 
# 845
const char_type *tolower(char_type *__lo, const char_type *__hi) const 
# 846
{ return this->do_tolower(__lo, __hi); } 
# 865
char_type widen(char __c) const 
# 866
{ 
# 867
if (_M_widen_ok) { 
# 868
return (_M_widen)[static_cast< unsigned char>(__c)]; }  
# 869
this->_M_widen_init(); 
# 870
return this->do_widen(__c); 
# 871
} 
# 892
const char *widen(const char *__lo, const char *__hi, char_type *__to) const 
# 893
{ 
# 894
if ((_M_widen_ok) == 1) 
# 895
{ 
# 896
__builtin_memcpy(__to, __lo, __hi - __lo); 
# 897
return __hi; 
# 898
}  
# 899
if (!(_M_widen_ok)) { 
# 900
this->_M_widen_init(); }  
# 901
return this->do_widen(__lo, __hi, __to); 
# 902
} 
# 923
char narrow(char_type __c, char __dfault) const 
# 924
{ 
# 925
if ((_M_narrow)[static_cast< unsigned char>(__c)]) { 
# 926
return (_M_narrow)[static_cast< unsigned char>(__c)]; }  
# 927
const char __t = this->do_narrow(__c, __dfault); 
# 928
if (__t != __dfault) { 
# 929
((_M_narrow)[static_cast< unsigned char>(__c)]) = __t; }  
# 930
return __t; 
# 931
} 
# 956
const char_type *narrow(const char_type *__lo, const char_type *__hi, char 
# 957
__dfault, char *__to) const 
# 958
{ 
# 959
if (__builtin_expect((_M_narrow_ok) == 1, true)) 
# 960
{ 
# 961
__builtin_memcpy(__to, __lo, __hi - __lo); 
# 962
return __hi; 
# 963
}  
# 964
if (!(_M_narrow_ok)) { 
# 965
this->_M_narrow_init(); }  
# 966
return this->do_narrow(__lo, __hi, __dfault, __to); 
# 967
} 
# 974
const mask *table() const throw() 
# 975
{ return _M_table; } 
# 979
static const mask *classic_table() throw(); 
# 989
protected: virtual ~ctype(); 
# 1005
virtual char_type do_toupper(char_type __c) const; 
# 1022
virtual const char_type *do_toupper(char_type * __lo, const char_type * __hi) const; 
# 1038
virtual char_type do_tolower(char_type __c) const; 
# 1055
virtual const char_type *do_tolower(char_type * __lo, const char_type * __hi) const; 
# 1075
virtual char_type do_widen(char __c) const 
# 1076
{ return __c; } 
# 1098
virtual const char *do_widen(const char *__lo, const char *__hi, char_type *__to) const 
# 1099
{ 
# 1100
__builtin_memcpy(__to, __lo, __hi - __lo); 
# 1101
return __hi; 
# 1102
} 
# 1124
virtual char do_narrow(char_type __c, char __dfault) const 
# 1125
{ return __c; } 
# 1150
virtual const char_type *do_narrow(const char_type *__lo, const char_type *__hi, char 
# 1151
__dfault, char *__to) const 
# 1152
{ 
# 1153
__builtin_memcpy(__to, __lo, __hi - __lo); 
# 1154
return __hi; 
# 1155
} 
# 1158
private: void _M_narrow_init() const; 
# 1159
void _M_widen_init() const; 
# 1160
}; 
# 1175
template<> class ctype< wchar_t>  : public __ctype_abstract_base< wchar_t>  { 
# 1180
public: typedef wchar_t char_type; 
# 1181
typedef wctype_t __wmask_type; 
# 1184
protected: __c_locale _M_c_locale_ctype; 
# 1187
bool _M_narrow_ok; 
# 1188
char _M_narrow[128]; 
# 1189
wint_t _M_widen[1 + (static_cast< unsigned char>(-1))]; 
# 1192
mask _M_bit[16]; 
# 1193
__wmask_type _M_wmask[16]; 
# 1198
public: static locale::id id; 
# 1208
explicit ctype(size_t __refs = 0); 
# 1219
explicit ctype(__c_locale __cloc, size_t __refs = 0); 
# 1223
protected: __wmask_type _M_convert_to_wmask(const mask __m) const throw(); 
# 1227
virtual ~ctype(); 
# 1243
virtual bool do_is(mask __m, char_type __c) const; 
# 1262
virtual const char_type *do_is(const char_type * __lo, const char_type * __hi, mask * __vec) const; 
# 1280
virtual const char_type *do_scan_is(mask __m, const char_type * __lo, const char_type * __hi) const; 
# 1298
virtual const char_type *do_scan_not(mask __m, const char_type * __lo, const char_type * __hi) const; 
# 1315
virtual char_type do_toupper(char_type __c) const; 
# 1332
virtual const char_type *do_toupper(char_type * __lo, const char_type * __hi) const; 
# 1348
virtual char_type do_tolower(char_type __c) const; 
# 1365
virtual const char_type *do_tolower(char_type * __lo, const char_type * __hi) const; 
# 1385
virtual char_type do_widen(char __c) const; 
# 1407
virtual const char *do_widen(const char * __lo, const char * __hi, char_type * __to) const; 
# 1430
virtual char do_narrow(char_type __c, char __dfault) const; 
# 1456
virtual const char_type *do_narrow(const char_type * __lo, const char_type * __hi, char __dfault, char * __to) const; 
# 1461
void _M_initialize_ctype() throw(); 
# 1462
}; 
# 1466
template< class _CharT> 
# 1467
class ctype_byname : public ctype< _CharT>  { 
# 1470
public: typedef typename ::std::ctype< _CharT> ::mask mask; 
# 1473
explicit ctype_byname(const char * __s, ::std::size_t __refs = 0); 
# 1477
protected: virtual ~ctype_byname() { } 
# 1478
}; 
# 1482
template<> class ctype_byname< char>  : public ctype< char>  { 
# 1486
public: explicit ctype_byname(const char * __s, size_t __refs = 0); 
# 1490
protected: virtual ~ctype_byname(); 
# 1491
}; 
# 1495
template<> class ctype_byname< wchar_t>  : public ctype< wchar_t>  { 
# 1499
public: explicit ctype_byname(const char * __s, size_t __refs = 0); 
# 1503
protected: virtual ~ctype_byname(); 
# 1504
}; 
# 1508
}
# 37 "/usr/include/x86_64-linux-gnu/c++/4.8/bits/ctype_inline.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 43
inline bool ctype< char> ::is(mask __m, char __c) const 
# 44
{ return ((_M_table)[static_cast< unsigned char>(__c)]) & __m; } 
# 48
inline const char *ctype< char> ::is(const char *__low, const char *__high, mask *__vec) const 
# 49
{ 
# 50
while (__low < __high) { 
# 51
(*(__vec++)) = ((_M_table)[static_cast< unsigned char>(*(__low++))]); }  
# 52
return __high; 
# 53
} 
# 57
inline const char *ctype< char> ::scan_is(mask __m, const char *__low, const char *__high) const 
# 58
{ 
# 59
while ((__low < __high) && (!(((_M_table)[static_cast< unsigned char>(*__low)]) & __m))) { 
# 61
++__low; }  
# 62
return __low; 
# 63
} 
# 67
inline const char *ctype< char> ::scan_not(mask __m, const char *__low, const char *__high) const 
# 68
{ 
# 69
while ((__low < __high) && ((((_M_table)[static_cast< unsigned char>(*__low)]) & __m) != 0)) { 
# 71
++__low; }  
# 72
return __low; 
# 73
} 
# 76
}
# 1513 "/usr/include/c++/4.8/bits/locale_facets.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 1518
class __num_base { 
# 1524
public: enum { 
# 1525
_S_ominus, 
# 1526
_S_oplus, 
# 1527
_S_ox, 
# 1528
_S_oX, 
# 1529
_S_odigits, 
# 1530
_S_odigits_end = 20, 
# 1531
_S_oudigits = 20, 
# 1532
_S_oudigits_end = 36, 
# 1533
_S_oe = 18, 
# 1534
_S_oE = 34, 
# 1535
_S_oend = 36
# 1536
}; 
# 1543
static const char *_S_atoms_out; 
# 1547
static const char *_S_atoms_in; 
# 1550
enum { 
# 1551
_S_iminus, 
# 1552
_S_iplus, 
# 1553
_S_ix, 
# 1554
_S_iX, 
# 1555
_S_izero, 
# 1556
_S_ie = 18, 
# 1557
_S_iE = 24, 
# 1558
_S_iend = 26
# 1559
}; 
# 1564
static void _S_format_float(const ios_base & __io, char * __fptr, char __mod) throw(); 
# 1565
}; 
# 1567
template< class _CharT> 
# 1568
struct __numpunct_cache : public locale::facet { 
# 1570
const char *_M_grouping; 
# 1571
size_t _M_grouping_size; 
# 1572
bool _M_use_grouping; 
# 1573
const _CharT *_M_truename; 
# 1574
size_t _M_truename_size; 
# 1575
const _CharT *_M_falsename; 
# 1576
size_t _M_falsename_size; 
# 1577
_CharT _M_decimal_point; 
# 1578
_CharT _M_thousands_sep; 
# 1584
_CharT _M_atoms_out[__num_base::_S_oend]; 
# 1590
_CharT _M_atoms_in[__num_base::_S_iend]; 
# 1592
bool _M_allocated; 
# 1594
__numpunct_cache(size_t __refs = 0) : locale::facet(__refs), _M_grouping((0)), _M_grouping_size((0)), _M_use_grouping(false), _M_truename((0)), _M_truename_size((0)), _M_falsename((0)), _M_falsename_size((0)), _M_decimal_point(_CharT()), _M_thousands_sep(_CharT()), _M_allocated(false) 
# 1600
{ } 
# 1602
virtual ~__numpunct_cache(); 
# 1605
void _M_cache(const locale & __loc); 
# 1609
private: __numpunct_cache &operator=(const __numpunct_cache &); 
# 1612
explicit __numpunct_cache(const __numpunct_cache &); 
# 1613
}; 
# 1615
template< class _CharT> 
# 1616
__numpunct_cache< _CharT> ::~__numpunct_cache() 
# 1617
{ 
# 1618
if (_M_allocated) 
# 1619
{ 
# 1620
delete [] (_M_grouping); 
# 1621
delete [] (_M_truename); 
# 1622
delete [] (_M_falsename); 
# 1623
}  
# 1624
} 
# 1640
template< class _CharT> 
# 1641
class numpunct : public locale::facet { 
# 1647
public: typedef _CharT char_type; 
# 1648
typedef basic_string< _CharT, char_traits< _CharT> , allocator< _CharT> >  string_type; 
# 1650
typedef __numpunct_cache< _CharT>  __cache_type; 
# 1653
protected: __cache_type *_M_data; 
# 1657
public: static locale::id id; 
# 1665
explicit numpunct(size_t __refs = 0) : locale::facet(__refs), _M_data((0)) 
# 1667
{ _M_initialize_numpunct(); } 
# 1679
explicit numpunct(__cache_type *__cache, size_t __refs = 0) : locale::facet(__refs), _M_data(__cache) 
# 1681
{ _M_initialize_numpunct(); } 
# 1693
explicit numpunct(__c_locale __cloc, size_t __refs = 0) : locale::facet(__refs), _M_data((0)) 
# 1695
{ _M_initialize_numpunct(__cloc); } 
# 1707
char_type decimal_point() const 
# 1708
{ return this->do_decimal_point(); } 
# 1720
char_type thousands_sep() const 
# 1721
{ return this->do_thousands_sep(); } 
# 1751
string grouping() const 
# 1752
{ return this->do_grouping(); } 
# 1764
string_type truename() const 
# 1765
{ return this->do_truename(); } 
# 1777
string_type falsename() const 
# 1778
{ return this->do_falsename(); } 
# 1783
protected: virtual ~numpunct(); 
# 1794
virtual char_type do_decimal_point() const 
# 1795
{ return (_M_data)->_M_decimal_point; } 
# 1806
virtual char_type do_thousands_sep() const 
# 1807
{ return (_M_data)->_M_thousands_sep; } 
# 1819
virtual string do_grouping() const 
# 1820
{ return ((_M_data)->_M_grouping); } 
# 1832
virtual string_type do_truename() const 
# 1833
{ return ((_M_data)->_M_truename); } 
# 1845
virtual string_type do_falsename() const 
# 1846
{ return ((_M_data)->_M_falsename); } 
# 1850
void _M_initialize_numpunct(__c_locale __cloc = 0); 
# 1851
}; 
# 1853
template< class _CharT> locale::id 
# 1854
numpunct< _CharT> ::id; 
# 1857
template<> numpunct< char> ::~numpunct(); 
# 1861
template<> void numpunct< char> ::_M_initialize_numpunct(__c_locale __cloc); 
# 1865
template<> numpunct< wchar_t> ::~numpunct(); 
# 1869
template<> void numpunct< wchar_t> ::_M_initialize_numpunct(__c_locale __cloc); 
# 1873
template< class _CharT> 
# 1874
class numpunct_byname : public numpunct< _CharT>  { 
# 1877
public: typedef _CharT char_type; 
# 1878
typedef basic_string< _CharT, char_traits< _CharT> , allocator< _CharT> >  string_type; 
# 1881
explicit numpunct_byname(const char *__s, ::std::size_t __refs = 0) : ::std::numpunct< _CharT> (__refs) 
# 1883
{ 
# 1884
if ((__builtin_strcmp(__s, "C") != 0) && (__builtin_strcmp(__s, "POSIX") != 0)) 
# 1886
{ 
# 1887
::std::__c_locale __tmp; 
# 1888
(this->_S_create_c_locale(__tmp, __s)); 
# 1889
(this->_M_initialize_numpunct(__tmp)); 
# 1890
(this->_S_destroy_c_locale(__tmp)); 
# 1891
}  
# 1892
} 
# 1896
protected: virtual ~numpunct_byname() { } 
# 1897
}; 
# 1914
template< class _CharT, class _InIter> 
# 1915
class num_get : public locale::facet { 
# 1921
public: typedef _CharT char_type; 
# 1922
typedef _InIter iter_type; 
# 1926
static locale::id id; 
# 1936
explicit num_get(size_t __refs = 0) : locale::facet(__refs) { } 
# 1962
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 1963
__err, bool &__v) const 
# 1964
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 1999
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2000
__err, long &__v) const 
# 2001
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2004
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2005
__err, unsigned short &__v) const 
# 2006
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2009
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2010
__err, unsigned &__v) const 
# 2011
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2014
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2015
__err, unsigned long &__v) const 
# 2016
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2020
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2021
__err, long long &__v) const 
# 2022
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2025
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2026
__err, unsigned long long &__v) const 
# 2027
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2059
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2060
__err, float &__v) const 
# 2061
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2064
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2065
__err, double &__v) const 
# 2066
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2069
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2070
__err, long double &__v) const 
# 2071
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2102
iter_type get(iter_type __in, iter_type __end, ios_base &__io, ios_base::iostate &
# 2103
__err, void *&__v) const 
# 2104
{ return (this->do_get(__in, __end, __io, __err, __v)); } 
# 2108
protected: virtual ~num_get() { } 
# 2111
iter_type _M_extract_float(iter_type, iter_type, ios_base &, ios_base::iostate &, string &) const; 
# 2114
template< class _ValueT> iter_type _M_extract_int(iter_type, iter_type, ios_base &, ios_base::iostate &, _ValueT &) const; 
# 2119
template< class _CharT2> typename __gnu_cxx::__enable_if< __is_char< _CharT2> ::__value, int> ::__type 
# 2121
_M_find(const _CharT2 *, size_t __len, _CharT2 __c) const 
# 2122
{ 
# 2123
int __ret = (-1); 
# 2124
if (__len <= (10)) 
# 2125
{ 
# 2126
if ((__c >= ((_CharT2)'0')) && (__c < ((_CharT2)(((_CharT2)'0') + __len)))) { 
# 2127
__ret = (__c - ((_CharT2)'0')); }  
# 2128
} else 
# 2130
{ 
# 2131
if ((__c >= ((_CharT2)'0')) && (__c <= ((_CharT2)'9'))) { 
# 2132
__ret = (__c - ((_CharT2)'0')); } else { 
# 2133
if ((__c >= ((_CharT2)'a')) && (__c <= ((_CharT2)'f'))) { 
# 2134
__ret = (10 + (__c - ((_CharT2)'a'))); } else { 
# 2135
if ((__c >= ((_CharT2)'A')) && (__c <= ((_CharT2)'F'))) { 
# 2136
__ret = (10 + (__c - ((_CharT2)'A'))); }  }  }  
# 2137
}  
# 2138
return __ret; 
# 2139
} 
# 2141
template< class _CharT2> typename __gnu_cxx::__enable_if< !__is_char< _CharT2> ::__value, int> ::__type 
# 2144
_M_find(const _CharT2 *__zero, size_t __len, _CharT2 __c) const 
# 2145
{ 
# 2146
int __ret = (-1); 
# 2147
const char_type *__q = char_traits< _CharT2> ::find(__zero, __len, __c); 
# 2148
if (__q) 
# 2149
{ 
# 2150
__ret = (__q - __zero); 
# 2151
if (__ret > 15) { 
# 2152
__ret -= 6; }  
# 2153
}  
# 2154
return __ret; 
# 2155
} 
# 2173
virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &, bool &) const; 
# 2176
virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 2177
__err, long &__v) const 
# 2178
{ return _M_extract_int(__beg, __end, __io, __err, __v); } 
# 2181
virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 2182
__err, unsigned short &__v) const 
# 2183
{ return _M_extract_int(__beg, __end, __io, __err, __v); } 
# 2186
virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 2187
__err, unsigned &__v) const 
# 2188
{ return _M_extract_int(__beg, __end, __io, __err, __v); } 
# 2191
virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 2192
__err, unsigned long &__v) const 
# 2193
{ return _M_extract_int(__beg, __end, __io, __err, __v); } 
# 2197
virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 2198
__err, long long &__v) const 
# 2199
{ return _M_extract_int(__beg, __end, __io, __err, __v); } 
# 2202
virtual iter_type do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 2203
__err, unsigned long long &__v) const 
# 2204
{ return _M_extract_int(__beg, __end, __io, __err, __v); } 
# 2208
virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &, float &) const; 
# 2211
virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &, double &) const; 
# 2221
virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &, long double &) const; 
# 2226
virtual iter_type do_get(iter_type, iter_type, ios_base &, ios_base::iostate &, void *&) const; 
# 2235
}; 
# 2237
template< class _CharT, class _InIter> locale::id 
# 2238
num_get< _CharT, _InIter> ::id; 
# 2253
template< class _CharT, class _OutIter> 
# 2254
class num_put : public locale::facet { 
# 2260
public: typedef _CharT char_type; 
# 2261
typedef _OutIter iter_type; 
# 2265
static locale::id id; 
# 2275
explicit num_put(size_t __refs = 0) : locale::facet(__refs) { } 
# 2293
iter_type put(iter_type __s, ios_base &__io, char_type __fill, bool __v) const 
# 2294
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2335
iter_type put(iter_type __s, ios_base &__io, char_type __fill, long __v) const 
# 2336
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2339
iter_type put(iter_type __s, ios_base &__io, char_type __fill, unsigned long 
# 2340
__v) const 
# 2341
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2345
iter_type put(iter_type __s, ios_base &__io, char_type __fill, long long __v) const 
# 2346
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2349
iter_type put(iter_type __s, ios_base &__io, char_type __fill, unsigned long long 
# 2350
__v) const 
# 2351
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2398
iter_type put(iter_type __s, ios_base &__io, char_type __fill, double __v) const 
# 2399
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2402
iter_type put(iter_type __s, ios_base &__io, char_type __fill, long double 
# 2403
__v) const 
# 2404
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2423
iter_type put(iter_type __s, ios_base &__io, char_type __fill, const void *
# 2424
__v) const 
# 2425
{ return (this->do_put(__s, __io, __fill, __v)); } 
# 2428
protected: template< class _ValueT> iter_type _M_insert_float(iter_type, ios_base & __io, char_type __fill, char __mod, _ValueT __v) const; 
# 2434
void _M_group_float(const char * __grouping, size_t __grouping_size, char_type __sep, const char_type * __p, char_type * __new, char_type * __cs, int & __len) const; 
# 2438
template< class _ValueT> iter_type _M_insert_int(iter_type, ios_base & __io, char_type __fill, _ValueT __v) const; 
# 2444
void _M_group_int(const char * __grouping, size_t __grouping_size, char_type __sep, ios_base & __io, char_type * __new, char_type * __cs, int & __len) const; 
# 2449
void _M_pad(char_type __fill, streamsize __w, ios_base & __io, char_type * __new, const char_type * __cs, int & __len) const; 
# 2454
virtual ~num_put() { } 
# 2471
virtual iter_type do_put(iter_type __s, ios_base & __io, char_type __fill, bool __v) const; 
# 2474
virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, long __v) const 
# 2475
{ return _M_insert_int(__s, __io, __fill, __v); } 
# 2478
virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, unsigned long 
# 2479
__v) const 
# 2480
{ return _M_insert_int(__s, __io, __fill, __v); } 
# 2484
virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, long long 
# 2485
__v) const 
# 2486
{ return _M_insert_int(__s, __io, __fill, __v); } 
# 2489
virtual iter_type do_put(iter_type __s, ios_base &__io, char_type __fill, unsigned long long 
# 2490
__v) const 
# 2491
{ return _M_insert_int(__s, __io, __fill, __v); } 
# 2495
virtual iter_type do_put(iter_type, ios_base &, char_type, double) const; 
# 2503
virtual iter_type do_put(iter_type, ios_base &, char_type, long double) const; 
# 2507
virtual iter_type do_put(iter_type, ios_base &, char_type, const void *) const; 
# 2515
}; 
# 2517
template< class _CharT, class _OutIter> locale::id 
# 2518
num_put< _CharT, _OutIter> ::id; 
# 2528
template< class _CharT> inline bool 
# 2530
isspace(_CharT __c, const locale &__loc) 
# 2531
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::space, __c)); } 
# 2534
template< class _CharT> inline bool 
# 2536
isprint(_CharT __c, const locale &__loc) 
# 2537
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::print, __c)); } 
# 2540
template< class _CharT> inline bool 
# 2542
iscntrl(_CharT __c, const locale &__loc) 
# 2543
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::cntrl, __c)); } 
# 2546
template< class _CharT> inline bool 
# 2548
isupper(_CharT __c, const locale &__loc) 
# 2549
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::upper, __c)); } 
# 2552
template< class _CharT> inline bool 
# 2554
islower(_CharT __c, const locale &__loc) 
# 2555
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::lower, __c)); } 
# 2558
template< class _CharT> inline bool 
# 2560
isalpha(_CharT __c, const locale &__loc) 
# 2561
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::alpha, __c)); } 
# 2564
template< class _CharT> inline bool 
# 2566
isdigit(_CharT __c, const locale &__loc) 
# 2567
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::digit, __c)); } 
# 2570
template< class _CharT> inline bool 
# 2572
ispunct(_CharT __c, const locale &__loc) 
# 2573
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::punct, __c)); } 
# 2576
template< class _CharT> inline bool 
# 2578
isxdigit(_CharT __c, const locale &__loc) 
# 2579
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::xdigit, __c)); } 
# 2582
template< class _CharT> inline bool 
# 2584
isalnum(_CharT __c, const locale &__loc) 
# 2585
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::alnum, __c)); } 
# 2588
template< class _CharT> inline bool 
# 2590
isgraph(_CharT __c, const locale &__loc) 
# 2591
{ return (use_facet< ctype< _CharT> > (__loc).is(ctype_base::graph, __c)); } 
# 2594
template< class _CharT> inline _CharT 
# 2596
toupper(_CharT __c, const locale &__loc) 
# 2597
{ return (use_facet< ctype< _CharT> > (__loc).toupper(__c)); } 
# 2600
template< class _CharT> inline _CharT 
# 2602
tolower(_CharT __c, const locale &__loc) 
# 2603
{ return (use_facet< ctype< _CharT> > (__loc).tolower(__c)); } 
# 2606
}
# 35 "/usr/include/c++/4.8/bits/locale_facets.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 41
template< class _Facet> 
# 42
struct __use_cache { 
# 45
const _Facet *operator()(const locale & __loc) const; 
# 46
}; 
# 49
template< class _CharT> 
# 50
struct __use_cache< __numpunct_cache< _CharT> >  { 
# 53
const __numpunct_cache< _CharT>  *operator()(const locale &__loc) const 
# 54
{ 
# 55
const size_t __i = (numpunct< _CharT> ::id._M_id)(); 
# 56
const locale::facet **__caches = (__loc._M_impl)->_M_caches; 
# 57
if (!(__caches[__i])) 
# 58
{ 
# 59
__numpunct_cache< _CharT>  *__tmp = (0); 
# 60
try 
# 61
{ 
# 62
__tmp = (new __numpunct_cache< _CharT> ); 
# 63
(__tmp->_M_cache(__loc)); 
# 64
} 
# 65
catch (...) 
# 66
{ 
# 67
delete __tmp; 
# 68
throw; 
# 69
}  
# 70
(__loc._M_impl)->_M_install_cache(__tmp, __i); 
# 71
}  
# 72
return static_cast< const __numpunct_cache< _CharT>  *>(__caches[__i]); 
# 73
} 
# 74
}; 
# 76
template< class _CharT> void 
# 78
__numpunct_cache< _CharT> ::_M_cache(const locale &__loc) 
# 79
{ 
# 80
(_M_allocated) = true; 
# 82
const numpunct< _CharT>  &__np = use_facet< numpunct< _CharT> > (__loc); 
# 84
char *__grouping = (0); 
# 85
_CharT *__truename = (0); 
# 86
_CharT *__falsename = (0); 
# 87
try 
# 88
{ 
# 89
(_M_grouping_size) = ((__np.grouping()).size()); 
# 90
__grouping = (new char [_M_grouping_size]); 
# 91
((__np.grouping()).copy(__grouping, _M_grouping_size)); 
# 92
(_M_grouping) = __grouping; 
# 93
(_M_use_grouping) = ((_M_grouping_size) && ((static_cast< signed char>((_M_grouping)[0])) > 0) && (((_M_grouping)[0]) != __gnu_cxx::__numeric_traits_integer< char> ::__max)); 
# 98
(_M_truename_size) = ((__np.truename()).size()); 
# 99
__truename = (new _CharT [_M_truename_size]); 
# 100
((__np.truename()).copy(__truename, _M_truename_size)); 
# 101
(_M_truename) = __truename; 
# 103
(_M_falsename_size) = ((__np.falsename()).size()); 
# 104
__falsename = (new _CharT [_M_falsename_size]); 
# 105
((__np.falsename()).copy(__falsename, _M_falsename_size)); 
# 106
(_M_falsename) = __falsename; 
# 108
(_M_decimal_point) = (__np.decimal_point()); 
# 109
(_M_thousands_sep) = (__np.thousands_sep()); 
# 111
const ctype< _CharT>  &__ct = use_facet< ctype< _CharT> > (__loc); 
# 112
(__ct.widen(__num_base::_S_atoms_out, __num_base::_S_atoms_out + __num_base::_S_oend, _M_atoms_out)); 
# 115
(__ct.widen(__num_base::_S_atoms_in, __num_base::_S_atoms_in + __num_base::_S_iend, _M_atoms_in)); 
# 118
} 
# 119
catch (...) 
# 120
{ 
# 121
delete [] __grouping; 
# 122
delete [] __truename; 
# 123
delete [] __falsename; 
# 124
throw; 
# 125
}  
# 126
} 
# 136
__attribute((__pure__)) bool 
# 137
__verify_grouping(const char * __grouping, size_t __grouping_size, const string & __grouping_tmp) throw(); 
# 142
template< class _CharT, class _InIter> _InIter 
# 145
num_get< _CharT, _InIter> ::_M_extract_float(_InIter __beg, _InIter __end, ios_base &__io, ios_base::iostate &
# 146
__err, string &__xtrc) const 
# 147
{ 
# 148
typedef char_traits< _CharT>  __traits_type; 
# 149
typedef __numpunct_cache< _CharT>  __cache_type; 
# 150
__use_cache< __numpunct_cache< _CharT> >  __uc; 
# 151
const locale &__loc = __io._M_getloc(); 
# 152
const __cache_type *__lc = __uc(__loc); 
# 153
const _CharT *__lit = ((__lc->_M_atoms_in)); 
# 154
char_type __c = (char_type()); 
# 157
bool __testeof = __beg == __end; 
# 160
if (!__testeof) 
# 161
{ 
# 162
__c = (*__beg); 
# 163
const bool __plus = __c == (__lit[__num_base::_S_iplus]); 
# 164
if ((__plus || (__c == (__lit[__num_base::_S_iminus]))) && (!((__lc->_M_use_grouping) && (__c == (__lc->_M_thousands_sep)))) && (!(__c == (__lc->_M_decimal_point)))) 
# 167
{ 
# 168
(__xtrc += (__plus ? '+' : '-')); 
# 169
if ((++__beg) != __end) { 
# 170
__c = (*__beg); } else { 
# 172
__testeof = true; }  
# 173
}  
# 174
}  
# 177
bool __found_mantissa = false; 
# 178
int __sep_pos = 0; 
# 179
while (!__testeof) 
# 180
{ 
# 181
if (((__lc->_M_use_grouping) && (__c == (__lc->_M_thousands_sep))) || (__c == (__lc->_M_decimal_point))) { 
# 183
break; } else { 
# 184
if (__c == (__lit[__num_base::_S_izero])) 
# 185
{ 
# 186
if (!__found_mantissa) 
# 187
{ 
# 188
(__xtrc += ('0')); 
# 189
__found_mantissa = true; 
# 190
}  
# 191
++__sep_pos; 
# 193
if ((++__beg) != __end) { 
# 194
__c = (*__beg); } else { 
# 196
__testeof = true; }  
# 197
} else { 
# 199
break; }  }  
# 200
}  
# 203
bool __found_dec = false; 
# 204
bool __found_sci = false; 
# 205
string __found_grouping; 
# 206
if (__lc->_M_use_grouping) { 
# 207
__found_grouping.reserve(32); }  
# 208
const char_type *__lit_zero = __lit + __num_base::_S_izero; 
# 210
if (!(__lc->_M_allocated)) { 
# 212
while (!__testeof) { 
# 213
{ 
# 214
const int __digit = _M_find(__lit_zero, 10, __c); 
# 215
if (__digit != (-1)) 
# 216
{ 
# 217
(__xtrc += (('0') + __digit)); 
# 218
__found_mantissa = true; 
# 219
} else { 
# 220
if ((__c == (__lc->_M_decimal_point)) && (!__found_dec) && (!__found_sci)) 
# 222
{ 
# 223
(__xtrc += ('.')); 
# 224
__found_dec = true; 
# 225
} else { 
# 226
if (((__c == (__lit[__num_base::_S_ie])) || (__c == (__lit[__num_base::_S_iE]))) && (!__found_sci) && __found_mantissa) 
# 229
{ 
# 231
(__xtrc += ('e')); 
# 232
__found_sci = true; 
# 235
if ((++__beg) != __end) 
# 236
{ 
# 237
__c = (*__beg); 
# 238
const bool __plus = __c == (__lit[__num_base::_S_iplus]); 
# 239
if (__plus || (__c == (__lit[__num_base::_S_iminus]))) { 
# 240
(__xtrc += (__plus ? '+' : '-')); } else { 
# 242
continue; }  
# 243
} else 
# 245
{ 
# 246
__testeof = true; 
# 247
break; 
# 248
}  
# 249
} else { 
# 251
break; }  }  }  
# 253
if ((++__beg) != __end) { 
# 254
__c = (*__beg); } else { 
# 256
__testeof = true; }  
# 257
} }  } else { 
# 259
while (!__testeof) { 
# 260
{ 
# 263
if ((__lc->_M_use_grouping) && (__c == (__lc->_M_thousands_sep))) 
# 264
{ 
# 265
if ((!__found_dec) && (!__found_sci)) 
# 266
{ 
# 269
if (__sep_pos) 
# 270
{ 
# 271
(__found_grouping += (static_cast< char>(__sep_pos))); 
# 272
__sep_pos = 0; 
# 273
} else 
# 275
{ 
# 278
__xtrc.clear(); 
# 279
break; 
# 280
}  
# 281
} else { 
# 283
break; }  
# 284
} else { 
# 285
if (__c == (__lc->_M_decimal_point)) 
# 286
{ 
# 287
if ((!__found_dec) && (!__found_sci)) 
# 288
{ 
# 292
if (__found_grouping.size()) { 
# 293
(__found_grouping += (static_cast< char>(__sep_pos))); }  
# 294
(__xtrc += ('.')); 
# 295
__found_dec = true; 
# 296
} else { 
# 298
break; }  
# 299
} else 
# 301
{ 
# 302
const char_type *__q = __traits_type::find(__lit_zero, 10, __c); 
# 304
if (__q) 
# 305
{ 
# 306
__xtrc += ('0' + (__q - __lit_zero)); 
# 307
__found_mantissa = true; 
# 308
++__sep_pos; 
# 309
} else { 
# 310
if (((__c == (__lit[__num_base::_S_ie])) || (__c == (__lit[__num_base::_S_iE]))) && (!__found_sci) && __found_mantissa) 
# 313
{ 
# 315
if ((__found_grouping.size()) && (!__found_dec)) { 
# 316
(__found_grouping += (static_cast< char>(__sep_pos))); }  
# 317
(__xtrc += ('e')); 
# 318
__found_sci = true; 
# 321
if ((++__beg) != __end) 
# 322
{ 
# 323
__c = (*__beg); 
# 324
const bool __plus = __c == (__lit[__num_base::_S_iplus]); 
# 325
if ((__plus || (__c == (__lit[__num_base::_S_iminus]))) && (!((__lc->_M_use_grouping) && (__c == (__lc->_M_thousands_sep)))) && (!(__c == (__lc->_M_decimal_point)))) { 
# 329
(__xtrc += (__plus ? '+' : '-')); } else { 
# 331
continue; }  
# 332
} else 
# 334
{ 
# 335
__testeof = true; 
# 336
break; 
# 337
}  
# 338
} else { 
# 340
break; }  }  
# 341
}  }  
# 343
if ((++__beg) != __end) { 
# 344
__c = (*__beg); } else { 
# 346
__testeof = true; }  
# 347
} }  }  
# 351
if (__found_grouping.size()) 
# 352
{ 
# 354
if ((!__found_dec) && (!__found_sci)) { 
# 355
(__found_grouping += (static_cast< char>(__sep_pos))); }  
# 357
if (!std::__verify_grouping((__lc->_M_grouping), (__lc->_M_grouping_size), __found_grouping)) { 
# 360
__err = ios_base::failbit; }  
# 361
}  
# 363
return __beg; 
# 364
} 
# 366
template< class _CharT, class _InIter> 
# 367
template< class _ValueT> _InIter 
# 370
num_get< _CharT, _InIter> ::_M_extract_int(_InIter __beg, _InIter __end, ios_base &__io, ios_base::iostate &
# 371
__err, _ValueT &__v) const 
# 372
{ 
# 373
typedef char_traits< _CharT>  __traits_type; 
# 374
using __gnu_cxx::__add_unsigned;
# 375
typedef typename __gnu_cxx::__add_unsigned< _ValueT> ::__type __unsigned_type; 
# 376
typedef __numpunct_cache< _CharT>  __cache_type; 
# 377
__use_cache< __numpunct_cache< _CharT> >  __uc; 
# 378
const locale &__loc = __io._M_getloc(); 
# 379
const __cache_type *__lc = __uc(__loc); 
# 380
const _CharT *__lit = ((__lc->_M_atoms_in)); 
# 381
char_type __c = (char_type()); 
# 384
const ios_base::fmtflags __basefield = ((__io.flags()) & ios_base::basefield); 
# 386
const bool __oct = __basefield == ios_base::oct; 
# 387
int __base = __oct ? 8 : ((__basefield == ios_base::hex) ? 16 : 10); 
# 390
bool __testeof = __beg == __end; 
# 393
bool __negative = false; 
# 394
if (!__testeof) 
# 395
{ 
# 396
__c = (*__beg); 
# 397
__negative = (__c == (__lit[__num_base::_S_iminus])); 
# 398
if ((__negative || (__c == (__lit[__num_base::_S_iplus]))) && (!((__lc->_M_use_grouping) && (__c == (__lc->_M_thousands_sep)))) && (!(__c == (__lc->_M_decimal_point)))) 
# 401
{ 
# 402
if ((++__beg) != __end) { 
# 403
__c = (*__beg); } else { 
# 405
__testeof = true; }  
# 406
}  
# 407
}  
# 411
bool __found_zero = false; 
# 412
int __sep_pos = 0; 
# 413
while (!__testeof) 
# 414
{ 
# 415
if (((__lc->_M_use_grouping) && (__c == (__lc->_M_thousands_sep))) || (__c == (__lc->_M_decimal_point))) { 
# 417
break; } else { 
# 418
if ((__c == (__lit[__num_base::_S_izero])) && ((!__found_zero) || (__base == 10))) 
# 420
{ 
# 421
__found_zero = true; 
# 422
++__sep_pos; 
# 423
if (__basefield == 0) { 
# 424
__base = 8; }  
# 425
if (__base == 8) { 
# 426
__sep_pos = 0; }  
# 427
} else { 
# 428
if (__found_zero && ((__c == (__lit[__num_base::_S_ix])) || (__c == (__lit[__num_base::_S_iX])))) 
# 431
{ 
# 432
if (__basefield == 0) { 
# 433
__base = 16; }  
# 434
if (__base == 16) 
# 435
{ 
# 436
__found_zero = false; 
# 437
__sep_pos = 0; 
# 438
} else { 
# 440
break; }  
# 441
} else { 
# 443
break; }  }  }  
# 445
if ((++__beg) != __end) 
# 446
{ 
# 447
__c = (*__beg); 
# 448
if (!__found_zero) { 
# 449
break; }  
# 450
} else { 
# 452
__testeof = true; }  
# 453
}  
# 457
const size_t __len = (__base == 16) ? (__num_base::_S_iend) - (__num_base::_S_izero) : __base; 
# 461
string __found_grouping; 
# 462
if (__lc->_M_use_grouping) { 
# 463
__found_grouping.reserve(32); }  
# 464
bool __testfail = false; 
# 465
bool __testoverflow = false; 
# 466
const __unsigned_type __max = (__negative && __gnu_cxx::__numeric_traits< _ValueT> ::__is_signed) ? -__gnu_cxx::__numeric_traits< _ValueT> ::__min : __gnu_cxx::__numeric_traits< _ValueT> ::__max; 
# 470
const __unsigned_type __smax = __max / __base; 
# 471
__unsigned_type __result = (0); 
# 472
int __digit = 0; 
# 473
const char_type *__lit_zero = __lit + __num_base::_S_izero; 
# 475
if (!(__lc->_M_allocated)) { 
# 477
while (!__testeof) 
# 478
{ 
# 479
__digit = _M_find(__lit_zero, __len, __c); 
# 480
if (__digit == (-1)) { 
# 481
break; }  
# 483
if (__result > __smax) { 
# 484
__testoverflow = true; } else 
# 486
{ 
# 487
__result *= __base; 
# 488
__testoverflow |= (__result > (__max - __digit)); 
# 489
__result += __digit; 
# 490
++__sep_pos; 
# 491
}  
# 493
if ((++__beg) != __end) { 
# 494
__c = (*__beg); } else { 
# 496
__testeof = true; }  
# 497
}  } else { 
# 499
while (!__testeof) 
# 500
{ 
# 503
if ((__lc->_M_use_grouping) && (__c == (__lc->_M_thousands_sep))) 
# 504
{ 
# 507
if (__sep_pos) 
# 508
{ 
# 509
(__found_grouping += (static_cast< char>(__sep_pos))); 
# 510
__sep_pos = 0; 
# 511
} else 
# 513
{ 
# 514
__testfail = true; 
# 515
break; 
# 516
}  
# 517
} else { 
# 518
if (__c == (__lc->_M_decimal_point)) { 
# 519
break; } else 
# 521
{ 
# 522
const char_type *__q = __traits_type::find(__lit_zero, __len, __c); 
# 524
if (!__q) { 
# 525
break; }  
# 527
__digit = (__q - __lit_zero); 
# 528
if (__digit > 15) { 
# 529
__digit -= 6; }  
# 530
if (__result > __smax) { 
# 531
__testoverflow = true; } else 
# 533
{ 
# 534
__result *= __base; 
# 535
__testoverflow |= (__result > (__max - __digit)); 
# 536
__result += __digit; 
# 537
++__sep_pos; 
# 538
}  
# 539
}  }  
# 541
if ((++__beg) != __end) { 
# 542
__c = (*__beg); } else { 
# 544
__testeof = true; }  
# 545
}  }  
# 549
if (__found_grouping.size()) 
# 550
{ 
# 552
(__found_grouping += (static_cast< char>(__sep_pos))); 
# 554
if (!std::__verify_grouping((__lc->_M_grouping), (__lc->_M_grouping_size), __found_grouping)) { 
# 557
__err = ios_base::failbit; }  
# 558
}  
# 562
if (((!__sep_pos) && (!__found_zero) && (!(__found_grouping.size()))) || __testfail) 
# 564
{ 
# 565
__v = 0; 
# 566
__err = ios_base::failbit; 
# 567
} else { 
# 568
if (__testoverflow) 
# 569
{ 
# 570
if (__negative && __gnu_cxx::__numeric_traits< _ValueT> ::__is_signed) { 
# 572
__v = __gnu_cxx::__numeric_traits< _ValueT> ::__min; } else { 
# 574
__v = __gnu_cxx::__numeric_traits< _ValueT> ::__max; }  
# 575
__err = ios_base::failbit; 
# 576
} else { 
# 578
__v = (__negative ? -__result : __result); }  }  
# 580
if (__testeof) { 
# 581
(__err |= ios_base::eofbit); }  
# 582
return __beg; 
# 583
} 
# 587
template< class _CharT, class _InIter> _InIter 
# 590
num_get< _CharT, _InIter> ::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 591
__err, bool &__v) const 
# 592
{ 
# 593
if (!(((__io.flags()) & ios_base::boolalpha))) 
# 594
{ 
# 598
long __l = (-1); 
# 599
__beg = _M_extract_int(__beg, __end, __io, __err, __l); 
# 600
if ((__l == (0)) || (__l == (1))) { 
# 601
__v = ((bool)__l); } else 
# 603
{ 
# 606
__v = true; 
# 607
__err = ios_base::failbit; 
# 608
if (__beg == __end) { 
# 609
(__err |= ios_base::eofbit); }  
# 610
}  
# 611
} else 
# 613
{ 
# 615
typedef __numpunct_cache< _CharT>  __cache_type; 
# 616
__use_cache< __numpunct_cache< _CharT> >  __uc; 
# 617
const locale &__loc = __io._M_getloc(); 
# 618
const __cache_type *__lc = __uc(__loc); 
# 620
bool __testf = true; 
# 621
bool __testt = true; 
# 622
bool __donef = (__lc->_M_falsename_size) == 0; 
# 623
bool __donet = (__lc->_M_truename_size) == 0; 
# 624
bool __testeof = false; 
# 625
size_t __n = (0); 
# 626
while ((!__donef) || (!__donet)) 
# 627
{ 
# 628
if (__beg == __end) 
# 629
{ 
# 630
__testeof = true; 
# 631
break; 
# 632
}  
# 634
const char_type __c = *__beg; 
# 636
if (!__donef) { 
# 637
__testf = (__c == ((__lc->_M_falsename)[__n])); }  
# 639
if ((!__testf) && __donet) { 
# 640
break; }  
# 642
if (!__donet) { 
# 643
__testt = (__c == ((__lc->_M_truename)[__n])); }  
# 645
if ((!__testt) && __donef) { 
# 646
break; }  
# 648
if ((!__testt) && (!__testf)) { 
# 649
break; }  
# 651
++__n; 
# 652
++__beg; 
# 654
__donef = ((!__testf) || (__n >= (__lc->_M_falsename_size))); 
# 655
__donet = ((!__testt) || (__n >= (__lc->_M_truename_size))); 
# 656
}  
# 657
if (__testf && (__n == (__lc->_M_falsename_size)) && __n) 
# 658
{ 
# 659
__v = false; 
# 660
if (__testt && (__n == (__lc->_M_truename_size))) { 
# 661
__err = ios_base::failbit; } else { 
# 663
__err = (__testeof ? ios_base::eofbit : ios_base::goodbit); }  
# 664
} else { 
# 665
if (__testt && (__n == (__lc->_M_truename_size)) && __n) 
# 666
{ 
# 667
__v = true; 
# 668
__err = (__testeof ? ios_base::eofbit : ios_base::goodbit); 
# 669
} else 
# 671
{ 
# 674
__v = false; 
# 675
__err = ios_base::failbit; 
# 676
if (__testeof) { 
# 677
(__err |= ios_base::eofbit); }  
# 678
}  }  
# 679
}  
# 680
return __beg; 
# 681
} 
# 683
template< class _CharT, class _InIter> _InIter 
# 686
num_get< _CharT, _InIter> ::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 687
__err, float &__v) const 
# 688
{ 
# 689
string __xtrc; 
# 690
__xtrc.reserve(32); 
# 691
__beg = _M_extract_float(__beg, __end, __io, __err, __xtrc); 
# 692
std::__convert_to_v(__xtrc.c_str(), __v, __err, _S_get_c_locale()); 
# 693
if (__beg == __end) { 
# 694
(__err |= ios_base::eofbit); }  
# 695
return __beg; 
# 696
} 
# 698
template< class _CharT, class _InIter> _InIter 
# 701
num_get< _CharT, _InIter> ::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 702
__err, double &__v) const 
# 703
{ 
# 704
string __xtrc; 
# 705
__xtrc.reserve(32); 
# 706
__beg = _M_extract_float(__beg, __end, __io, __err, __xtrc); 
# 707
std::__convert_to_v(__xtrc.c_str(), __v, __err, _S_get_c_locale()); 
# 708
if (__beg == __end) { 
# 709
(__err |= ios_base::eofbit); }  
# 710
return __beg; 
# 711
} 
# 730
template< class _CharT, class _InIter> _InIter 
# 733
num_get< _CharT, _InIter> ::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 734
__err, long double &__v) const 
# 735
{ 
# 736
string __xtrc; 
# 737
__xtrc.reserve(32); 
# 738
__beg = _M_extract_float(__beg, __end, __io, __err, __xtrc); 
# 739
std::__convert_to_v(__xtrc.c_str(), __v, __err, _S_get_c_locale()); 
# 740
if (__beg == __end) { 
# 741
(__err |= ios_base::eofbit); }  
# 742
return __beg; 
# 743
} 
# 745
template< class _CharT, class _InIter> _InIter 
# 748
num_get< _CharT, _InIter> ::do_get(iter_type __beg, iter_type __end, ios_base &__io, ios_base::iostate &
# 749
__err, void *&__v) const 
# 750
{ 
# 752
typedef ios_base::fmtflags fmtflags; 
# 753
const fmtflags __fmt = __io.flags(); 
# 754
__io.flags((((__fmt & ((~ios_base::basefield)))) | ios_base::hex)); 
# 758
typedef __gnu_cxx::__conditional_type< true, unsigned long, unsigned long long> ::__type _UIntPtrType; 
# 760
_UIntPtrType __ul; 
# 761
__beg = _M_extract_int(__beg, __end, __io, __err, __ul); 
# 764
__io.flags(__fmt); 
# 766
__v = (reinterpret_cast< void *>(__ul)); 
# 767
return __beg; 
# 768
} 
# 772
template< class _CharT, class _OutIter> void 
# 775
num_put< _CharT, _OutIter> ::_M_pad(_CharT __fill, streamsize __w, ios_base &__io, _CharT *
# 776
__new, const _CharT *__cs, int &__len) const 
# 777
{ 
# 780
__pad< _CharT, char_traits< _CharT> > ::_S_pad(__io, __fill, __new, __cs, __w, __len); 
# 782
__len = (static_cast< int>(__w)); 
# 783
} 
# 787
template< class _CharT, class _ValueT> int 
# 789
__int_to_char(_CharT *__bufend, _ValueT __v, const _CharT *__lit, ios_base::fmtflags 
# 790
__flags, bool __dec) 
# 791
{ 
# 792
_CharT *__buf = __bufend; 
# 793
if (__builtin_expect(__dec, true)) 
# 794
{ 
# 796
do 
# 797
{ 
# 798
(*(--__buf)) = (__lit[(__v % 10) + __num_base::_S_odigits]); 
# 799
__v /= 10; 
# 800
} 
# 801
while (__v != 0); 
# 802
} else { 
# 803
if (((__flags & ios_base::basefield)) == ios_base::oct) 
# 804
{ 
# 806
do 
# 807
{ 
# 808
(*(--__buf)) = (__lit[(__v & 7) + __num_base::_S_odigits]); 
# 809
__v >>= 3; 
# 810
} 
# 811
while (__v != 0); 
# 812
} else 
# 814
{ 
# 816
const bool __uppercase = (__flags & ios_base::uppercase); 
# 817
const int __case_offset = __uppercase ? __num_base::_S_oudigits : __num_base::_S_odigits; 
# 819
do 
# 820
{ 
# 821
(*(--__buf)) = (__lit[(__v & 15) + __case_offset]); 
# 822
__v >>= 4; 
# 823
} 
# 824
while (__v != 0); 
# 825
}  }  
# 826
return __bufend - __buf; 
# 827
} 
# 831
template< class _CharT, class _OutIter> void 
# 834
num_put< _CharT, _OutIter> ::_M_group_int(const char *__grouping, size_t __grouping_size, _CharT __sep, ios_base &, _CharT *
# 835
__new, _CharT *__cs, int &__len) const 
# 836
{ 
# 837
_CharT *__p = std::__add_grouping(__new, __sep, __grouping, __grouping_size, __cs, __cs + __len); 
# 839
__len = (__p - __new); 
# 840
} 
# 842
template< class _CharT, class _OutIter> 
# 843
template< class _ValueT> _OutIter 
# 846
num_put< _CharT, _OutIter> ::_M_insert_int(_OutIter __s, ios_base &__io, _CharT __fill, _ValueT 
# 847
__v) const 
# 848
{ 
# 849
using __gnu_cxx::__add_unsigned;
# 850
typedef typename __gnu_cxx::__add_unsigned< _ValueT> ::__type __unsigned_type; 
# 851
typedef __numpunct_cache< _CharT>  __cache_type; 
# 852
__use_cache< __numpunct_cache< _CharT> >  __uc; 
# 853
const locale &__loc = __io._M_getloc(); 
# 854
const __cache_type *__lc = __uc(__loc); 
# 855
const _CharT *__lit = ((__lc->_M_atoms_out)); 
# 856
const ios_base::fmtflags __flags = __io.flags(); 
# 859
const int __ilen = ((5) * sizeof(_ValueT)); 
# 860
_CharT *__cs = static_cast< _CharT *>(__builtin_alloca(sizeof(_CharT) * __ilen)); 
# 865
const ios_base::fmtflags __basefield = (__flags & ios_base::basefield); 
# 866
const bool __dec = (__basefield != ios_base::oct) && (__basefield != ios_base::hex); 
# 868
const __unsigned_type __u = ((__v > 0) || (!__dec)) ? (__unsigned_type)__v : (-((__unsigned_type)__v)); 
# 871
int __len = __int_to_char(__cs + __ilen, __u, __lit, __flags, __dec); 
# 872
__cs += (__ilen - __len); 
# 875
if (__lc->_M_use_grouping) 
# 876
{ 
# 879
_CharT *__cs2 = static_cast< _CharT *>(__builtin_alloca((sizeof(_CharT) * (__len + 1)) * (2))); 
# 882
_M_group_int((__lc->_M_grouping), (__lc->_M_grouping_size), (__lc->_M_thousands_sep), __io, __cs2 + 2, __cs, __len); 
# 884
__cs = (__cs2 + 2); 
# 885
}  
# 888
if (__builtin_expect(__dec, true)) 
# 889
{ 
# 891
if (__v >= 0) 
# 892
{ 
# 893
if (((bool)((__flags & ios_base::showpos))) && __gnu_cxx::__numeric_traits< _ValueT> ::__is_signed) { 
# 895
((*(--__cs)) = (__lit[__num_base::_S_oplus])), (++__len); }  
# 896
} else { 
# 898
((*(--__cs)) = (__lit[__num_base::_S_ominus])), (++__len); }  
# 899
} else { 
# 900
if (((bool)((__flags & ios_base::showbase))) && __v) 
# 901
{ 
# 902
if (__basefield == ios_base::oct) { 
# 903
((*(--__cs)) = (__lit[__num_base::_S_odigits])), (++__len); } else 
# 905
{ 
# 907
const bool __uppercase = (__flags & ios_base::uppercase); 
# 908
(*(--__cs)) = (__lit[(__num_base::_S_ox) + __uppercase]); 
# 910
(*(--__cs)) = (__lit[__num_base::_S_odigits]); 
# 911
__len += 2; 
# 912
}  
# 913
}  }  
# 916
const streamsize __w = __io.width(); 
# 917
if (__w > (static_cast< streamsize>(__len))) 
# 918
{ 
# 919
_CharT *__cs3 = static_cast< _CharT *>(__builtin_alloca(sizeof(_CharT) * __w)); 
# 921
_M_pad(__fill, __w, __io, __cs3, __cs, __len); 
# 922
__cs = __cs3; 
# 923
}  
# 924
__io.width(0); 
# 928
return std::__write(__s, __cs, __len); 
# 929
} 
# 931
template< class _CharT, class _OutIter> void 
# 934
num_put< _CharT, _OutIter> ::_M_group_float(const char *__grouping, size_t __grouping_size, _CharT 
# 935
__sep, const _CharT *__p, _CharT *__new, _CharT *
# 936
__cs, int &__len) const 
# 937
{ 
# 941
const int __declen = (__p) ? __p - __cs : __len; 
# 942
_CharT *__p2 = std::__add_grouping(__new, __sep, __grouping, __grouping_size, __cs, __cs + __declen); 
# 947
int __newlen = __p2 - __new; 
# 948
if (__p) 
# 949
{ 
# 950
char_traits< _CharT> ::copy(__p2, __p, __len - __declen); 
# 951
__newlen += (__len - __declen); 
# 952
}  
# 953
__len = __newlen; 
# 954
} 
# 966
template< class _CharT, class _OutIter> 
# 967
template< class _ValueT> _OutIter 
# 970
num_put< _CharT, _OutIter> ::_M_insert_float(_OutIter __s, ios_base &__io, _CharT __fill, char __mod, _ValueT 
# 971
__v) const 
# 972
{ 
# 973
typedef __numpunct_cache< _CharT>  __cache_type; 
# 974
__use_cache< __numpunct_cache< _CharT> >  __uc; 
# 975
const locale &__loc = __io._M_getloc(); 
# 976
const __cache_type *__lc = __uc(__loc); 
# 979
const streamsize __prec = (__io.precision() < (0)) ? 6 : __io.precision(); 
# 981
const int __max_digits = (__gnu_cxx::__numeric_traits< _ValueT> ::__digits10); 
# 985
int __len; 
# 987
char __fbuf[16]; 
# 988
__num_base::_S_format_float(__io, __fbuf, __mod); 
# 993
int __cs_size = (__max_digits * 3); 
# 994
char *__cs = static_cast< char *>(__builtin_alloca(__cs_size)); 
# 995
__len = std::__convert_from_v(_S_get_c_locale(), __cs, __cs_size, __fbuf, __prec, __v); 
# 999
if (__len >= __cs_size) 
# 1000
{ 
# 1001
__cs_size = (__len + 1); 
# 1002
__cs = (static_cast< char *>(__builtin_alloca(__cs_size))); 
# 1003
__len = std::__convert_from_v(_S_get_c_locale(), __cs, __cs_size, __fbuf, __prec, __v); 
# 1005
}  
# 1027
const ctype< _CharT>  &__ctype = use_facet< ctype< _CharT> > (__loc); 
# 1029
_CharT *__ws = static_cast< _CharT *>(__builtin_alloca(sizeof(_CharT) * __len)); 
# 1031
(__ctype.widen(__cs, __cs + __len, __ws)); 
# 1034
_CharT *__wp = (0); 
# 1035
const char *__p = char_traits< char> ::find(__cs, __len, '.'); 
# 1036
if (__p) 
# 1037
{ 
# 1038
__wp = (__ws + (__p - __cs)); 
# 1039
(*__wp) = (__lc->_M_decimal_point); 
# 1040
}  
# 1045
if ((__lc->_M_use_grouping) && ((__wp || (__len < 3)) || (((__cs[1]) <= ('9')) && ((__cs[2]) <= ('9')) && ((__cs[1]) >= ('0')) && ((__cs[2]) >= ('0'))))) 
# 1048
{ 
# 1051
_CharT *__ws2 = static_cast< _CharT *>(__builtin_alloca((sizeof(_CharT) * __len) * (2))); 
# 1054
streamsize __off = (0); 
# 1055
if (((__cs[0]) == ('-')) || ((__cs[0]) == ('+'))) 
# 1056
{ 
# 1057
__off = (1); 
# 1058
(__ws2[0]) = (__ws[0]); 
# 1059
__len -= 1; 
# 1060
}  
# 1062
_M_group_float((__lc->_M_grouping), (__lc->_M_grouping_size), (__lc->_M_thousands_sep), __wp, __ws2 + __off, __ws + __off, __len); 
# 1065
__len += __off; 
# 1067
__ws = __ws2; 
# 1068
}  
# 1071
const streamsize __w = __io.width(); 
# 1072
if (__w > (static_cast< streamsize>(__len))) 
# 1073
{ 
# 1074
_CharT *__ws3 = static_cast< _CharT *>(__builtin_alloca(sizeof(_CharT) * __w)); 
# 1076
_M_pad(__fill, __w, __io, __ws3, __ws, __len); 
# 1077
__ws = __ws3; 
# 1078
}  
# 1079
__io.width(0); 
# 1083
return std::__write(__s, __ws, __len); 
# 1084
} 
# 1086
template< class _CharT, class _OutIter> _OutIter 
# 1089
num_put< _CharT, _OutIter> ::do_put(iter_type __s, ios_base &__io, char_type __fill, bool __v) const 
# 1090
{ 
# 1091
const ios_base::fmtflags __flags = __io.flags(); 
# 1092
if (((__flags & ios_base::boolalpha)) == 0) 
# 1093
{ 
# 1094
const long __l = __v; 
# 1095
__s = _M_insert_int(__s, __io, __fill, __l); 
# 1096
} else 
# 1098
{ 
# 1099
typedef __numpunct_cache< _CharT>  __cache_type; 
# 1100
__use_cache< __numpunct_cache< _CharT> >  __uc; 
# 1101
const locale &__loc = __io._M_getloc(); 
# 1102
const __cache_type *__lc = __uc(__loc); 
# 1104
const _CharT *__name = __v ? __lc->_M_truename : (__lc->_M_falsename); 
# 1106
int __len = __v ? __lc->_M_truename_size : (__lc->_M_falsename_size); 
# 1109
const streamsize __w = __io.width(); 
# 1110
if (__w > (static_cast< streamsize>(__len))) 
# 1111
{ 
# 1112
const streamsize __plen = __w - __len; 
# 1113
_CharT *__ps = static_cast< _CharT *>(__builtin_alloca(sizeof(_CharT) * __plen)); 
# 1117
char_traits< _CharT> ::assign(__ps, __plen, __fill); 
# 1118
__io.width(0); 
# 1120
if (((__flags & ios_base::adjustfield)) == ios_base::left) 
# 1121
{ 
# 1122
__s = std::__write(__s, __name, __len); 
# 1123
__s = std::__write(__s, __ps, __plen); 
# 1124
} else 
# 1126
{ 
# 1127
__s = std::__write(__s, __ps, __plen); 
# 1128
__s = std::__write(__s, __name, __len); 
# 1129
}  
# 1130
return __s; 
# 1131
}  
# 1132
__io.width(0); 
# 1133
__s = std::__write(__s, __name, __len); 
# 1134
}  
# 1135
return __s; 
# 1136
} 
# 1138
template< class _CharT, class _OutIter> _OutIter 
# 1141
num_put< _CharT, _OutIter> ::do_put(iter_type __s, ios_base &__io, char_type __fill, double __v) const 
# 1142
{ return _M_insert_float(__s, __io, __fill, ((char)0), __v); } 
# 1152
template< class _CharT, class _OutIter> _OutIter 
# 1155
num_put< _CharT, _OutIter> ::do_put(iter_type __s, ios_base &__io, char_type __fill, long double 
# 1156
__v) const 
# 1157
{ return _M_insert_float(__s, __io, __fill, 'L', __v); } 
# 1159
template< class _CharT, class _OutIter> _OutIter 
# 1162
num_put< _CharT, _OutIter> ::do_put(iter_type __s, ios_base &__io, char_type __fill, const void *
# 1163
__v) const 
# 1164
{ 
# 1165
const ios_base::fmtflags __flags = __io.flags(); 
# 1166
const ios_base::fmtflags __fmt = (~((ios_base::basefield | ios_base::uppercase))); 
# 1168
__io.flags((((__flags & __fmt)) | ((ios_base::hex | ios_base::showbase)))); 
# 1172
typedef __gnu_cxx::__conditional_type< true, unsigned long, unsigned long long> ::__type _UIntPtrType; 
# 1174
__s = _M_insert_int(__s, __io, __fill, reinterpret_cast< _UIntPtrType>(__v)); 
# 1176
__io.flags(__flags); 
# 1177
return __s; 
# 1178
} 
# 1189
template< class _CharT, class _Traits> void 
# 1191
__pad< _CharT, _Traits> ::_S_pad(ios_base &__io, _CharT __fill, _CharT *
# 1192
__news, const _CharT *__olds, streamsize 
# 1193
__newlen, streamsize __oldlen) 
# 1194
{ 
# 1195
const size_t __plen = static_cast< size_t>(__newlen - __oldlen); 
# 1196
const ios_base::fmtflags __adjust = ((__io.flags()) & ios_base::adjustfield); 
# 1199
if (__adjust == ios_base::left) 
# 1200
{ 
# 1201
_Traits::copy(__news, __olds, __oldlen); 
# 1202
_Traits::assign(__news + __oldlen, __plen, __fill); 
# 1203
return; 
# 1204
}  
# 1206
size_t __mod = (0); 
# 1207
if (__adjust == ios_base::internal) 
# 1208
{ 
# 1212
const locale &__loc = __io._M_getloc(); 
# 1213
const ctype< _CharT>  &__ctype = use_facet< ctype< _CharT> > (__loc); 
# 1215
if (((__ctype.widen('-')) == (__olds[0])) || ((__ctype.widen('+')) == (__olds[0]))) 
# 1217
{ 
# 1218
(__news[0]) = (__olds[0]); 
# 1219
__mod = (1); 
# 1220
++__news; 
# 1221
} else { 
# 1222
if (((__ctype.widen('0')) == (__olds[0])) && (__oldlen > (1)) && (((__ctype.widen('x')) == (__olds[1])) || ((__ctype.widen('X')) == (__olds[1])))) 
# 1226
{ 
# 1227
(__news[0]) = (__olds[0]); 
# 1228
(__news[1]) = (__olds[1]); 
# 1229
__mod = (2); 
# 1230
__news += 2; 
# 1231
}  }  
# 1233
}  
# 1234
_Traits::assign(__news, __plen, __fill); 
# 1235
_Traits::copy(__news + __plen, __olds + __mod, __oldlen - __mod); 
# 1236
} 
# 1238
template< class _CharT> _CharT *
# 1240
__add_grouping(_CharT *__s, _CharT __sep, const char *
# 1241
__gbeg, size_t __gsize, const _CharT *
# 1242
__first, const _CharT *__last) 
# 1243
{ 
# 1244
size_t __idx = (0); 
# 1245
size_t __ctr = (0); 
# 1247
while (((__last - __first) > (__gbeg[__idx])) && ((static_cast< signed char>(__gbeg[__idx])) > 0) && ((__gbeg[__idx]) != __gnu_cxx::__numeric_traits_integer< char> ::__max)) 
# 1250
{ 
# 1251
__last -= (__gbeg[__idx]); 
# 1252
(__idx < (__gsize - (1))) ? ++__idx : (++__ctr); 
# 1253
}  
# 1255
while (__first != __last) { 
# 1256
(*(__s++)) = (*(__first++)); }  
# 1258
while (__ctr--) 
# 1259
{ 
# 1260
(*(__s++)) = __sep; 
# 1261
for (char __i = __gbeg[__idx]; __i > 0; --__i) { 
# 1262
(*(__s++)) = (*(__first++)); }  
# 1263
}  
# 1265
while (__idx--) 
# 1266
{ 
# 1267
(*(__s++)) = __sep; 
# 1268
for (char __i = __gbeg[__idx]; __i > 0; --__i) { 
# 1269
(*(__s++)) = (*(__first++)); }  
# 1270
}  
# 1272
return __s; 
# 1273
} 
# 1278
extern template class numpunct< char> ;
# 1279
extern template class numpunct_byname< char> ;
# 1280
extern template class num_get< char, istreambuf_iterator< char, char_traits< char> > > ;
# 1281
extern template class num_put< char, ostreambuf_iterator< char, char_traits< char> > > ;
# 1284
extern template const ctype< char>  &use_facet< ctype< char> > (const locale &);
# 1288
extern template const numpunct< char>  &use_facet< numpunct< char> > (const locale &);
# 1292
extern template const num_put< char, ostreambuf_iterator< char, char_traits< char> > >  &use_facet< num_put< char, ostreambuf_iterator< char, char_traits< char> > > > (const locale &);
# 1296
extern template const num_get< char, istreambuf_iterator< char, char_traits< char> > >  &use_facet< num_get< char, istreambuf_iterator< char, char_traits< char> > > > (const locale &);
# 1300
extern template bool has_facet< ctype< char> > (const locale &) throw();
# 1304
extern template bool has_facet< numpunct< char> > (const locale &) throw();
# 1308
extern template bool has_facet< num_put< char, ostreambuf_iterator< char, char_traits< char> > > > (const locale &) throw();
# 1312
extern template bool has_facet< num_get< char, istreambuf_iterator< char, char_traits< char> > > > (const locale &) throw();
# 1317
extern template class numpunct< wchar_t> ;
# 1318
extern template class numpunct_byname< wchar_t> ;
# 1319
extern template class num_get< wchar_t, istreambuf_iterator< wchar_t, char_traits< wchar_t> > > ;
# 1320
extern template class num_put< wchar_t, ostreambuf_iterator< wchar_t, char_traits< wchar_t> > > ;
# 1323
extern template const ctype< wchar_t>  &use_facet< ctype< wchar_t> > (const locale &);
# 1327
extern template const numpunct< wchar_t>  &use_facet< numpunct< wchar_t> > (const locale &);
# 1331
extern template const num_put< wchar_t, ostreambuf_iterator< wchar_t, char_traits< wchar_t> > >  &use_facet< num_put< wchar_t, ostreambuf_iterator< wchar_t, char_traits< wchar_t> > > > (const locale &);
# 1335
extern template const num_get< wchar_t, istreambuf_iterator< wchar_t, char_traits< wchar_t> > >  &use_facet< num_get< wchar_t, istreambuf_iterator< wchar_t, char_traits< wchar_t> > > > (const locale &);
# 1339
extern template bool has_facet< ctype< wchar_t> > (const locale &) throw();
# 1343
extern template bool has_facet< numpunct< wchar_t> > (const locale &) throw();
# 1347
extern template bool has_facet< num_put< wchar_t, ostreambuf_iterator< wchar_t, char_traits< wchar_t> > > > (const locale &) throw();
# 1351
extern template bool has_facet< num_get< wchar_t, istreambuf_iterator< wchar_t, char_traits< wchar_t> > > > (const locale &) throw();
# 1358
}
# 40 "/usr/include/c++/4.8/bits/basic_ios.h" 3
namespace std __attribute((__visibility__("default"))) { 
# 44
template< class _Facet> inline const _Facet &
# 46
__check_facet(const _Facet *__f) 
# 47
{ 
# 48
if (!__f) { 
# 49
__throw_bad_cast(); }  
# 50
return *__f; 
# 51
} 
# 65
template< class _CharT, class _Traits> 
# 66
class basic_ios : public ios_base { 
# 75
public: typedef _CharT char_type; 
# 76
typedef typename _Traits::int_type int_type; 
# 77
typedef typename _Traits::pos_type pos_type; 
# 78
typedef typename _Traits::off_type off_type; 
# 79
typedef _Traits traits_type; 
# 86
typedef ctype< _CharT>  __ctype_type; 
# 88
typedef num_put< _CharT, ostreambuf_iterator< _CharT, _Traits> >  __num_put_type; 
# 90
typedef num_get< _CharT, istreambuf_iterator< _CharT, _Traits> >  __num_get_type; 
# 95
protected: basic_ostream< _CharT, _Traits>  *_M_tie; 
# 96
mutable char_type _M_fill; 
# 97
mutable bool _M_fill_init; 
# 98
basic_streambuf< _CharT, _Traits>  *_M_streambuf; 
# 101
const __ctype_type *_M_ctype; 
# 103
const __num_put_type *_M_num_put; 
# 105
const __num_get_type *_M_num_get; 
# 115
public: operator void *() const 
# 116
{ return this->fail() ? 0 : (const_cast< basic_ios *>(this)); } 
# 119
bool operator!() const 
# 120
{ return this->fail(); } 
# 131
iostate rdstate() const 
# 132
{ return ios_base::_M_streambuf_state; } 
# 142
void clear(iostate __state = goodbit); 
# 151
void setstate(iostate __state) 
# 152
{ this->clear(((this->rdstate()) | __state)); } 
# 158
void _M_setstate(iostate __state) 
# 159
{ 
# 162
((ios_base::_M_streambuf_state) |= __state); 
# 163
if (((this->exceptions()) & __state)) { 
# 164
throw; }  
# 165
} 
# 174
bool good() const 
# 175
{ return (this->rdstate()) == 0; } 
# 184
bool eof() const 
# 185
{ return (((this->rdstate()) & eofbit)) != 0; } 
# 195
bool fail() const 
# 196
{ return (((this->rdstate()) & ((badbit | failbit)))) != 0; } 
# 205
bool bad() const 
# 206
{ return (((this->rdstate()) & badbit)) != 0; } 
# 216
iostate exceptions() const 
# 217
{ return ios_base::_M_exception; } 
# 251
void exceptions(iostate __except) 
# 252
{ 
# 253
(ios_base::_M_exception) = __except; 
# 254
this->clear(ios_base::_M_streambuf_state); 
# 255
} 
# 264
explicit basic_ios(basic_streambuf< _CharT, _Traits>  *__sb) : ios_base(), _M_tie((0)), _M_fill(), _M_fill_init(false), _M_streambuf((0)), _M_ctype((0)), _M_num_put((0)), _M_num_get((0)) 
# 267
{ this->init(__sb); } 
# 276
virtual ~basic_ios() { } 
# 289
basic_ostream< _CharT, _Traits>  *tie() const 
# 290
{ return _M_tie; } 
# 301
basic_ostream< _CharT, _Traits>  *tie(basic_ostream< _CharT, _Traits>  *__tiestr) 
# 302
{ 
# 303
basic_ostream< _CharT, _Traits>  *__old = _M_tie; 
# 304
(_M_tie) = __tiestr; 
# 305
return __old; 
# 306
} 
# 315
basic_streambuf< _CharT, _Traits>  *rdbuf() const 
# 316
{ return _M_streambuf; } 
# 341
basic_streambuf< _CharT, _Traits>  *rdbuf(basic_streambuf< _CharT, _Traits>  * __sb); 
# 355
basic_ios &copyfmt(const basic_ios & __rhs); 
# 364
char_type fill() const 
# 365
{ 
# 366
if (!(_M_fill_init)) 
# 367
{ 
# 368
(_M_fill) = this->widen(' '); 
# 369
(_M_fill_init) = true; 
# 370
}  
# 371
return _M_fill; 
# 372
} 
# 384
char_type fill(char_type __ch) 
# 385
{ 
# 386
char_type __old = (this->fill()); 
# 387
(_M_fill) = __ch; 
# 388
return __old; 
# 389
} 
# 404
locale imbue(const locale & __loc); 
# 424
char narrow(char_type __c, char __dfault) const 
# 425
{ return (__check_facet(_M_ctype).narrow(__c, __dfault)); } 
# 443
char_type widen(char __c) const 
# 444
{ return (__check_facet(_M_ctype).widen(__c)); } 
# 454
protected: basic_ios() : ios_base(), _M_tie((0)), _M_fill(char_type()), _M_fill_init(false), _M_streambuf((0)), _M_ctype((0)), _M_num_put((0)), _M_num_get((0)) 
# 457
{ } 
# 466
void init(basic_streambuf< _CharT, _Traits>  * __sb); 
# 469
void _M_cache_locale(const locale & __loc); 
# 470
}; 
# 473
}
# 35 "/usr/include/c++/4.8/bits/basic_ios.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 39
template< class _CharT, class _Traits> void 
# 41
basic_ios< _CharT, _Traits> ::clear(iostate __state) 
# 42
{ 
# 43
if ((this->rdbuf())) { 
# 44
(ios_base::_M_streambuf_state) = __state; } else { 
# 46
(ios_base::_M_streambuf_state) = ((__state | badbit)); }  
# 47
if (((this->exceptions()) & (this->rdstate()))) { 
# 48
__throw_ios_failure("basic_ios::clear"); }  
# 49
} 
# 51
template< class _CharT, class _Traits> basic_streambuf< _CharT, _Traits>  *
# 53
basic_ios< _CharT, _Traits> ::rdbuf(basic_streambuf< _CharT, _Traits>  *__sb) 
# 54
{ 
# 55
basic_streambuf< _CharT, _Traits>  *__old = _M_streambuf; 
# 56
(_M_streambuf) = __sb; 
# 57
this->clear(); 
# 58
return __old; 
# 59
} 
# 61
template< class _CharT, class _Traits> basic_ios< _CharT, _Traits>  &
# 63
basic_ios< _CharT, _Traits> ::copyfmt(const basic_ios &__rhs) 
# 64
{ 
# 67
if (this != (&__rhs)) 
# 68
{ 
# 73
_Words *__words = ((__rhs.ios_base::_M_word_size) <= (_S_local_word_size)) ? ios_base::_M_local_word : (new _Words [__rhs.ios_base::_M_word_size]); 
# 77
_Callback_list *__cb = __rhs.ios_base::_M_callbacks; 
# 78
if (__cb) { 
# 79
__cb->_M_add_reference(); }  
# 80
this->ios_base::_M_call_callbacks(erase_event); 
# 81
if ((ios_base::_M_word) != (ios_base::_M_local_word)) 
# 82
{ 
# 83
delete [] (ios_base::_M_word); 
# 84
(ios_base::_M_word) = (0); 
# 85
}  
# 86
this->ios_base::_M_dispose_callbacks(); 
# 89
(ios_base::_M_callbacks) = __cb; 
# 90
for (int __i = 0; __i < (__rhs.ios_base::_M_word_size); ++__i) { 
# 91
(__words[__i]) = ((__rhs.ios_base::_M_word)[__i]); }  
# 92
(ios_base::_M_word) = __words; 
# 93
(ios_base::_M_word_size) = (__rhs.ios_base::_M_word_size); 
# 95
this->flags(__rhs.flags()); 
# 96
this->width(__rhs.width()); 
# 97
this->precision(__rhs.precision()); 
# 98
(this->tie((__rhs.tie()))); 
# 99
(this->fill((__rhs.fill()))); 
# 100
((ios_base::_M_ios_locale) = (__rhs.getloc())); 
# 101
_M_cache_locale(ios_base::_M_ios_locale); 
# 103
this->ios_base::_M_call_callbacks(copyfmt_event); 
# 106
this->exceptions(__rhs.exceptions()); 
# 107
}  
# 108
return *this; 
# 109
} 
# 112
template< class _CharT, class _Traits> locale 
# 114
basic_ios< _CharT, _Traits> ::imbue(const locale &__loc) 
# 115
{ 
# 116
locale __old(this->getloc()); 
# 117
this->ios_base::imbue(__loc); 
# 118
_M_cache_locale(__loc); 
# 119
if ((this->rdbuf()) != 0) { 
# 120
((this->rdbuf())->pubimbue(__loc)); }  
# 121
return __old; 
# 122
} 
# 124
template< class _CharT, class _Traits> void 
# 126
basic_ios< _CharT, _Traits> ::init(basic_streambuf< _CharT, _Traits>  *__sb) 
# 127
{ 
# 129
this->ios_base::_M_init(); 
# 132
_M_cache_locale(ios_base::_M_ios_locale); 
# 146
(_M_fill) = _CharT(); 
# 147
(_M_fill_init) = false; 
# 149
(_M_tie) = 0; 
# 150
(ios_base::_M_exception) = goodbit; 
# 151
(_M_streambuf) = __sb; 
# 152
(ios_base::_M_streambuf_state) = ((__sb) ? goodbit : badbit); 
# 153
} 
# 155
template< class _CharT, class _Traits> void 
# 157
basic_ios< _CharT, _Traits> ::_M_cache_locale(const locale &__loc) 
# 158
{ 
# 159
if (__builtin_expect(has_facet< __ctype_type> (__loc), true)) { 
# 160
(_M_ctype) = (&use_facet< __ctype_type> (__loc)); } else { 
# 162
(_M_ctype) = 0; }  
# 164
if (__builtin_expect(has_facet< __num_put_type> (__loc), true)) { 
# 165
(_M_num_put) = (&use_facet< __num_put_type> (__loc)); } else { 
# 167
(_M_num_put) = 0; }  
# 169
if (__builtin_expect(has_facet< __num_get_type> (__loc), true)) { 
# 170
(_M_num_get) = (&use_facet< __num_get_type> (__loc)); } else { 
# 172
(_M_num_get) = 0; }  
# 173
} 
# 178
extern template class basic_ios< char, char_traits< char> > ;
# 181
extern template class basic_ios< wchar_t, char_traits< wchar_t> > ;
# 186
}
# 41 "/usr/include/c++/4.8/ostream" 3
namespace std __attribute((__visibility__("default"))) { 
# 57
template< class _CharT, class _Traits> 
# 58
class basic_ostream : virtual public basic_ios< _CharT, _Traits>  { 
# 62
public: typedef _CharT char_type; 
# 63
typedef typename _Traits::int_type int_type; 
# 64
typedef typename _Traits::pos_type pos_type; 
# 65
typedef typename _Traits::off_type off_type; 
# 66
typedef _Traits traits_type; 
# 69
typedef basic_streambuf< _CharT, _Traits>  __streambuf_type; 
# 70
typedef ::std::basic_ios< _CharT, _Traits>  __ios_type; 
# 71
typedef basic_ostream __ostream_type; 
# 73
typedef num_put< _CharT, ostreambuf_iterator< _CharT, _Traits> >  __num_put_type; 
# 74
typedef ctype< _CharT>  __ctype_type; 
# 84
explicit basic_ostream(__streambuf_type *__sb) 
# 85
{ (this->init(__sb)); } 
# 93
virtual ~basic_ostream() { } 
# 96
class sentry; 
# 97
friend class sentry; 
# 108
__ostream_type &operator<<(__ostream_type &(*__pf)(__ostream_type &)) 
# 109
{ 
# 113
return __pf(*this); 
# 114
} 
# 117
__ostream_type &operator<<(__ios_type &(*__pf)(__ios_type &)) 
# 118
{ 
# 122
__pf(*this); 
# 123
return *this; 
# 124
} 
# 127
__ostream_type &operator<<(::std::ios_base &(*__pf)(::std::ios_base &)) 
# 128
{ 
# 132
__pf(*this); 
# 133
return *this; 
# 134
} 
# 166
__ostream_type &operator<<(long __n) 
# 167
{ return _M_insert(__n); } 
# 170
__ostream_type &operator<<(unsigned long __n) 
# 171
{ return _M_insert(__n); } 
# 174
__ostream_type &operator<<(bool __n) 
# 175
{ return _M_insert(__n); } 
# 178
__ostream_type &operator<<(short __n); 
# 181
__ostream_type &operator<<(unsigned short __n) 
# 182
{ 
# 185
return _M_insert(static_cast< unsigned long>(__n)); 
# 186
} 
# 189
__ostream_type &operator<<(int __n); 
# 192
__ostream_type &operator<<(unsigned __n) 
# 193
{ 
# 196
return _M_insert(static_cast< unsigned long>(__n)); 
# 197
} 
# 201
__ostream_type &operator<<(long long __n) 
# 202
{ return _M_insert(__n); } 
# 205
__ostream_type &operator<<(unsigned long long __n) 
# 206
{ return _M_insert(__n); } 
# 220
__ostream_type &operator<<(double __f) 
# 221
{ return _M_insert(__f); } 
# 224
__ostream_type &operator<<(float __f) 
# 225
{ 
# 228
return _M_insert(static_cast< double>(__f)); 
# 229
} 
# 232
__ostream_type &operator<<(long double __f) 
# 233
{ return _M_insert(__f); } 
# 245
__ostream_type &operator<<(const void *__p) 
# 246
{ return _M_insert(__p); } 
# 270
__ostream_type &operator<<(__streambuf_type * __sb); 
# 303
__ostream_type &put(char_type __c); 
# 311
void _M_write(const char_type *__s, ::std::streamsize __n) 
# 312
{ 
# 313
const ::std::streamsize __put = ((this->rdbuf())->sputn(__s, __n)); 
# 314
if (__put != __n) { 
# 315
(this->setstate(ios_base::badbit)); }  
# 316
} 
# 335
__ostream_type &write(const char_type * __s, ::std::streamsize __n); 
# 348
__ostream_type &flush(); 
# 358
pos_type tellp(); 
# 369
__ostream_type &seekp(pos_type); 
# 381
__ostream_type &seekp(off_type, ::std::ios_base::seekdir); 
# 384
protected: basic_ostream() 
# 385
{ (this->init(0)); } 
# 387
template< class _ValueT> __ostream_type &_M_insert(_ValueT __v); 
# 390
}; 
# 399
template< class _CharT, class _Traits> 
# 400
class basic_ostream< _CharT, _Traits> ::sentry { 
# 403
bool _M_ok; 
# 404
basic_ostream &_M_os; 
# 419
public: explicit sentry(basic_ostream & __os); 
# 428
~sentry() 
# 429
{ 
# 431
if (((bool)(((_M_os).flags()) & ios_base::unitbuf)) && (!uncaught_exception())) 
# 432
{ 
# 434
if (((_M_os).rdbuf()) && ((((_M_os).rdbuf())->pubsync()) == (-1))) { 
# 435
((_M_os).setstate(ios_base::badbit)); }  
# 436
}  
# 437
} 
# 449
explicit operator bool() const 
# 450
{ return _M_ok; } 
# 451
}; 
# 469
template< class _CharT, class _Traits> inline basic_ostream< _CharT, _Traits>  &
# 471
operator<<(basic_ostream< _CharT, _Traits>  &__out, _CharT __c) 
# 472
{ return __ostream_insert(__out, &__c, 1); } 
# 474
template< class _CharT, class _Traits> inline basic_ostream< _CharT, _Traits>  &
# 476
operator<<(basic_ostream< _CharT, _Traits>  &__out, char __c) 
# 477
{ return __out << (__out.widen(__c)); } 
# 480
template< class _Traits> inline basic_ostream< char, _Traits>  &
# 482
operator<<(basic_ostream< char, _Traits>  &__out, char __c) 
# 483
{ return __ostream_insert(__out, &__c, 1); } 
# 486
template< class _Traits> inline basic_ostream< char, _Traits>  &
# 488
operator<<(basic_ostream< char, _Traits>  &__out, signed char __c) 
# 489
{ return __out << (static_cast< char>(__c)); } 
# 491
template< class _Traits> inline basic_ostream< char, _Traits>  &
# 493
operator<<(basic_ostream< char, _Traits>  &__out, unsigned char __c) 
# 494
{ return __out << (static_cast< char>(__c)); } 
# 511
template< class _CharT, class _Traits> inline basic_ostream< _CharT, _Traits>  &
# 513
operator<<(basic_ostream< _CharT, _Traits>  &__out, const _CharT *__s) 
# 514
{ 
# 515
if (!__s) { 
# 516
(__out.setstate(ios_base::badbit)); } else { 
# 518
__ostream_insert(__out, __s, static_cast< streamsize>(_Traits::length(__s))); }  
# 520
return __out; 
# 521
} 
# 523
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &operator<<(basic_ostream< _CharT, _Traits>  & __out, const char * __s); 
# 528
template< class _Traits> inline basic_ostream< char, _Traits>  &
# 530
operator<<(basic_ostream< char, _Traits>  &__out, const char *__s) 
# 531
{ 
# 532
if (!__s) { 
# 533
(__out.setstate(ios_base::badbit)); } else { 
# 535
__ostream_insert(__out, __s, static_cast< streamsize>(_Traits::length(__s))); }  
# 537
return __out; 
# 538
} 
# 541
template< class _Traits> inline basic_ostream< char, _Traits>  &
# 543
operator<<(basic_ostream< char, _Traits>  &__out, const signed char *__s) 
# 544
{ return __out << (reinterpret_cast< const char *>(__s)); } 
# 546
template< class _Traits> inline basic_ostream< char, _Traits>  &
# 548
operator<<(basic_ostream< char, _Traits>  &__out, const unsigned char *__s) 
# 549
{ return __out << (reinterpret_cast< const char *>(__s)); } 
# 562
template< class _CharT, class _Traits> inline basic_ostream< _CharT, _Traits>  &
# 564
endl(basic_ostream< _CharT, _Traits>  &__os) 
# 565
{ return flush((__os.put((__os.widen('\n'))))); } 
# 574
template< class _CharT, class _Traits> inline basic_ostream< _CharT, _Traits>  &
# 576
ends(basic_ostream< _CharT, _Traits>  &__os) 
# 577
{ return (__os.put(_CharT())); } 
# 584
template< class _CharT, class _Traits> inline basic_ostream< _CharT, _Traits>  &
# 586
flush(basic_ostream< _CharT, _Traits>  &__os) 
# 587
{ return (__os.flush()); } 
# 600
template< class _CharT, class _Traits, class _Tp> inline basic_ostream< _CharT, _Traits>  &
# 602
operator<<(basic_ostream< _CharT, _Traits>  &&__os, const _Tp &__x) 
# 603
{ 
# 604
__os << __x; 
# 605
return __os; 
# 606
} 
# 610
}
# 41 "/usr/include/c++/4.8/bits/ostream.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 45
template< class _CharT, class _Traits> 
# 47
basic_ostream< _CharT, _Traits> ::sentry::sentry(basic_ostream &__os) : _M_ok(false), _M_os(__os) 
# 49
{ 
# 51
if ((__os.tie()) && (__os.good())) { 
# 52
((__os.tie())->flush()); }  
# 54
if ((__os.good())) { 
# 55
(_M_ok) = true; } else { 
# 57
(__os.setstate(ios_base::failbit)); }  
# 58
} 
# 60
template< class _CharT, class _Traits> 
# 61
template< class _ValueT> basic_ostream< _CharT, _Traits>  &
# 64
basic_ostream< _CharT, _Traits> ::_M_insert(_ValueT __v) 
# 65
{ 
# 66
sentry __cerb(*this); 
# 67
if (__cerb) 
# 68
{ 
# 69
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 70
try 
# 71
{ 
# 72
const __num_put_type &__np = __check_facet((this->_M_num_put)); 
# 73
if (((__np.put(*this, *this, (this->fill()), __v)).failed())) { 
# 74
(__err |= ::std::ios_base::badbit); }  
# 75
} 
# 76
catch (::__cxxabiv1::__forced_unwind &) 
# 77
{ 
# 78
(this->_M_setstate(ios_base::badbit)); 
# 79
throw; 
# 80
} 
# 81
catch (...) 
# 82
{ (this->_M_setstate(ios_base::badbit)); }  
# 83
if (__err) { 
# 84
(this->setstate(__err)); }  
# 85
}  
# 86
return *this; 
# 87
} 
# 89
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 92
basic_ostream< _CharT, _Traits> ::operator<<(short __n) 
# 93
{ 
# 96
const ::std::ios_base::fmtflags __fmt = (this->flags()) & ios_base::basefield; 
# 97
if ((__fmt == ::std::ios_base::oct) || (__fmt == ::std::ios_base::hex)) { 
# 98
return _M_insert(static_cast< long>(static_cast< unsigned short>(__n))); } else { 
# 100
return _M_insert(static_cast< long>(__n)); }  
# 101
} 
# 103
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 106
basic_ostream< _CharT, _Traits> ::operator<<(int __n) 
# 107
{ 
# 110
const ::std::ios_base::fmtflags __fmt = (this->flags()) & ios_base::basefield; 
# 111
if ((__fmt == ::std::ios_base::oct) || (__fmt == ::std::ios_base::hex)) { 
# 112
return _M_insert(static_cast< long>(static_cast< unsigned>(__n))); } else { 
# 114
return _M_insert(static_cast< long>(__n)); }  
# 115
} 
# 117
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 120
basic_ostream< _CharT, _Traits> ::operator<<(__streambuf_type *__sbin) 
# 121
{ 
# 122
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 123
sentry __cerb(*this); 
# 124
if (__cerb && __sbin) 
# 125
{ 
# 126
try 
# 127
{ 
# 128
if (!__copy_streambufs(__sbin, (this->rdbuf()))) { 
# 129
(__err |= ::std::ios_base::failbit); }  
# 130
} 
# 131
catch (::__cxxabiv1::__forced_unwind &) 
# 132
{ 
# 133
(this->_M_setstate(ios_base::badbit)); 
# 134
throw; 
# 135
} 
# 136
catch (...) 
# 137
{ (this->_M_setstate(ios_base::failbit)); }  
# 138
} else { 
# 139
if (!__sbin) { 
# 140
(__err |= ::std::ios_base::badbit); }  }  
# 141
if (__err) { 
# 142
(this->setstate(__err)); }  
# 143
return *this; 
# 144
} 
# 146
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 149
basic_ostream< _CharT, _Traits> ::put(char_type __c) 
# 150
{ 
# 157
sentry __cerb(*this); 
# 158
if (__cerb) 
# 159
{ 
# 160
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 161
try 
# 162
{ 
# 163
const int_type __put = ((this->rdbuf())->sputc(__c)); 
# 164
if (traits_type::eq_int_type(__put, traits_type::eof())) { 
# 165
(__err |= ::std::ios_base::badbit); }  
# 166
} 
# 167
catch (::__cxxabiv1::__forced_unwind &) 
# 168
{ 
# 169
(this->_M_setstate(ios_base::badbit)); 
# 170
throw; 
# 171
} 
# 172
catch (...) 
# 173
{ (this->_M_setstate(ios_base::badbit)); }  
# 174
if (__err) { 
# 175
(this->setstate(__err)); }  
# 176
}  
# 177
return *this; 
# 178
} 
# 180
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 183
basic_ostream< _CharT, _Traits> ::write(const _CharT *__s, ::std::streamsize __n) 
# 184
{ 
# 192
sentry __cerb(*this); 
# 193
if (__cerb) 
# 194
{ 
# 195
try 
# 196
{ _M_write(__s, __n); } 
# 197
catch (::__cxxabiv1::__forced_unwind &) 
# 198
{ 
# 199
(this->_M_setstate(ios_base::badbit)); 
# 200
throw; 
# 201
} 
# 202
catch (...) 
# 203
{ (this->_M_setstate(ios_base::badbit)); }  
# 204
}  
# 205
return *this; 
# 206
} 
# 208
template< class _CharT, class _Traits> typename basic_ostream< _CharT, _Traits> ::__ostream_type &
# 211
basic_ostream< _CharT, _Traits> ::flush() 
# 212
{ 
# 216
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 217
try 
# 218
{ 
# 219
if ((this->rdbuf()) && (((this->rdbuf())->pubsync()) == (-1))) { 
# 220
(__err |= ::std::ios_base::badbit); }  
# 221
} 
# 222
catch (::__cxxabiv1::__forced_unwind &) 
# 223
{ 
# 224
(this->_M_setstate(ios_base::badbit)); 
# 225
throw; 
# 226
} 
# 227
catch (...) 
# 228
{ (this->_M_setstate(ios_base::badbit)); }  
# 229
if (__err) { 
# 230
(this->setstate(__err)); }  
# 231
return *this; 
# 232
} 
# 234
template< class _CharT, class _Traits> typename basic_ostream< _CharT, _Traits> ::pos_type 
# 237
basic_ostream< _CharT, _Traits> ::tellp() 
# 238
{ 
# 239
pos_type __ret = ((pos_type)(-1)); 
# 240
try 
# 241
{ 
# 242
if (!(this->fail())) { 
# 243
__ret = ((this->rdbuf())->pubseekoff(0, ios_base::cur, ios_base::out)); }  
# 244
} 
# 245
catch (::__cxxabiv1::__forced_unwind &) 
# 246
{ 
# 247
(this->_M_setstate(ios_base::badbit)); 
# 248
throw; 
# 249
} 
# 250
catch (...) 
# 251
{ (this->_M_setstate(ios_base::badbit)); }  
# 252
return __ret; 
# 253
} 
# 255
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 258
basic_ostream< _CharT, _Traits> ::seekp(pos_type __pos) 
# 259
{ 
# 260
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 261
try 
# 262
{ 
# 263
if (!(this->fail())) 
# 264
{ 
# 267
const pos_type __p = ((this->rdbuf())->pubseekpos(__pos, ios_base::out)); 
# 271
if (__p == ((pos_type)((off_type)(-1)))) { 
# 272
(__err |= ::std::ios_base::failbit); }  
# 273
}  
# 274
} 
# 275
catch (::__cxxabiv1::__forced_unwind &) 
# 276
{ 
# 277
(this->_M_setstate(ios_base::badbit)); 
# 278
throw; 
# 279
} 
# 280
catch (...) 
# 281
{ (this->_M_setstate(ios_base::badbit)); }  
# 282
if (__err) { 
# 283
(this->setstate(__err)); }  
# 284
return *this; 
# 285
} 
# 287
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 290
basic_ostream< _CharT, _Traits> ::seekp(off_type __off, ::std::ios_base::seekdir __dir) 
# 291
{ 
# 292
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 293
try 
# 294
{ 
# 295
if (!(this->fail())) 
# 296
{ 
# 299
const pos_type __p = ((this->rdbuf())->pubseekoff(__off, __dir, ios_base::out)); 
# 303
if (__p == ((pos_type)((off_type)(-1)))) { 
# 304
(__err |= ::std::ios_base::failbit); }  
# 305
}  
# 306
} 
# 307
catch (::__cxxabiv1::__forced_unwind &) 
# 308
{ 
# 309
(this->_M_setstate(ios_base::badbit)); 
# 310
throw; 
# 311
} 
# 312
catch (...) 
# 313
{ (this->_M_setstate(ios_base::badbit)); }  
# 314
if (__err) { 
# 315
(this->setstate(__err)); }  
# 316
return *this; 
# 317
} 
# 319
template< class _CharT, class _Traits> basic_ostream< _CharT, _Traits>  &
# 321
operator<<(basic_ostream< _CharT, _Traits>  &__out, const char *__s) 
# 322
{ 
# 323
if (!__s) { 
# 324
(__out.setstate(ios_base::badbit)); } else 
# 326
{ 
# 329
const size_t __clen = char_traits< char> ::length(__s); 
# 330
try 
# 331
{ 
# 332
struct __ptr_guard { 
# 334
_CharT *__p; 
# 335
__ptr_guard(_CharT *__ip) : __p(__ip) { } 
# 336
~__ptr_guard() { delete [] (__p); } 
# 337
_CharT *__get() { return __p; } 
# 338
} __pg(new _CharT [__clen]); 
# 340
_CharT *__ws = __pg.__get(); 
# 341
for (size_t __i = (0); __i < __clen; ++__i) { 
# 342
(__ws[__i]) = (__out.widen(__s[__i])); }  
# 343
__ostream_insert(__out, __ws, __clen); 
# 344
} 
# 345
catch (__cxxabiv1::__forced_unwind &) 
# 346
{ 
# 347
(__out._M_setstate(ios_base::badbit)); 
# 348
throw; 
# 349
} 
# 350
catch (...) 
# 351
{ (__out._M_setstate(ios_base::badbit)); }  
# 352
}  
# 353
return __out; 
# 354
} 
# 359
extern template class basic_ostream< char, char_traits< char> > ;
# 360
extern template basic_ostream< char, char_traits< char> >  &endl(basic_ostream< char, char_traits< char> >  & __os);
# 361
extern template basic_ostream< char, char_traits< char> >  &ends(basic_ostream< char, char_traits< char> >  & __os);
# 362
extern template basic_ostream< char, char_traits< char> >  &flush(basic_ostream< char, char_traits< char> >  & __os);
# 363
extern template basic_ostream< char, char_traits< char> >  &operator<<(basic_ostream< char, char_traits< char> >  & __out, char __c);
# 364
extern template basic_ostream< char, char_traits< char> >  &operator<<(basic_ostream< char, char_traits< char> >  & __out, unsigned char __c);
# 365
extern template basic_ostream< char, char_traits< char> >  &operator<<(basic_ostream< char, char_traits< char> >  & __out, signed char __c);
# 366
extern template basic_ostream< char, char_traits< char> >  &operator<<(basic_ostream< char, char_traits< char> >  & __out, const char * __s);
# 367
extern template basic_ostream< char, char_traits< char> >  &operator<<(basic_ostream< char, char_traits< char> >  & __out, const unsigned char * __s);
# 368
extern template basic_ostream< char, char_traits< char> >  &operator<<(basic_ostream< char, char_traits< char> >  & __out, const signed char * __s);
# 370
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(long __v);
# 371
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(unsigned long __v);
# 372
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(bool __v);
# 374
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(long long __v);
# 375
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(unsigned long long __v);
# 377
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(double __v);
# 378
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(long double __v);
# 379
extern template basic_ostream< char, char_traits< char> > ::__ostream_type &basic_ostream< char, char_traits< char> > ::_M_insert(const void * __v);
# 382
extern template class basic_ostream< wchar_t, char_traits< wchar_t> > ;
# 383
extern template basic_ostream< wchar_t, char_traits< wchar_t> >  &endl(basic_ostream< wchar_t, char_traits< wchar_t> >  & __os);
# 384
extern template basic_ostream< wchar_t, char_traits< wchar_t> >  &ends(basic_ostream< wchar_t, char_traits< wchar_t> >  & __os);
# 385
extern template basic_ostream< wchar_t, char_traits< wchar_t> >  &flush(basic_ostream< wchar_t, char_traits< wchar_t> >  & __os);
# 386
extern template basic_ostream< wchar_t, char_traits< wchar_t> >  &operator<<(basic_ostream< wchar_t, char_traits< wchar_t> >  & __out, wchar_t __c);
# 387
extern template basic_ostream< wchar_t, char_traits< wchar_t> >  &operator<<(basic_ostream< wchar_t, char_traits< wchar_t> >  & __out, char __c);
# 388
extern template basic_ostream< wchar_t, char_traits< wchar_t> >  &operator<<(basic_ostream< wchar_t, char_traits< wchar_t> >  & __out, const wchar_t * __s);
# 389
extern template basic_ostream< wchar_t, char_traits< wchar_t> >  &operator<<(basic_ostream< wchar_t, char_traits< wchar_t> >  & __out, const char * __s);
# 391
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(long __v);
# 392
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(unsigned long __v);
# 393
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(bool __v);
# 395
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(long long __v);
# 396
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(unsigned long long __v);
# 398
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(double __v);
# 399
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(long double __v);
# 400
extern template basic_ostream< wchar_t, char_traits< wchar_t> > ::__ostream_type &basic_ostream< wchar_t, char_traits< wchar_t> > ::_M_insert(const void * __v);
# 405
}
# 41 "/usr/include/c++/4.8/istream" 3
namespace std __attribute((__visibility__("default"))) { 
# 57
template< class _CharT, class _Traits> 
# 58
class basic_istream : virtual public basic_ios< _CharT, _Traits>  { 
# 62
public: typedef _CharT char_type; 
# 63
typedef typename _Traits::int_type int_type; 
# 64
typedef typename _Traits::pos_type pos_type; 
# 65
typedef typename _Traits::off_type off_type; 
# 66
typedef _Traits traits_type; 
# 69
typedef basic_streambuf< _CharT, _Traits>  __streambuf_type; 
# 70
typedef ::std::basic_ios< _CharT, _Traits>  __ios_type; 
# 71
typedef basic_istream __istream_type; 
# 73
typedef num_get< _CharT, istreambuf_iterator< _CharT, _Traits> >  __num_get_type; 
# 74
typedef ctype< _CharT>  __ctype_type; 
# 82
protected: ::std::streamsize _M_gcount; 
# 93
public: explicit basic_istream(__streambuf_type *__sb) : _M_gcount(((::std::streamsize)0)) 
# 95
{ (this->init(__sb)); } 
# 103
virtual ~basic_istream() 
# 104
{ (_M_gcount) = ((::std::streamsize)0); } 
# 107
class sentry; 
# 108
friend class sentry; 
# 120
__istream_type &operator>>(__istream_type &(*__pf)(__istream_type &)) 
# 121
{ return __pf(*this); } 
# 124
__istream_type &operator>>(__ios_type &(*__pf)(__ios_type &)) 
# 125
{ 
# 126
__pf(*this); 
# 127
return *this; 
# 128
} 
# 131
__istream_type &operator>>(::std::ios_base &(*__pf)(::std::ios_base &)) 
# 132
{ 
# 133
__pf(*this); 
# 134
return *this; 
# 135
} 
# 168
__istream_type &operator>>(bool &__n) 
# 169
{ return _M_extract(__n); } 
# 172
__istream_type &operator>>(short & __n); 
# 175
__istream_type &operator>>(unsigned short &__n) 
# 176
{ return _M_extract(__n); } 
# 179
__istream_type &operator>>(int & __n); 
# 182
__istream_type &operator>>(unsigned &__n) 
# 183
{ return _M_extract(__n); } 
# 186
__istream_type &operator>>(long &__n) 
# 187
{ return _M_extract(__n); } 
# 190
__istream_type &operator>>(unsigned long &__n) 
# 191
{ return _M_extract(__n); } 
# 195
__istream_type &operator>>(long long &__n) 
# 196
{ return _M_extract(__n); } 
# 199
__istream_type &operator>>(unsigned long long &__n) 
# 200
{ return _M_extract(__n); } 
# 214
__istream_type &operator>>(float &__f) 
# 215
{ return _M_extract(__f); } 
# 218
__istream_type &operator>>(double &__f) 
# 219
{ return _M_extract(__f); } 
# 222
__istream_type &operator>>(long double &__f) 
# 223
{ return _M_extract(__f); } 
# 235
__istream_type &operator>>(void *&__p) 
# 236
{ return _M_extract(__p); } 
# 259
__istream_type &operator>>(__streambuf_type * __sb); 
# 269
::std::streamsize gcount() const 
# 270
{ return _M_gcount; } 
# 302
int_type get(); 
# 316
__istream_type &get(char_type & __c); 
# 343
__istream_type &get(char_type * __s, ::std::streamsize __n, char_type __delim); 
# 354
__istream_type &get(char_type *__s, ::std::streamsize __n) 
# 355
{ return (this->get(__s, __n, (this->widen('\n')))); } 
# 377
__istream_type &get(__streambuf_type & __sb, char_type __delim); 
# 387
__istream_type &get(__streambuf_type &__sb) 
# 388
{ return (this->get(__sb, (this->widen('\n')))); } 
# 416
__istream_type &getline(char_type * __s, ::std::streamsize __n, char_type __delim); 
# 427
__istream_type &getline(char_type *__s, ::std::streamsize __n) 
# 428
{ return (this->getline(__s, __n, (this->widen('\n')))); } 
# 451
__istream_type &ignore(::std::streamsize __n, int_type __delim); 
# 454
__istream_type &ignore(::std::streamsize __n); 
# 457
__istream_type &ignore(); 
# 468
int_type peek(); 
# 486
__istream_type &read(char_type * __s, ::std::streamsize __n); 
# 505
::std::streamsize readsome(char_type * __s, ::std::streamsize __n); 
# 522
__istream_type &putback(char_type __c); 
# 538
__istream_type &unget(); 
# 556
int sync(); 
# 571
pos_type tellg(); 
# 586
__istream_type &seekg(pos_type); 
# 602
__istream_type &seekg(off_type, ::std::ios_base::seekdir); 
# 606
protected: basic_istream() : _M_gcount(((::std::streamsize)0)) 
# 608
{ (this->init(0)); } 
# 610
template< class _ValueT> __istream_type &_M_extract(_ValueT & __v); 
# 613
}; 
# 619
template<> basic_istream< char, char_traits< char> >  &basic_istream< char, char_traits< char> > ::getline(char_type * __s, streamsize __n, char_type __delim); 
# 624
template<> basic_istream< char, char_traits< char> >  &basic_istream< char, char_traits< char> > ::ignore(streamsize __n); 
# 629
template<> basic_istream< char, char_traits< char> >  &basic_istream< char, char_traits< char> > ::ignore(streamsize __n, int_type __delim); 
# 635
template<> basic_istream< wchar_t, char_traits< wchar_t> >  &basic_istream< wchar_t, char_traits< wchar_t> > ::getline(char_type * __s, streamsize __n, char_type __delim); 
# 640
template<> basic_istream< wchar_t, char_traits< wchar_t> >  &basic_istream< wchar_t, char_traits< wchar_t> > ::ignore(streamsize __n); 
# 645
template<> basic_istream< wchar_t, char_traits< wchar_t> >  &basic_istream< wchar_t, char_traits< wchar_t> > ::ignore(streamsize __n, int_type __delim); 
# 656
template< class _CharT, class _Traits> 
# 657
class basic_istream< _CharT, _Traits> ::sentry { 
# 660
bool _M_ok; 
# 664
public: typedef _Traits traits_type; 
# 665
typedef basic_streambuf< _CharT, _Traits>  __streambuf_type; 
# 666
typedef basic_istream __istream_type; 
# 667
typedef typename ::std::basic_istream< _CharT, _Traits> ::__ctype_type __ctype_type; 
# 668
typedef typename _Traits::int_type __int_type; 
# 693
explicit sentry(basic_istream & __is, bool __noskipws = false); 
# 705
explicit operator bool() const 
# 706
{ return _M_ok; } 
# 707
}; 
# 721
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &operator>>(basic_istream< _CharT, _Traits>  & __in, _CharT & __c); 
# 725
template< class _Traits> inline basic_istream< char, _Traits>  &
# 727
operator>>(basic_istream< char, _Traits>  &__in, unsigned char &__c) 
# 728
{ return __in >> (reinterpret_cast< char &>(__c)); } 
# 730
template< class _Traits> inline basic_istream< char, _Traits>  &
# 732
operator>>(basic_istream< char, _Traits>  &__in, signed char &__c) 
# 733
{ return __in >> (reinterpret_cast< char &>(__c)); } 
# 763
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &operator>>(basic_istream< _CharT, _Traits>  & __in, _CharT * __s); 
# 770
template<> basic_istream< char, char_traits< char> >  &operator>>(basic_istream< char, char_traits< char> >  & __in, char * __s); 
# 772
template< class _Traits> inline basic_istream< char, _Traits>  &
# 774
operator>>(basic_istream< char, _Traits>  &__in, unsigned char *__s) 
# 775
{ return __in >> (reinterpret_cast< char *>(__s)); } 
# 777
template< class _Traits> inline basic_istream< char, _Traits>  &
# 779
operator>>(basic_istream< char, _Traits>  &__in, signed char *__s) 
# 780
{ return __in >> (reinterpret_cast< char *>(__s)); } 
# 794
template< class _CharT, class _Traits> 
# 795
class basic_iostream : public basic_istream< _CharT, _Traits> , public basic_ostream< _CharT, _Traits>  { 
# 803
public: typedef _CharT char_type; 
# 804
typedef typename _Traits::int_type int_type; 
# 805
typedef typename _Traits::pos_type pos_type; 
# 806
typedef typename _Traits::off_type off_type; 
# 807
typedef _Traits traits_type; 
# 810
typedef ::std::basic_istream< _CharT, _Traits>  __istream_type; 
# 811
typedef ::std::basic_ostream< _CharT, _Traits>  __ostream_type; 
# 820
explicit basic_iostream(basic_streambuf< _CharT, _Traits>  *__sb) : __istream_type(__sb), __ostream_type(__sb) 
# 821
{ } 
# 827
virtual ~basic_iostream() { } 
# 830
protected: basic_iostream() : __istream_type(), __ostream_type() 
# 831
{ } 
# 832
}; 
# 854
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &ws(basic_istream< _CharT, _Traits>  & __is); 
# 870
template< class _CharT, class _Traits, class _Tp> inline basic_istream< _CharT, _Traits>  &
# 872
operator>>(basic_istream< _CharT, _Traits>  &&__is, _Tp &__x) 
# 873
{ 
# 874
__is >> __x; 
# 875
return __is; 
# 876
} 
# 880
}
# 41 "/usr/include/c++/4.8/bits/istream.tcc" 3
namespace std __attribute((__visibility__("default"))) { 
# 45
template< class _CharT, class _Traits> 
# 47
basic_istream< _CharT, _Traits> ::sentry::sentry(basic_istream &__in, bool __noskip) : _M_ok(false) 
# 48
{ 
# 49
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 50
if ((__in.good())) 
# 51
{ 
# 52
if ((__in.tie())) { 
# 53
((__in.tie())->flush()); }  
# 54
if ((!__noskip) && ((bool)((__in.flags()) & ios_base::skipws))) 
# 55
{ 
# 56
const __int_type __eof = traits_type::eof(); 
# 57
__streambuf_type *__sb = (__in.rdbuf()); 
# 58
__int_type __c = (__sb->sgetc()); 
# 60
const __ctype_type &__ct = __check_facet((__in._M_ctype)); 
# 61
while ((!traits_type::eq_int_type(__c, __eof)) && (__ct.is(ctype_base::space, traits_type::to_char_type(__c)))) { 
# 64
__c = (__sb->snextc()); }  
# 69
if (traits_type::eq_int_type(__c, __eof)) { 
# 70
(__err |= ::std::ios_base::eofbit); }  
# 71
}  
# 72
}  
# 74
if ((__in.good()) && (__err == ::std::ios_base::goodbit)) { 
# 75
(_M_ok) = true; } else 
# 77
{ 
# 78
(__err |= ::std::ios_base::failbit); 
# 79
(__in.setstate(__err)); 
# 80
}  
# 81
} 
# 83
template< class _CharT, class _Traits> 
# 84
template< class _ValueT> basic_istream< _CharT, _Traits>  &
# 87
basic_istream< _CharT, _Traits> ::_M_extract(_ValueT &__v) 
# 88
{ 
# 89
sentry __cerb(*this, false); 
# 90
if (__cerb) 
# 91
{ 
# 92
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 93
try 
# 94
{ 
# 95
const __num_get_type &__ng = __check_facet((this->_M_num_get)); 
# 96
(__ng.get(*this, 0, *this, __err, __v)); 
# 97
} 
# 98
catch (::__cxxabiv1::__forced_unwind &) 
# 99
{ 
# 100
(this->_M_setstate(ios_base::badbit)); 
# 101
throw; 
# 102
} 
# 103
catch (...) 
# 104
{ (this->_M_setstate(ios_base::badbit)); }  
# 105
if (__err) { 
# 106
(this->setstate(__err)); }  
# 107
}  
# 108
return *this; 
# 109
} 
# 111
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 114
basic_istream< _CharT, _Traits> ::operator>>(short &__n) 
# 115
{ 
# 118
sentry __cerb(*this, false); 
# 119
if (__cerb) 
# 120
{ 
# 121
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 122
try 
# 123
{ 
# 124
long __l; 
# 125
const __num_get_type &__ng = __check_facet((this->_M_num_get)); 
# 126
(__ng.get(*this, 0, *this, __err, __l)); 
# 130
if (__l < ::__gnu_cxx::__numeric_traits_integer< short> ::__min) 
# 131
{ 
# 132
(__err |= ::std::ios_base::failbit); 
# 133
__n = ::__gnu_cxx::__numeric_traits_integer< short> ::__min; 
# 134
} else { 
# 135
if (__l > ::__gnu_cxx::__numeric_traits_integer< short> ::__max) 
# 136
{ 
# 137
(__err |= ::std::ios_base::failbit); 
# 138
__n = ::__gnu_cxx::__numeric_traits_integer< short> ::__max; 
# 139
} else { 
# 141
__n = ((short)__l); }  }  
# 142
} 
# 143
catch (::__cxxabiv1::__forced_unwind &) 
# 144
{ 
# 145
(this->_M_setstate(ios_base::badbit)); 
# 146
throw; 
# 147
} 
# 148
catch (...) 
# 149
{ (this->_M_setstate(ios_base::badbit)); }  
# 150
if (__err) { 
# 151
(this->setstate(__err)); }  
# 152
}  
# 153
return *this; 
# 154
} 
# 156
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 159
basic_istream< _CharT, _Traits> ::operator>>(int &__n) 
# 160
{ 
# 163
sentry __cerb(*this, false); 
# 164
if (__cerb) 
# 165
{ 
# 166
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 167
try 
# 168
{ 
# 169
long __l; 
# 170
const __num_get_type &__ng = __check_facet((this->_M_num_get)); 
# 171
(__ng.get(*this, 0, *this, __err, __l)); 
# 175
if (__l < ::__gnu_cxx::__numeric_traits_integer< int> ::__min) 
# 176
{ 
# 177
(__err |= ::std::ios_base::failbit); 
# 178
__n = ::__gnu_cxx::__numeric_traits_integer< int> ::__min; 
# 179
} else { 
# 180
if (__l > ::__gnu_cxx::__numeric_traits_integer< int> ::__max) 
# 181
{ 
# 182
(__err |= ::std::ios_base::failbit); 
# 183
__n = ::__gnu_cxx::__numeric_traits_integer< int> ::__max; 
# 184
} else { 
# 186
__n = ((int)__l); }  }  
# 187
} 
# 188
catch (::__cxxabiv1::__forced_unwind &) 
# 189
{ 
# 190
(this->_M_setstate(ios_base::badbit)); 
# 191
throw; 
# 192
} 
# 193
catch (...) 
# 194
{ (this->_M_setstate(ios_base::badbit)); }  
# 195
if (__err) { 
# 196
(this->setstate(__err)); }  
# 197
}  
# 198
return *this; 
# 199
} 
# 201
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 204
basic_istream< _CharT, _Traits> ::operator>>(__streambuf_type *__sbout) 
# 205
{ 
# 206
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 207
sentry __cerb(*this, false); 
# 208
if (__cerb && __sbout) 
# 209
{ 
# 210
try 
# 211
{ 
# 212
bool __ineof; 
# 213
if (!__copy_streambufs_eof((this->rdbuf()), __sbout, __ineof)) { 
# 214
(__err |= ::std::ios_base::failbit); }  
# 215
if (__ineof) { 
# 216
(__err |= ::std::ios_base::eofbit); }  
# 217
} 
# 218
catch (::__cxxabiv1::__forced_unwind &) 
# 219
{ 
# 220
(this->_M_setstate(ios_base::failbit)); 
# 221
throw; 
# 222
} 
# 223
catch (...) 
# 224
{ (this->_M_setstate(ios_base::failbit)); }  
# 225
} else { 
# 226
if (!__sbout) { 
# 227
(__err |= ::std::ios_base::failbit); }  }  
# 228
if (__err) { 
# 229
(this->setstate(__err)); }  
# 230
return *this; 
# 231
} 
# 233
template< class _CharT, class _Traits> typename basic_istream< _CharT, _Traits> ::int_type 
# 236
basic_istream< _CharT, _Traits> ::get() 
# 237
{ 
# 238
const int_type __eof = traits_type::eof(); 
# 239
int_type __c = __eof; 
# 240
(_M_gcount) = (0); 
# 241
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 242
sentry __cerb(*this, true); 
# 243
if (__cerb) 
# 244
{ 
# 245
try 
# 246
{ 
# 247
__c = ((this->rdbuf())->sbumpc()); 
# 249
if (!traits_type::eq_int_type(__c, __eof)) { 
# 250
(_M_gcount) = (1); } else { 
# 252
(__err |= ::std::ios_base::eofbit); }  
# 253
} 
# 254
catch (::__cxxabiv1::__forced_unwind &) 
# 255
{ 
# 256
(this->_M_setstate(ios_base::badbit)); 
# 257
throw; 
# 258
} 
# 259
catch (...) 
# 260
{ (this->_M_setstate(ios_base::badbit)); }  
# 261
}  
# 262
if (!(_M_gcount)) { 
# 263
(__err |= ::std::ios_base::failbit); }  
# 264
if (__err) { 
# 265
(this->setstate(__err)); }  
# 266
return __c; 
# 267
} 
# 269
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 272
basic_istream< _CharT, _Traits> ::get(char_type &__c) 
# 273
{ 
# 274
(_M_gcount) = (0); 
# 275
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 276
sentry __cerb(*this, true); 
# 277
if (__cerb) 
# 278
{ 
# 279
try 
# 280
{ 
# 281
const int_type __cb = ((this->rdbuf())->sbumpc()); 
# 283
if (!traits_type::eq_int_type(__cb, traits_type::eof())) 
# 284
{ 
# 285
(_M_gcount) = (1); 
# 286
__c = traits_type::to_char_type(__cb); 
# 287
} else { 
# 289
(__err |= ::std::ios_base::eofbit); }  
# 290
} 
# 291
catch (::__cxxabiv1::__forced_unwind &) 
# 292
{ 
# 293
(this->_M_setstate(ios_base::badbit)); 
# 294
throw; 
# 295
} 
# 296
catch (...) 
# 297
{ (this->_M_setstate(ios_base::badbit)); }  
# 298
}  
# 299
if (!(_M_gcount)) { 
# 300
(__err |= ::std::ios_base::failbit); }  
# 301
if (__err) { 
# 302
(this->setstate(__err)); }  
# 303
return *this; 
# 304
} 
# 306
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 309
basic_istream< _CharT, _Traits> ::get(char_type *__s, ::std::streamsize __n, char_type __delim) 
# 310
{ 
# 311
(_M_gcount) = (0); 
# 312
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 313
sentry __cerb(*this, true); 
# 314
if (__cerb) 
# 315
{ 
# 316
try 
# 317
{ 
# 318
const int_type __idelim = traits_type::to_int_type(__delim); 
# 319
const int_type __eof = traits_type::eof(); 
# 320
__streambuf_type *__sb = (this->rdbuf()); 
# 321
int_type __c = (__sb->sgetc()); 
# 323
while ((((_M_gcount) + (1)) < __n) && (!traits_type::eq_int_type(__c, __eof)) && (!traits_type::eq_int_type(__c, __idelim))) 
# 326
{ 
# 327
(*(__s++)) = traits_type::to_char_type(__c); 
# 328
++(_M_gcount); 
# 329
__c = (__sb->snextc()); 
# 330
}  
# 331
if (traits_type::eq_int_type(__c, __eof)) { 
# 332
(__err |= ::std::ios_base::eofbit); }  
# 333
} 
# 334
catch (::__cxxabiv1::__forced_unwind &) 
# 335
{ 
# 336
(this->_M_setstate(ios_base::badbit)); 
# 337
throw; 
# 338
} 
# 339
catch (...) 
# 340
{ (this->_M_setstate(ios_base::badbit)); }  
# 341
}  
# 344
if (__n > (0)) { 
# 345
(*__s) = char_type(); }  
# 346
if (!(_M_gcount)) { 
# 347
(__err |= ::std::ios_base::failbit); }  
# 348
if (__err) { 
# 349
(this->setstate(__err)); }  
# 350
return *this; 
# 351
} 
# 353
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 356
basic_istream< _CharT, _Traits> ::get(__streambuf_type &__sb, char_type __delim) 
# 357
{ 
# 358
(_M_gcount) = (0); 
# 359
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 360
sentry __cerb(*this, true); 
# 361
if (__cerb) 
# 362
{ 
# 363
try 
# 364
{ 
# 365
const int_type __idelim = traits_type::to_int_type(__delim); 
# 366
const int_type __eof = traits_type::eof(); 
# 367
__streambuf_type *__this_sb = (this->rdbuf()); 
# 368
int_type __c = (__this_sb->sgetc()); 
# 369
char_type __c2 = traits_type::to_char_type(__c); 
# 371
while ((!traits_type::eq_int_type(__c, __eof)) && (!traits_type::eq_int_type(__c, __idelim)) && (!traits_type::eq_int_type((__sb.sputc(__c2)), __eof))) 
# 374
{ 
# 375
++(_M_gcount); 
# 376
__c = (__this_sb->snextc()); 
# 377
__c2 = traits_type::to_char_type(__c); 
# 378
}  
# 379
if (traits_type::eq_int_type(__c, __eof)) { 
# 380
(__err |= ::std::ios_base::eofbit); }  
# 381
} 
# 382
catch (::__cxxabiv1::__forced_unwind &) 
# 383
{ 
# 384
(this->_M_setstate(ios_base::badbit)); 
# 385
throw; 
# 386
} 
# 387
catch (...) 
# 388
{ (this->_M_setstate(ios_base::badbit)); }  
# 389
}  
# 390
if (!(_M_gcount)) { 
# 391
(__err |= ::std::ios_base::failbit); }  
# 392
if (__err) { 
# 393
(this->setstate(__err)); }  
# 394
return *this; 
# 395
} 
# 397
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 400
basic_istream< _CharT, _Traits> ::getline(char_type *__s, ::std::streamsize __n, char_type __delim) 
# 401
{ 
# 402
(_M_gcount) = (0); 
# 403
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 404
sentry __cerb(*this, true); 
# 405
if (__cerb) 
# 406
{ 
# 407
try 
# 408
{ 
# 409
const int_type __idelim = traits_type::to_int_type(__delim); 
# 410
const int_type __eof = traits_type::eof(); 
# 411
__streambuf_type *__sb = (this->rdbuf()); 
# 412
int_type __c = (__sb->sgetc()); 
# 414
while ((((_M_gcount) + (1)) < __n) && (!traits_type::eq_int_type(__c, __eof)) && (!traits_type::eq_int_type(__c, __idelim))) 
# 417
{ 
# 418
(*(__s++)) = traits_type::to_char_type(__c); 
# 419
__c = (__sb->snextc()); 
# 420
++(_M_gcount); 
# 421
}  
# 422
if (traits_type::eq_int_type(__c, __eof)) { 
# 423
(__err |= ::std::ios_base::eofbit); } else 
# 425
{ 
# 426
if (traits_type::eq_int_type(__c, __idelim)) 
# 427
{ 
# 428
(__sb->sbumpc()); 
# 429
++(_M_gcount); 
# 430
} else { 
# 432
(__err |= ::std::ios_base::failbit); }  
# 433
}  
# 434
} 
# 435
catch (::__cxxabiv1::__forced_unwind &) 
# 436
{ 
# 437
(this->_M_setstate(ios_base::badbit)); 
# 438
throw; 
# 439
} 
# 440
catch (...) 
# 441
{ (this->_M_setstate(ios_base::badbit)); }  
# 442
}  
# 445
if (__n > (0)) { 
# 446
(*__s) = char_type(); }  
# 447
if (!(_M_gcount)) { 
# 448
(__err |= ::std::ios_base::failbit); }  
# 449
if (__err) { 
# 450
(this->setstate(__err)); }  
# 451
return *this; 
# 452
} 
# 457
template< class _CharT, class _Traits> typename basic_istream< _CharT, _Traits> ::__istream_type &
# 460
basic_istream< _CharT, _Traits> ::ignore() 
# 461
{ 
# 462
(_M_gcount) = (0); 
# 463
sentry __cerb(*this, true); 
# 464
if (__cerb) 
# 465
{ 
# 466
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 467
try 
# 468
{ 
# 469
const int_type __eof = traits_type::eof(); 
# 470
__streambuf_type *__sb = (this->rdbuf()); 
# 472
if (traits_type::eq_int_type((__sb->sbumpc()), __eof)) { 
# 473
(__err |= ::std::ios_base::eofbit); } else { 
# 475
(_M_gcount) = (1); }  
# 476
} 
# 477
catch (::__cxxabiv1::__forced_unwind &) 
# 478
{ 
# 479
(this->_M_setstate(ios_base::badbit)); 
# 480
throw; 
# 481
} 
# 482
catch (...) 
# 483
{ (this->_M_setstate(ios_base::badbit)); }  
# 484
if (__err) { 
# 485
(this->setstate(__err)); }  
# 486
}  
# 487
return *this; 
# 488
} 
# 490
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 493
basic_istream< _CharT, _Traits> ::ignore(::std::streamsize __n) 
# 494
{ 
# 495
(_M_gcount) = (0); 
# 496
sentry __cerb(*this, true); 
# 497
if (__cerb && (__n > (0))) 
# 498
{ 
# 499
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 500
try 
# 501
{ 
# 502
const int_type __eof = traits_type::eof(); 
# 503
__streambuf_type *__sb = (this->rdbuf()); 
# 504
int_type __c = (__sb->sgetc()); 
# 513
bool __large_ignore = false; 
# 514
while (true) 
# 515
{ 
# 516
while (((_M_gcount) < __n) && (!traits_type::eq_int_type(__c, __eof))) 
# 518
{ 
# 519
++(_M_gcount); 
# 520
__c = (__sb->snextc()); 
# 521
}  
# 522
if ((__n == ::__gnu_cxx::__numeric_traits_integer< long> ::__max) && (!traits_type::eq_int_type(__c, __eof))) 
# 524
{ 
# 525
(_M_gcount) = ::__gnu_cxx::__numeric_traits_integer< long> ::__min; 
# 527
__large_ignore = true; 
# 528
} else { 
# 530
break; }  
# 531
}  
# 533
if (__large_ignore) { 
# 534
(_M_gcount) = ::__gnu_cxx::__numeric_traits_integer< long> ::__max; }  
# 536
if (traits_type::eq_int_type(__c, __eof)) { 
# 537
(__err |= ::std::ios_base::eofbit); }  
# 538
} 
# 539
catch (::__cxxabiv1::__forced_unwind &) 
# 540
{ 
# 541
(this->_M_setstate(ios_base::badbit)); 
# 542
throw; 
# 543
} 
# 544
catch (...) 
# 545
{ (this->_M_setstate(ios_base::badbit)); }  
# 546
if (__err) { 
# 547
(this->setstate(__err)); }  
# 548
}  
# 549
return *this; 
# 550
} 
# 552
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 555
basic_istream< _CharT, _Traits> ::ignore(::std::streamsize __n, int_type __delim) 
# 556
{ 
# 557
(_M_gcount) = (0); 
# 558
sentry __cerb(*this, true); 
# 559
if (__cerb && (__n > (0))) 
# 560
{ 
# 561
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 562
try 
# 563
{ 
# 564
const int_type __eof = traits_type::eof(); 
# 565
__streambuf_type *__sb = (this->rdbuf()); 
# 566
int_type __c = (__sb->sgetc()); 
# 569
bool __large_ignore = false; 
# 570
while (true) 
# 571
{ 
# 572
while (((_M_gcount) < __n) && (!traits_type::eq_int_type(__c, __eof)) && (!traits_type::eq_int_type(__c, __delim))) 
# 575
{ 
# 576
++(_M_gcount); 
# 577
__c = (__sb->snextc()); 
# 578
}  
# 579
if ((__n == ::__gnu_cxx::__numeric_traits_integer< long> ::__max) && (!traits_type::eq_int_type(__c, __eof)) && (!traits_type::eq_int_type(__c, __delim))) 
# 582
{ 
# 583
(_M_gcount) = ::__gnu_cxx::__numeric_traits_integer< long> ::__min; 
# 585
__large_ignore = true; 
# 586
} else { 
# 588
break; }  
# 589
}  
# 591
if (__large_ignore) { 
# 592
(_M_gcount) = ::__gnu_cxx::__numeric_traits_integer< long> ::__max; }  
# 594
if (traits_type::eq_int_type(__c, __eof)) { 
# 595
(__err |= ::std::ios_base::eofbit); } else { 
# 596
if (traits_type::eq_int_type(__c, __delim)) 
# 597
{ 
# 598
if ((_M_gcount) < ::__gnu_cxx::__numeric_traits_integer< long> ::__max) { 
# 600
++(_M_gcount); }  
# 601
(__sb->sbumpc()); 
# 602
}  }  
# 603
} 
# 604
catch (::__cxxabiv1::__forced_unwind &) 
# 605
{ 
# 606
(this->_M_setstate(ios_base::badbit)); 
# 607
throw; 
# 608
} 
# 609
catch (...) 
# 610
{ (this->_M_setstate(ios_base::badbit)); }  
# 611
if (__err) { 
# 612
(this->setstate(__err)); }  
# 613
}  
# 614
return *this; 
# 615
} 
# 617
template< class _CharT, class _Traits> typename basic_istream< _CharT, _Traits> ::int_type 
# 620
basic_istream< _CharT, _Traits> ::peek() 
# 621
{ 
# 622
int_type __c = traits_type::eof(); 
# 623
(_M_gcount) = (0); 
# 624
sentry __cerb(*this, true); 
# 625
if (__cerb) 
# 626
{ 
# 627
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 628
try 
# 629
{ 
# 630
__c = ((this->rdbuf())->sgetc()); 
# 631
if (traits_type::eq_int_type(__c, traits_type::eof())) { 
# 632
(__err |= ::std::ios_base::eofbit); }  
# 633
} 
# 634
catch (::__cxxabiv1::__forced_unwind &) 
# 635
{ 
# 636
(this->_M_setstate(ios_base::badbit)); 
# 637
throw; 
# 638
} 
# 639
catch (...) 
# 640
{ (this->_M_setstate(ios_base::badbit)); }  
# 641
if (__err) { 
# 642
(this->setstate(__err)); }  
# 643
}  
# 644
return __c; 
# 645
} 
# 647
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 650
basic_istream< _CharT, _Traits> ::read(char_type *__s, ::std::streamsize __n) 
# 651
{ 
# 652
(_M_gcount) = (0); 
# 653
sentry __cerb(*this, true); 
# 654
if (__cerb) 
# 655
{ 
# 656
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 657
try 
# 658
{ 
# 659
(_M_gcount) = ((this->rdbuf())->sgetn(__s, __n)); 
# 660
if ((_M_gcount) != __n) { 
# 661
(__err |= ((::std::ios_base::eofbit | ::std::ios_base::failbit))); }  
# 662
} 
# 663
catch (::__cxxabiv1::__forced_unwind &) 
# 664
{ 
# 665
(this->_M_setstate(ios_base::badbit)); 
# 666
throw; 
# 667
} 
# 668
catch (...) 
# 669
{ (this->_M_setstate(ios_base::badbit)); }  
# 670
if (__err) { 
# 671
(this->setstate(__err)); }  
# 672
}  
# 673
return *this; 
# 674
} 
# 676
template< class _CharT, class _Traits> streamsize 
# 679
basic_istream< _CharT, _Traits> ::readsome(char_type *__s, ::std::streamsize __n) 
# 680
{ 
# 681
(_M_gcount) = (0); 
# 682
sentry __cerb(*this, true); 
# 683
if (__cerb) 
# 684
{ 
# 685
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 686
try 
# 687
{ 
# 689
const ::std::streamsize __num = ((this->rdbuf())->in_avail()); 
# 690
if (__num > (0)) { 
# 691
(_M_gcount) = ((this->rdbuf())->sgetn(__s, std::min(__num, __n))); } else { 
# 692
if (__num == (-1)) { 
# 693
(__err |= ::std::ios_base::eofbit); }  }  
# 694
} 
# 695
catch (::__cxxabiv1::__forced_unwind &) 
# 696
{ 
# 697
(this->_M_setstate(ios_base::badbit)); 
# 698
throw; 
# 699
} 
# 700
catch (...) 
# 701
{ (this->_M_setstate(ios_base::badbit)); }  
# 702
if (__err) { 
# 703
(this->setstate(__err)); }  
# 704
}  
# 705
return _M_gcount; 
# 706
} 
# 708
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 711
basic_istream< _CharT, _Traits> ::putback(char_type __c) 
# 712
{ 
# 715
(_M_gcount) = (0); 
# 717
(this->clear((this->rdstate()) & ((~::std::ios_base::eofbit)))); 
# 718
sentry __cerb(*this, true); 
# 719
if (__cerb) 
# 720
{ 
# 721
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 722
try 
# 723
{ 
# 724
const int_type __eof = traits_type::eof(); 
# 725
__streambuf_type *__sb = (this->rdbuf()); 
# 726
if ((!__sb) || traits_type::eq_int_type((__sb->sputbackc(__c)), __eof)) { 
# 728
(__err |= ::std::ios_base::badbit); }  
# 729
} 
# 730
catch (::__cxxabiv1::__forced_unwind &) 
# 731
{ 
# 732
(this->_M_setstate(ios_base::badbit)); 
# 733
throw; 
# 734
} 
# 735
catch (...) 
# 736
{ (this->_M_setstate(ios_base::badbit)); }  
# 737
if (__err) { 
# 738
(this->setstate(__err)); }  
# 739
}  
# 740
return *this; 
# 741
} 
# 743
template< class _CharT, class _Traits> typename basic_istream< _CharT, _Traits> ::__istream_type &
# 746
basic_istream< _CharT, _Traits> ::unget() 
# 747
{ 
# 750
(_M_gcount) = (0); 
# 752
(this->clear((this->rdstate()) & ((~::std::ios_base::eofbit)))); 
# 753
sentry __cerb(*this, true); 
# 754
if (__cerb) 
# 755
{ 
# 756
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 757
try 
# 758
{ 
# 759
const int_type __eof = traits_type::eof(); 
# 760
__streambuf_type *__sb = (this->rdbuf()); 
# 761
if ((!__sb) || traits_type::eq_int_type((__sb->sungetc()), __eof)) { 
# 763
(__err |= ::std::ios_base::badbit); }  
# 764
} 
# 765
catch (::__cxxabiv1::__forced_unwind &) 
# 766
{ 
# 767
(this->_M_setstate(ios_base::badbit)); 
# 768
throw; 
# 769
} 
# 770
catch (...) 
# 771
{ (this->_M_setstate(ios_base::badbit)); }  
# 772
if (__err) { 
# 773
(this->setstate(__err)); }  
# 774
}  
# 775
return *this; 
# 776
} 
# 778
template< class _CharT, class _Traits> int 
# 781
basic_istream< _CharT, _Traits> ::sync() 
# 782
{ 
# 785
int __ret = (-1); 
# 786
sentry __cerb(*this, true); 
# 787
if (__cerb) 
# 788
{ 
# 789
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 790
try 
# 791
{ 
# 792
__streambuf_type *__sb = (this->rdbuf()); 
# 793
if (__sb) 
# 794
{ 
# 795
if ((__sb->pubsync()) == (-1)) { 
# 796
(__err |= ::std::ios_base::badbit); } else { 
# 798
__ret = 0; }  
# 799
}  
# 800
} 
# 801
catch (::__cxxabiv1::__forced_unwind &) 
# 802
{ 
# 803
(this->_M_setstate(ios_base::badbit)); 
# 804
throw; 
# 805
} 
# 806
catch (...) 
# 807
{ (this->_M_setstate(ios_base::badbit)); }  
# 808
if (__err) { 
# 809
(this->setstate(__err)); }  
# 810
}  
# 811
return __ret; 
# 812
} 
# 814
template< class _CharT, class _Traits> typename basic_istream< _CharT, _Traits> ::pos_type 
# 817
basic_istream< _CharT, _Traits> ::tellg() 
# 818
{ 
# 821
pos_type __ret = ((pos_type)(-1)); 
# 822
sentry __cerb(*this, true); 
# 823
if (__cerb) 
# 824
{ 
# 825
try 
# 826
{ 
# 827
if (!(this->fail())) { 
# 828
__ret = ((this->rdbuf())->pubseekoff(0, ios_base::cur, ios_base::in)); }  
# 830
} 
# 831
catch (::__cxxabiv1::__forced_unwind &) 
# 832
{ 
# 833
(this->_M_setstate(ios_base::badbit)); 
# 834
throw; 
# 835
} 
# 836
catch (...) 
# 837
{ (this->_M_setstate(ios_base::badbit)); }  
# 838
}  
# 839
return __ret; 
# 840
} 
# 842
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 845
basic_istream< _CharT, _Traits> ::seekg(pos_type __pos) 
# 846
{ 
# 850
(this->clear((this->rdstate()) & ((~::std::ios_base::eofbit)))); 
# 851
sentry __cerb(*this, true); 
# 852
if (__cerb) 
# 853
{ 
# 854
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 855
try 
# 856
{ 
# 857
if (!(this->fail())) 
# 858
{ 
# 860
const pos_type __p = ((this->rdbuf())->pubseekpos(__pos, ios_base::in)); 
# 864
if (__p == ((pos_type)((off_type)(-1)))) { 
# 865
(__err |= ::std::ios_base::failbit); }  
# 866
}  
# 867
} 
# 868
catch (::__cxxabiv1::__forced_unwind &) 
# 869
{ 
# 870
(this->_M_setstate(ios_base::badbit)); 
# 871
throw; 
# 872
} 
# 873
catch (...) 
# 874
{ (this->_M_setstate(ios_base::badbit)); }  
# 875
if (__err) { 
# 876
(this->setstate(__err)); }  
# 877
}  
# 878
return *this; 
# 879
} 
# 881
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 884
basic_istream< _CharT, _Traits> ::seekg(off_type __off, ::std::ios_base::seekdir __dir) 
# 885
{ 
# 889
(this->clear((this->rdstate()) & ((~::std::ios_base::eofbit)))); 
# 890
sentry __cerb(*this, true); 
# 891
if (__cerb) 
# 892
{ 
# 893
::std::ios_base::iostate __err = ::std::ios_base::goodbit; 
# 894
try 
# 895
{ 
# 896
if (!(this->fail())) 
# 897
{ 
# 899
const pos_type __p = ((this->rdbuf())->pubseekoff(__off, __dir, ios_base::in)); 
# 903
if (__p == ((pos_type)((off_type)(-1)))) { 
# 904
(__err |= ::std::ios_base::failbit); }  
# 905
}  
# 906
} 
# 907
catch (::__cxxabiv1::__forced_unwind &) 
# 908
{ 
# 909
(this->_M_setstate(ios_base::badbit)); 
# 910
throw; 
# 911
} 
# 912
catch (...) 
# 913
{ (this->_M_setstate(ios_base::badbit)); }  
# 914
if (__err) { 
# 915
(this->setstate(__err)); }  
# 916
}  
# 917
return *this; 
# 918
} 
# 921
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 923
operator>>(basic_istream< _CharT, _Traits>  &__in, _CharT &__c) 
# 924
{ 
# 925
typedef basic_istream< _CharT, _Traits>  __istream_type; 
# 926
typedef typename basic_istream< _CharT, _Traits> ::int_type __int_type; 
# 928
typename basic_istream< _CharT, _Traits> ::sentry __cerb(__in, false); 
# 929
if (__cerb) 
# 930
{ 
# 931
ios_base::iostate __err = ios_base::goodbit; 
# 932
try 
# 933
{ 
# 934
const __int_type __cb = ((__in.rdbuf())->sbumpc()); 
# 935
if (!_Traits::eq_int_type(__cb, _Traits::eof())) { 
# 936
__c = _Traits::to_char_type(__cb); } else { 
# 938
(__err |= ((ios_base::eofbit | ios_base::failbit))); }  
# 939
} 
# 940
catch (__cxxabiv1::__forced_unwind &) 
# 941
{ 
# 942
(__in._M_setstate(ios_base::badbit)); 
# 943
throw; 
# 944
} 
# 945
catch (...) 
# 946
{ (__in._M_setstate(ios_base::badbit)); }  
# 947
if (__err) { 
# 948
(__in.setstate(__err)); }  
# 949
}  
# 950
return __in; 
# 951
} 
# 953
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 955
operator>>(basic_istream< _CharT, _Traits>  &__in, _CharT *__s) 
# 956
{ 
# 957
typedef basic_istream< _CharT, _Traits>  __istream_type; 
# 958
typedef basic_streambuf< _CharT, _Traits>  __streambuf_type; 
# 959
typedef typename _Traits::int_type int_type; 
# 960
typedef _CharT char_type; 
# 961
typedef ctype< _CharT>  __ctype_type; 
# 963
streamsize __extracted = (0); 
# 964
ios_base::iostate __err = ios_base::goodbit; 
# 965
typename basic_istream< _CharT, _Traits> ::sentry __cerb(__in, false); 
# 966
if (__cerb) 
# 967
{ 
# 968
try 
# 969
{ 
# 971
streamsize __num = (__in.width()); 
# 972
if (__num <= (0)) { 
# 973
__num = __gnu_cxx::__numeric_traits_integer< long> ::__max; }  
# 975
const __ctype_type &__ct = use_facet< ctype< _CharT> > ((__in.getloc())); 
# 977
const int_type __eof = _Traits::eof(); 
# 978
__streambuf_type *__sb = (__in.rdbuf()); 
# 979
int_type __c = (__sb->sgetc()); 
# 981
while ((__extracted < (__num - (1))) && (!_Traits::eq_int_type(__c, __eof)) && (!(__ct.is(ctype_base::space, _Traits::to_char_type(__c))))) 
# 985
{ 
# 986
(*(__s++)) = _Traits::to_char_type(__c); 
# 987
++__extracted; 
# 988
__c = (__sb->snextc()); 
# 989
}  
# 990
if (_Traits::eq_int_type(__c, __eof)) { 
# 991
(__err |= ios_base::eofbit); }  
# 995
(*__s) = char_type(); 
# 996
(__in.width(0)); 
# 997
} 
# 998
catch (__cxxabiv1::__forced_unwind &) 
# 999
{ 
# 1000
(__in._M_setstate(ios_base::badbit)); 
# 1001
throw; 
# 1002
} 
# 1003
catch (...) 
# 1004
{ (__in._M_setstate(ios_base::badbit)); }  
# 1005
}  
# 1006
if (!__extracted) { 
# 1007
(__err |= ios_base::failbit); }  
# 1008
if (__err) { 
# 1009
(__in.setstate(__err)); }  
# 1010
return __in; 
# 1011
} 
# 1014
template< class _CharT, class _Traits> basic_istream< _CharT, _Traits>  &
# 1016
ws(basic_istream< _CharT, _Traits>  &__in) 
# 1017
{ 
# 1018
typedef basic_istream< _CharT, _Traits>  __istream_type; 
# 1019
typedef basic_streambuf< _CharT, _Traits>  __streambuf_type; 
# 1020
typedef typename basic_istream< _CharT, _Traits> ::int_type __int_type; 
# 1021
typedef ctype< _CharT>  __ctype_type; 
# 1023
const __ctype_type &__ct = use_facet< ctype< _CharT> > ((__in.getloc())); 
# 1024
const __int_type __eof = _Traits::eof(); 
# 1025
__streambuf_type *__sb = (__in.rdbuf()); 
# 1026
__int_type __c = (__sb->sgetc()); 
# 1028
while ((!_Traits::eq_int_type(__c, __eof)) && (__ct.is(ctype_base::space, _Traits::to_char_type(__c)))) { 
# 1030
__c = (__sb->snextc()); }  
# 1032
if (_Traits::eq_int_type(__c, __eof)) { 
# 1033
(__in.setstate(ios_base::eofbit)); }  
# 1034
return __in; 
# 1035
} 
# 1040
extern template class basic_istream< char, char_traits< char> > ;
# 1041
extern template basic_istream< char, char_traits< char> >  &ws(basic_istream< char, char_traits< char> >  & __is);
# 1042
extern template basic_istream< char, char_traits< char> >  &operator>>(basic_istream< char, char_traits< char> >  & __in, char & __c);
# 1043
extern template basic_istream< char, char_traits< char> >  &operator>>(basic_istream< char, char_traits< char> >  &, char *);
# 1044
extern template basic_istream< char, char_traits< char> >  &operator>>(basic_istream< char, char_traits< char> >  & __in, unsigned char & __c);
# 1045
extern template basic_istream< char, char_traits< char> >  &operator>>(basic_istream< char, char_traits< char> >  & __in, signed char & __c);
# 1046
extern template basic_istream< char, char_traits< char> >  &operator>>(basic_istream< char, char_traits< char> >  & __in, unsigned char * __s);
# 1047
extern template basic_istream< char, char_traits< char> >  &operator>>(basic_istream< char, char_traits< char> >  & __in, signed char * __s);
# 1049
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(unsigned short & __v);
# 1050
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(unsigned & __v);
# 1051
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(long & __v);
# 1052
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(unsigned long & __v);
# 1053
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(bool & __v);
# 1055
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(long long & __v);
# 1056
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(unsigned long long & __v);
# 1058
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(float & __v);
# 1059
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(double & __v);
# 1060
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(long double & __v);
# 1061
extern template basic_istream< char, char_traits< char> > ::__istream_type &basic_istream< char, char_traits< char> > ::_M_extract(void *& __v);
# 1063
extern template class basic_iostream< char, char_traits< char> > ;
# 1066
extern template class basic_istream< wchar_t, char_traits< wchar_t> > ;
# 1067
extern template basic_istream< wchar_t, char_traits< wchar_t> >  &ws(basic_istream< wchar_t, char_traits< wchar_t> >  & __is);
# 1068
extern template basic_istream< wchar_t, char_traits< wchar_t> >  &operator>>(basic_istream< wchar_t, char_traits< wchar_t> >  & __in, wchar_t & __c);
# 1069
extern template basic_istream< wchar_t, char_traits< wchar_t> >  &operator>>(basic_istream< wchar_t, char_traits< wchar_t> >  &, wchar_t *);
# 1071
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(unsigned short & __v);
# 1072
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(unsigned & __v);
# 1073
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(long & __v);
# 1074
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(unsigned long & __v);
# 1075
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(bool & __v);
# 1077
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(long long & __v);
# 1078
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(unsigned long long & __v);
# 1080
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(float & __v);
# 1081
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(double & __v);
# 1082
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(long double & __v);
# 1083
extern template basic_istream< wchar_t, char_traits< wchar_t> > ::__istream_type &basic_istream< wchar_t, char_traits< wchar_t> > ::_M_extract(void *& __v);
# 1085
extern template class basic_iostream< wchar_t, char_traits< wchar_t> > ;
# 1090
}
# 42 "/usr/include/c++/4.8/iostream" 3
namespace std __attribute((__visibility__("default"))) { 
# 60
extern istream cin; 
# 61
extern ostream cout; 
# 62
extern ostream cerr; 
# 63
extern ostream clog; 
# 66
extern wistream wcin; 
# 67
extern wostream wcout; 
# 68
extern wostream wcerr; 
# 69
extern wostream wclog; 
# 74
static ios_base::Init __ioinit; 
# 77
}
# 29 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/math.h"
namespace math { 
# 31
template< class T> struct constants; 
# 35
template<> struct constants< float>  { 
# 37
static float one() { return (1.0F); } 
# 38
static float zero() { return (0.0F); } 
# 39
static float pi() { return (3.141592741F); } 
# 40
static float e() { return (2.718281746F); } 
# 41
static float sqrtHalf() { return (0.7071067691F); } 
# 42
static float sqrtTwo() { return (1.414213538F); } 
# 43
static float epsilon() { return (9.999999939e-09F); } 
# 44
}; 
# 47
template<> struct constants< double>  { 
# 49
static double one() { return (1.0); } 
# 50
static double zero() { return (0.0); } 
# 51
static double pi() { return (3.141592653589793116); } 
# 52
static double e() { return (2.718281828459045091); } 
# 53
static double sqrtHalf() { return (0.7071067811865475727); } 
# 54
static double sqrtTwo() { return (1.414213562373095145); } 
# 55
static double epsilon() { return (9.999999999999999395e-12); } 
# 56
}; 
# 59
template<> struct constants< long double>  { 
# 61
static long double one() { return (1.0L); } 
# 62
static long double zero() { return (0.0L); } 
# 63
static long double pi() { return (3.141592653589793116L); } 
# 64
static long double e() { return (2.718281828459045091L); } 
# 65
static long double sqrtHalf() { return (0.7071067811865475727L); } 
# 66
static long double sqrtTwo() { return (1.414213562373095145L); } 
# 67
static long double epsilon() { return (1.00000000000000003e-13L); } 
# 68
}; 
# 104
inline float min(float a, float b) 
# 105
{ 
# 106
return fminf(a, b); 
# 107
} 
# 108
inline float max(float a, float b) 
# 109
{ 
# 110
return fmaxf(a, b); 
# 111
} 
# 112
inline float abs(float a) 
# 113
{ 
# 114
return fabsf(a); 
# 115
} 
# 117
inline float exp(float a) 
# 118
{ 
# 119
return expf(a); 
# 120
} 
# 121
inline float frexp(float a, int *b) 
# 122
{ 
# 123
return frexpf(a, b); 
# 124
} 
# 125
inline float ldexp(float a, int b) 
# 126
{ 
# 127
return ldexpf(a, b); 
# 128
} 
# 129
inline float log(float a) 
# 130
{ 
# 131
return logf(a); 
# 132
} 
# 133
inline float log10(float a) 
# 134
{ 
# 135
return log10f(a); 
# 136
} 
# 137
inline float modf(float a, float *b) 
# 138
{ 
# 139
return modff(a, b); 
# 140
} 
# 142
inline float cos(float a) 
# 143
{ 
# 144
return cosf(a); 
# 145
} 
# 146
inline float sin(float a) 
# 147
{ 
# 148
return sinf(a); 
# 149
} 
# 150
inline float tan(float a) 
# 151
{ 
# 152
return tanf(a); 
# 153
} 
# 154
inline float acos(float a) 
# 155
{ 
# 156
return acosf(a); 
# 157
} 
# 158
inline float asin(float a) 
# 159
{ 
# 160
return asinf(a); 
# 161
} 
# 162
inline float atan(float a) 
# 163
{ 
# 164
return atanf(a); 
# 165
} 
# 166
inline float atan2(float a) 
# 167
{ 
# 168
return expf(a); 
# 169
} 
# 170
inline float cosh(float a) 
# 171
{ 
# 172
return coshf(a); 
# 173
} 
# 174
inline float sinh(float a) 
# 175
{ 
# 176
return sinhf(a); 
# 177
} 
# 178
inline float tanh(float a) 
# 179
{ 
# 180
return expf(a); 
# 181
} 
# 183
inline float pow(float a, float b) 
# 184
{ 
# 185
return powf(a, b); 
# 186
} 
# 187
inline float sqrt(float a) 
# 188
{ 
# 189
return sqrtf(a); 
# 190
} 
# 192
inline float floor(float a) 
# 193
{ 
# 194
return floorf(a); 
# 195
} 
# 196
inline float ceil(float a) 
# 197
{ 
# 198
return ceilf(a); 
# 199
} 
# 201
inline float fmod(float a, float b) 
# 202
{ 
# 203
return fmodf(a, b); 
# 204
} 
# 207
template< class T> inline T 
# 208
clamp(T v, T min = constants< T> ::zero(), T max = constants< T> ::one()) 
# 209
{ 
# 210
return static_cast< T>(math::min(math::max(v, min), max)); 
# 211
} 
# 213
inline float saturate(float v) 
# 214
{ 
# 215
return clamp(v, (0.0F), (1.0F)); 
# 216
} 
# 218
inline double saturate(double v) 
# 219
{ 
# 220
return clamp(v, (0.0), (1.0)); 
# 221
} 
# 223
inline long double saturate(long double v) 
# 224
{ 
# 225
return clamp(v, (0.0L), (1.0L)); 
# 226
} 
# 228
inline float rcp(float v) 
# 229
{ 
# 230
return (1.0F) / v; 
# 231
} 
# 233
inline double rcp(double v) 
# 234
{ 
# 235
return (1.0) / v; 
# 236
} 
# 238
inline long double rcp(long double v) 
# 239
{ 
# 240
return (1.0L) / v; 
# 241
} 
# 243
inline float frac(float v) 
# 244
{ 
# 245
return v - floor(v); 
# 246
} 
# 248
inline double frac(double v) 
# 249
{ 
# 250
return v - (floor(v)); 
# 251
} 
# 253
inline long double frac(long double v) 
# 254
{ 
# 255
return v - (floor(v)); 
# 256
} 
# 258
inline float half(float v) 
# 259
{ 
# 260
return v * (0.5F); 
# 261
} 
# 263
inline double half(double v) 
# 264
{ 
# 265
return v * (0.5); 
# 266
} 
# 268
inline long double half(long double v) 
# 269
{ 
# 270
return v * (0.5L); 
# 271
} 
# 273
inline float lerp(float a, float b, float t) 
# 274
{ 
# 275
return (((1.0F) - t) * a) + (t * b); 
# 276
} 
# 278
inline double lerp(double a, double b, double t) 
# 279
{ 
# 280
return (((1.0) - t) * a) + (t * b); 
# 281
} 
# 283
inline long double lerp(long double a, long double b, long double t) 
# 284
{ 
# 285
return (((1.0L) - t) * a) + (t * b); 
# 286
} 
# 288
inline float smoothstep(float t) 
# 289
{ 
# 290
return (t * t) * ((3.0F) - ((2.0F) * t)); 
# 291
} 
# 293
inline double smoothstep(double t) 
# 294
{ 
# 295
return (t * t) * ((3.0) - ((2.0) * t)); 
# 296
} 
# 298
inline long double smoothstep(long double t) 
# 299
{ 
# 300
return (t * t) * ((3.0L) - ((2.0L) * t)); 
# 301
} 
# 303
inline float smootherstep(float t) 
# 304
{ 
# 305
return ((t * t) * t) * ((t * ((t * (6.0F)) - (15.0F))) + (10.0F)); 
# 306
} 
# 308
inline double smootherstep(double t) 
# 309
{ 
# 310
return ((t * t) * t) * ((t * ((t * (6.0)) - (15.0))) + (10.0)); 
# 311
} 
# 313
inline long double smootherstep(long double t) 
# 314
{ 
# 315
return ((t * t) * t) * ((t * ((t * (6.0L)) - (15.0L))) + (10.0L)); 
# 316
} 
# 317
}
# 28 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/framework/../../dependencies/math/vector.h"
#pragma warning (push)
#pragma warning (disable : 4522)
# 31
namespace math { 
# 33
template< class T, unsigned D> class vector; 
# 36
template< class T> 
# 37
class vector< T, 2U>  { 
# 40
public: static const unsigned dim = 2U; 
# 41
typedef T field_type; 
# 43
T x; 
# 44
T y; 
# 46
vector() = default;
# 48
explicit vector(T a) : x(a), y(a) 
# 50
{ 
# 51
} 
# 53
vector(T x, T y) : x(x), y(y) 
# 55
{ 
# 56
} 
# 58
template< class U> 
# 59
vector(const math::vector< U, 2U>  &v) : x((v.x)), y((v.y)) 
# 61
{ 
# 62
} 
# 64
const math::vector< T, 2U>  &operator=(const math::vector< T, 2U>  &v) 
# 65
{ 
# 66
(x) = (v.x); 
# 67
(y) = (v.y); 
# 68
return *this; 
# 69
} 
# 70
const math::vector< T, 2U>  &operator=(const volatile math::vector< T, 2U>  &v) 
# 71
{ 
# 72
(x) = (v.x); 
# 73
(y) = (v.y); 
# 74
return *this; 
# 75
} 
# 76
const volatile math::vector< T, 2U>  &operator=(const math::vector< T, 2U>  &v) volatile 
# 77
{ 
# 78
(x) = (v.x); 
# 79
(y) = (v.y); 
# 80
return *this; 
# 81
} 
# 83
math::vector< T, 2U>  yx() const 
# 84
{ 
# 85
return math::vector< T, 2U> (y, x); 
# 86
} 
# 88
math::vector< T, 2U>  operator-() const 
# 89
{ 
# 90
return math::vector< T, 2U> (-(x), -(y)); 
# 91
} 
# 93
math::vector< T, 2U>  &operator+=(const math::vector< T, 2U>  &v) 
# 94
{ 
# 95
(x) += (v.x); 
# 96
(y) += (v.y); 
# 97
return *this; 
# 98
} 
# 100
math::vector< T, 2U>  &operator-=(const math::vector< T, 2U>  &v) 
# 101
{ 
# 102
(x) -= (v.x); 
# 103
(y) -= (v.y); 
# 104
return *this; 
# 105
} 
# 107
math::vector< T, 2U>  &operator-=(T a) 
# 108
{ 
# 109
(x) -= a; 
# 110
(y) -= a; 
# 111
return *this; 
# 112
} 
# 114
math::vector< T, 2U>  &operator*=(T a) 
# 115
{ 
# 116
(x) *= a; 
# 117
(y) *= a; 
# 118
return *this; 
# 119
} 
# 121
math::vector< T, 2U>  &operator*=(const math::vector< T, 2U>  &v) 
# 122
{ 
# 123
(x) *= (v.x); 
# 124
(y) *= (v.y); 
# 125
return *this; 
# 126
} 
# 128
math::vector< T, 2U>  &operator/=(T a) 
# 129
{ 
# 130
(x) /= a; 
# 131
(y) /= a; 
# 132
return *this; 
# 133
} 
# 135
friend inline const math::vector< T, 2U>  operator+(const math::vector< T, 2U>  &a, const math::vector< T, 2U>  &b) 
# 136
{ 
# 137
return math::vector< T, 2U> ((a.x) + (b.x), (a.y) + (b.y)); 
# 138
} 
# 140
friend inline const math::vector< T, 2U>  operator+(const math::vector< T, 2U>  &a, T b) 
# 141
{ 
# 142
return math::vector< T, 2U> ((a.x) + b, (a.y) + b); 
# 143
} 
# 145
friend inline const math::vector< T, 2U>  operator+(T a, const math::vector< T, 2U>  &b) 
# 146
{ 
# 147
return math::vector< T, 2U> (a + (b.x), a + (b.y)); 
# 148
} 
# 150
friend inline const math::vector< T, 2U>  operator-(const math::vector< T, 2U>  &a, const math::vector< T, 2U>  &b) 
# 151
{ 
# 152
return math::vector< T, 2U> ((a.x) - (b.x), (a.y) - (b.y)); 
# 153
} 
# 155
friend inline const math::vector< T, 2U>  operator-(const math::vector< T, 2U>  &a, T b) 
# 156
{ 
# 157
return math::vector< T, 2U> ((a.x) - b, (a.y) - b); 
# 158
} 
# 159
friend inline const math::vector< T, 2U>  operator-(T b, const math::vector< T, 2U>  &a) 
# 160
{ 
# 161
return math::vector< T, 2U> (b - (a.x), b - (a.y)); 
# 162
} 
# 164
friend inline const math::vector< T, 2U>  operator*(T a, const math::vector< T, 2U>  &v) 
# 165
{ 
# 166
return math::vector< T, 2U> (a * (v.x), a * (v.y)); 
# 167
} 
# 169
friend inline const math::vector< T, 2U>  operator*(const math::vector< T, 2U>  &v, T a) 
# 170
{ 
# 171
return a * v; 
# 172
} 
# 174
friend inline const math::vector< T, 2U>  operator*(const math::vector< T, 2U>  &a, const math::vector< T, 2U>  &b) 
# 175
{ 
# 176
return math::vector< T, 2U> ((a.x) * (b.x), (a.y) * (b.y)); 
# 177
} 
# 179
friend inline const math::vector< T, 2U>  operator/(const math::vector< T, 2U>  &v, T a) 
# 180
{ 
# 181
return math::vector< T, 2U> ((v.x) / a, (v.y) / a); 
# 182
} 
# 184
friend inline T dot(const math::vector< T, 2U>  &a, const math::vector< T, 2U>  &b) 
# 185
{ 
# 186
return ((a.x) * (b.x)) + ((a.y) * (b.y)); 
# 187
} 
# 189
friend inline math::vector< T, 2U>  abs(const math::vector< T, 2U>  &v) 
# 190
{ 
# 191
return math::vector< T, 2U> (abs(v.x), abs(v.y)); 
# 192
} 
# 194
friend inline math::vector< T, 2U>  floor(const math::vector< T, 2U>  &v) 
# 195
{ 
# 196
return math::vector< T, 2U> (floor(v.x), floor(v.y)); 
# 197
} 
# 199
friend inline math::vector< T, 2U>  ceil(const math::vector< T, 2U>  &v) 
# 200
{ 
# 201
return math::vector< T, 2U> (ceil(v.x), ceil(v.y)); 
# 202
} 
# 204
friend inline math::vector< T, 2U>  max(const math::vector< T, 2U>  &v, T c) 
# 205
{ 
# 206
return math::vector< T, 2U> (max(v.x, c), max(v.y, c)); 
# 207
} 
# 208
friend inline math::vector< T, 2U>  max(T c, const math::vector< T, 2U>  &v) 
# 209
{ 
# 210
return math::vector< T, 2U> (max(v.x, c), max(v.y, c)); 
# 211
} 
# 212
friend inline math::vector< T, 2U>  min(const math::vector< T, 2U>  &v, T c) 
# 213
{ 
# 214
return math::vector< T, 2U> (min(v.x, c), min(v.y, c)); 
# 215
} 
# 216
friend inline math::vector< T, 2U>  min(T c, const math::vector< T, 2U>  &v) 
# 217
{ 
# 218
return math::vector< T, 2U> (min(v.x, c), min(v.y, c)); 
# 219
} 
# 221
friend inline math::vector< T, 2U>  min(const math::vector< T, 2U>  &a, const math::vector< T, 2U>  &b) 
# 222
{ 
# 223
return math::vector< T, 2U> (min(a.x, b.x), min(a.y, b.y)); 
# 224
} 
# 225
friend inline math::vector< T, 2U>  max(const math::vector< T, 2U>  &a, const math::vector< T, 2U>  &b) 
# 226
{ 
# 227
return math::vector< T, 2U> (max(a.x, b.x), max(a.y, b.y)); 
# 228
} 
# 229
friend inline T min(const math::vector< T, 2U>  &a) 
# 230
{ 
# 231
return min(a.x, a.y); 
# 232
} 
# 233
friend inline T max(const math::vector< T, 2U>  &a) 
# 234
{ 
# 235
return max(a.x, a.y); 
# 236
} 
# 238
friend inline T length(const math::vector< T, 2U>  &v) 
# 239
{ 
# 240
return sqrt(dot(v, v)); 
# 241
} 
# 243
friend inline math::vector< T, 2U>  normalize(const math::vector< T, 2U>  &v) 
# 244
{ 
# 245
return v * rcp(length(v)); 
# 246
} 
# 248
friend inline math::vector< T, 2U>  pow(const math::vector< T, 2U>  &v, T exponent) 
# 249
{ 
# 250
return math::vector< T, 2U> (pow(v.x, exponent), pow(v.y, exponent)); 
# 251
} 
# 253
friend inline math::vector< T, 2U>  lerp(const math::vector< T, 2U>  &a, const math::vector< T, 2U>  &b, T t) 
# 254
{ 
# 255
return math::vector< T, 2U> (lerp(a.x, b.x, t), lerp(a.y, b.y, t)); 
# 256
} 
# 258
friend inline math::vector< T, 2U>  rcp(const math::vector< T, 2U>  &v) 
# 259
{ 
# 260
return math::vector< T, 2U> (rcp(v.x), rcp(v.y)); 
# 261
} 
# 263
bool operator==(const math::vector< T, 2U>  &v) const 
# 264
{ 
# 265
return ((x) == (v.x)) && ((y) == (v.y)); 
# 266
} 
# 267
bool operator!=(const math::vector< T, 2U>  &v) const 
# 268
{ 
# 269
return ((x) != (v.x)) || ((y) == (v.y)); 
# 270
} 
# 272
friend inline std::ostream &operator<<(std::ostream &stream, const math::vector< T, 2U>  &v) 
# 273
{ 
# 274
(stream << (v.x)) << ", "; 
# 275
(stream << (v.y)) << ", "; 
# 276
return stream; 
# 277
} 
# 279
friend inline std::istream &operator>>(std::istream &stream, math::vector< T, 2U>  &v) 
# 280
{ 
# 281
char c; 
# 282
stream >> (v.x); 
# 283
(stream >> c); 
# 284
if (c != (',')) { 
# 285
return stream; }  
# 286
stream >> (v.y); 
# 287
(stream >> c); 
# 288
if (c != (',')) { 
# 289
return stream; }  
# 290
return stream; 
# 291
} 
# 292
}; 
# 294
template< class T> 
# 295
class vector< T, 3U>  { 
# 298
public: static const unsigned dim = 3U; 
# 299
typedef T field_type; 
# 301
T x; 
# 302
T y; 
# 303
T z; 
# 305
vector() = default;
# 307
explicit vector(T a) : x(a), y(a), z(a) 
# 309
{ 
# 310
} 
# 312
vector(T x, T y, T z) : x(x), y(y), z(z) 
# 314
{ 
# 315
} 
# 317
template< class U> 
# 318
vector(const math::vector< U, 3U>  &v) : x((v.x)), y((v.y)), z((v.z)) 
# 320
{ 
# 321
} 
# 323
template< class U> 
# 324
vector(const math::vector< U, 2U>  &v, T z) : x((v.x)), y((v.y)), z(z) 
# 326
{ 
# 327
} 
# 329
template< class U> 
# 330
vector(T _x, const math::vector< U, 2U>  &v) : x(_x), y((v.x)), z((v.y)) 
# 332
{ 
# 333
} 
# 335
const math::vector< T, 3U>  &operator=(const math::vector< T, 3U>  &v) 
# 336
{ 
# 337
(x) = (v.x); 
# 338
(y) = (v.y); 
# 339
(z) = (v.z); 
# 340
return *this; 
# 341
} 
# 342
const math::vector< T, 3U>  &operator=(const volatile math::vector< T, 3U>  &v) 
# 343
{ 
# 344
(x) = (v.x); 
# 345
(y) = (v.y); 
# 346
(z) = (v.z); 
# 347
return *this; 
# 348
} 
# 349
const volatile math::vector< T, 3U>  &operator=(const math::vector< T, 3U>  &v) volatile 
# 350
{ 
# 351
(x) = (v.x); 
# 352
(y) = (v.y); 
# 353
(z) = (v.z); 
# 354
return *this; 
# 355
} 
# 357
math::vector< T, 2U>  xy() const 
# 358
{ 
# 359
return math::vector< T, 2U> (x, y); 
# 360
} 
# 361
math::vector< T, 2U>  yx() const 
# 362
{ 
# 363
return math::vector< T, 2U> (y, x); 
# 364
} 
# 365
math::vector< T, 2U>  xz() const 
# 366
{ 
# 367
return math::vector< T, 2U> (x, z); 
# 368
} 
# 369
math::vector< T, 2U>  zx() const 
# 370
{ 
# 371
return math::vector< T, 2U> (z, x); 
# 372
} 
# 373
math::vector< T, 2U>  yz() const 
# 374
{ 
# 375
return math::vector< T, 2U> (y, z); 
# 376
} 
# 377
math::vector< T, 2U>  zy() const 
# 378
{ 
# 379
return math::vector< T, 2U> (z, y); 
# 380
} 
# 381
math::vector< T, 3U>  xzy() const 
# 382
{ 
# 383
return math::vector< T, 3U> (x, y, x); 
# 384
} 
# 385
math::vector< T, 3U>  yxz() const 
# 386
{ 
# 387
return math::vector< T, 3U> (y, x, z); 
# 388
} 
# 389
math::vector< T, 3U>  yzx() const 
# 390
{ 
# 391
return math::vector< T, 3U> (y, z, z); 
# 392
} 
# 393
math::vector< T, 3U>  zxy() const 
# 394
{ 
# 395
return math::vector< T, 3U> (z, x, y); 
# 396
} 
# 397
math::vector< T, 3U>  zyx() const 
# 398
{ 
# 399
return math::vector< T, 3U> (z, y, x); 
# 400
} 
# 402
math::vector< T, 3U>  operator-() const 
# 403
{ 
# 404
return math::vector< T, 3U> (-(x), -(y), -(z)); 
# 405
} 
# 407
math::vector< T, 3U>  &operator+=(const math::vector< T, 3U>  &v) 
# 408
{ 
# 409
(x) += (v.x); 
# 410
(y) += (v.y); 
# 411
(z) += (v.z); 
# 412
return *this; 
# 413
} 
# 415
math::vector< T, 3U>  &operator-=(const math::vector< T, 3U>  &v) 
# 416
{ 
# 417
(x) -= (v.x); 
# 418
(y) -= (v.y); 
# 419
(z) -= (v.z); 
# 420
return *this; 
# 421
} 
# 423
math::vector< T, 3U>  &operator-=(T a) 
# 424
{ 
# 425
(x) -= a; 
# 426
(y) -= a; 
# 427
(z) -= a; 
# 428
return *this; 
# 429
} 
# 431
math::vector< T, 3U>  &operator*=(T a) 
# 432
{ 
# 433
(x) *= a; 
# 434
(y) *= a; 
# 435
(z) *= a; 
# 436
return *this; 
# 437
} 
# 439
math::vector< T, 3U>  &operator/=(T a) 
# 440
{ 
# 441
(x) /= a; 
# 442
(y) /= a; 
# 443
(z) /= a; 
# 444
return *this; 
# 445
} 
# 447
math::vector< T, 3U>  &operator*=(const math::vector< T, 3U>  &v) 
# 448
{ 
# 449
(x) *= (v.x); 
# 450
(y) *= (v.y); 
# 451
(z) *= (v.z); 
# 452
return *this; 
# 453
} 
# 455
T &operator[](size_t pos) 
# 456
{ 
# 457
return *((&(this->x)) + pos); 
# 458
} 
# 460
T operator[](size_t pos) const 
# 461
{ 
# 462
return *((&(this->x)) + pos); 
# 463
} 
# 465
friend inline const math::vector< T, 3U>  operator+(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b) 
# 466
{ 
# 467
return math::vector< T, 3U> ((a.x) + (b.x), (a.y) + (b.y), (a.z) + (b.z)); 
# 468
} 
# 470
friend inline const math::vector< T, 3U>  operator+(const math::vector< T, 3U>  &a, T b) 
# 471
{ 
# 472
return math::vector< T, 3U> ((a.x) + b, (a.y) + b, (a.z) + b); 
# 473
} 
# 475
friend inline const math::vector< T, 3U>  operator+(T a, const math::vector< T, 3U>  &b) 
# 476
{ 
# 477
return math::vector< T, 3U> (a + (b.x), a + (b.y), a + (b.z)); 
# 478
} 
# 480
friend inline const math::vector< T, 3U>  operator-(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b) 
# 481
{ 
# 482
return math::vector< T, 3U> ((a.x) - (b.x), (a.y) - (b.y), (a.z) - (b.z)); 
# 483
} 
# 485
friend inline const math::vector< T, 3U>  operator-(const math::vector< T, 3U>  &a, T b) 
# 486
{ 
# 487
return math::vector< T, 3U> ((a.x) - b, (a.y) - b, (a.z) - b); 
# 488
} 
# 490
friend inline const math::vector< T, 3U>  operator-(T b, const math::vector< T, 3U>  &a) 
# 491
{ 
# 492
return math::vector< T, 3U> (b - (a.x), b - (a.y), b - (a.z)); 
# 493
} 
# 495
friend inline const math::vector< T, 3U>  operator*(T a, const math::vector< T, 3U>  &v) 
# 496
{ 
# 497
return math::vector< T, 3U> (a * (v.x), a * (v.y), a * (v.z)); 
# 498
} 
# 500
friend inline const math::vector< T, 3U>  operator/(const math::vector< T, 3U>  &v, T a) 
# 501
{ 
# 502
return math::vector< T, 3U> ((v.x) / a, (v.y) / a, (v.z) / a); 
# 503
} 
# 505
friend inline const math::vector< T, 3U>  operator*(const math::vector< T, 3U>  &v, T a) 
# 506
{ 
# 507
return a * v; 
# 508
} 
# 510
friend inline const math::vector< T, 3U>  operator*(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b) 
# 511
{ 
# 512
return math::vector< T, 3U> ((a.x) * (b.x), (a.y) * (b.y), (a.z) * (b.z)); 
# 513
} 
# 515
friend inline T dot(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b) 
# 516
{ 
# 517
return (((a.x) * (b.x)) + ((a.y) * (b.y))) + ((a.z) * (b.z)); 
# 518
} 
# 520
friend inline math::vector< T, 3U>  cross(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b) 
# 521
{ 
# 522
return math::vector< T, 3U> (((a.y) * (b.z)) - ((a.z) * (b.y)), ((a.z) * (b.x)) - ((a.x) * (b.z)), ((a.x) * (b.y)) - ((a.y) * (b.x))); 
# 525
} 
# 527
friend inline math::vector< T, 3U>  abs(const math::vector< T, 3U>  &v) 
# 528
{ 
# 529
return math::vector< T, 3U> (abs(v.x), abs(v.y), abs(v.z)); 
# 530
} 
# 532
friend inline math::vector< T, 3U>  floor(const math::vector< T, 3U>  &v) 
# 533
{ 
# 534
return math::vector< T, 3U> (floor(v.x), floor(v.y)); 
# 535
} 
# 537
friend inline math::vector< T, 3U>  ceil(const math::vector< T, 3U>  &v) 
# 538
{ 
# 539
return math::vector< T, 3U> (ceil(v.x), ceil(v.y)); 
# 540
} 
# 542
friend inline math::vector< T, 3U>  max(const math::vector< T, 3U>  &v, T c) 
# 543
{ 
# 544
return math::vector< T, 3U> (max(v.x, c), max(v.y, c), max(v.z, c)); 
# 545
} 
# 546
friend inline math::vector< T, 3U>  max(T c, const math::vector< T, 3U>  &v) 
# 547
{ 
# 548
return math::vector< T, 3U> (max(v.x, c), max(v.y, c), max(v.z, c)); 
# 549
} 
# 550
friend inline math::vector< T, 3U>  min(const math::vector< T, 3U>  &v, T c) 
# 551
{ 
# 552
return math::vector< T, 3U> (min(v.x, c), min(v.y, c), min(v.z, c)); 
# 553
} 
# 554
friend inline math::vector< T, 3U>  min(T c, const math::vector< T, 3U>  &v) 
# 555
{ 
# 556
return math::vector< T, 3U> (min(v.x, c), min(v.y, c), min(v.z, c)); 
# 557
} 
# 559
friend inline math::vector< T, 3U>  min(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b) 
# 560
{ 
# 561
return math::vector< T, 3U> (min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); 
# 562
} 
# 563
friend inline math::vector< T, 3U>  max(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b) 
# 564
{ 
# 565
return math::vector< T, 3U> (max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); 
# 566
} 
# 567
friend inline T min(const math::vector< T, 3U>  &a) 
# 568
{ 
# 569
return min(min(a.x, a.y), a.z); 
# 570
} 
# 571
friend inline T max(const math::vector< T, 3U>  &a) 
# 572
{ 
# 573
return max(max(a.x, a.y), a.z); 
# 574
} 
# 576
friend inline T length(const math::vector< T, 3U>  &v) 
# 577
{ 
# 578
return sqrt(dot(v, v)); 
# 579
} 
# 581
friend inline math::vector< T, 3U>  normalize(const math::vector< T, 3U>  &v) 
# 582
{ 
# 583
return v * rcp(length(v)); 
# 584
} 
# 586
friend inline math::vector< T, 3U>  pow(const math::vector< T, 3U>  &v, T exponent) 
# 587
{ 
# 588
return math::vector< T, 3U> (pow(v.x, exponent), pow(v.y, exponent), pow(v.z, exponent)); 
# 589
} 
# 591
friend inline math::vector< T, 3U>  lerp(const math::vector< T, 3U>  &a, const math::vector< T, 3U>  &b, T t) 
# 592
{ 
# 593
return math::vector< T, 3U> (lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t)); 
# 594
} 
# 596
friend inline math::vector< T, 3U>  rcp(const math::vector< T, 3U>  &v) 
# 597
{ 
# 598
return math::vector< T, 3U> (rcp(v.x), rcp(v.y), rcp(v.z)); 
# 599
} 
# 601
bool operator==(const math::vector< T, 3U>  &v) const 
# 602
{ 
# 603
return ((x) == (v.x)) && ((y) == (v.y)) && ((z) == (v.z)); 
# 604
} 
# 605
bool operator!=(const math::vector< T, 3U>  &v) const 
# 606
{ 
# 607
return (((x) != (v.x)) || ((y) == (v.y))) || ((z) == (v.z)); 
# 608
} 
# 610
friend inline std::ostream &operator<<(std::ostream &stream, const math::vector< T, 3U>  &v) 
# 611
{ 
# 612
(stream << (v.x)) << ", "; 
# 613
(stream << (v.y)) << ", "; 
# 614
(stream << (v.z)) << ", "; 
# 615
return stream; 
# 616
} 
# 618
friend inline std::istream &operator>>(std::istream &stream, math::vector< T, 3U>  &v) 
# 619
{ 
# 620
char c; 
# 621
stream >> (v.x); 
# 622
(stream >> c); 
# 623
if (c != (',')) { 
# 624
return stream; }  
# 625
stream >> (v.y); 
# 626
(stream >> c); 
# 627
if (c != (',')) { 
# 628
return stream; }  
# 629
stream >> (v.z); 
# 630
(stream >> c); 
# 631
if (c != (',')) { 
# 632
return stream; }  
# 633
return stream; 
# 634
} 
# 635
}; 
# 637
template< class T> 
# 638
class vector< T, 4U>  { 
# 641
public: static const unsigned dim = 4U; 
# 642
typedef T field_type; 
# 644
T x; 
# 645
T y; 
# 646
T z; 
# 647
T w; 
# 649
vector() = default;
# 651
explicit vector(T a) : x(a), y(a), z(a), w(a) 
# 653
{ 
# 654
} 
# 656
vector(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) 
# 658
{ 
# 659
} 
# 661
template< class U> 
# 662
vector(const math::vector< U, 4>  &v) : x((v.x)), y((v.y)), z((v.z)), w((v.w)) 
# 664
{ 
# 665
} 
# 667
template< class U> 
# 668
vector(const math::vector< U, 3>  &v, T w) : x((v.x)), y((v.y)), z((v.z)), w(w) 
# 670
{ 
# 671
} 
# 673
template< class U> 
# 674
vector(T x, const math::vector< U, 3>  &v) : x(x), y((v.x)), z((v.y)), w((v.z)) 
# 676
{ 
# 677
} 
# 679
template< class U> 
# 680
vector(T x, T y, const math::vector< U, 2>  &v) : x(x), y(y), z((v.x)), w((v.y)) 
# 682
{ 
# 683
} 
# 685
template< class U> 
# 686
vector(const math::vector< U, 2>  &v2, T z, T w) : x((v2.x)), y((v2.y)), z(z), w(w) 
# 688
{ 
# 689
} 
# 691
template< class U> 
# 692
vector(const math::vector< U, 2>  &v1, const math::vector< U, 2U>  &v2) : x((v1.x)), y((v1.y)), z((v2.x)), w((v2.y)) 
# 694
{ 
# 695
} 
# 697
const math::vector< T, 4U>  &operator=(const math::vector< T, 4U>  &v) 
# 698
{ 
# 699
(x) = (v.x); 
# 700
(y) = (v.y); 
# 701
(z) = (v.z); 
# 702
(w) = (v.w); 
# 703
return *this; 
# 704
} 
# 705
const math::vector< T, 4U>  &operator=(const volatile math::vector< T, 4U>  &v) 
# 706
{ 
# 707
(x) = (v.x); 
# 708
(y) = (v.y); 
# 709
(z) = (v.z); 
# 710
(w) = (v.w); 
# 711
return *this; 
# 712
} 
# 713
const volatile math::vector< T, 4U>  &operator=(const math::vector< T, 4U>  &v) volatile 
# 714
{ 
# 715
(x) = (v.x); 
# 716
(y) = (v.y); 
# 717
(z) = (v.z); 
# 718
(w) = (v.w); 
# 719
return *this; 
# 720
} 
# 722
math::vector< T, 3U>  xyz() const 
# 723
{ 
# 724
return math::vector< T, 3U> (x, y, z); 
# 725
} 
# 726
math::vector< T, 3U>  xyw() const 
# 727
{ 
# 728
return math::vector< T, 3U> (x, y, w); 
# 729
} 
# 730
math::vector< T, 2U>  xy() const 
# 731
{ 
# 732
return math::vector< T, 2U> (x, y); 
# 733
} 
# 735
math::vector< T, 4U>  operator-() const 
# 736
{ 
# 737
return math::vector< T, 4U> (-(x), -(y), -(z), -(w)); 
# 738
} 
# 740
math::vector< T, 4U>  &operator+=(const math::vector< T, 4U>  &v) 
# 741
{ 
# 742
(x) += (v.x); 
# 743
(y) += (v.y); 
# 744
(z) += (v.z); 
# 745
(w) += (v.w); 
# 746
return *this; 
# 747
} 
# 749
math::vector< T, 4U>  &operator-=(const math::vector< T, 4U>  &v) 
# 750
{ 
# 751
(x) -= (v.x); 
# 752
(y) -= (v.y); 
# 753
(z) -= (v.z); 
# 754
(w) -= (v.w); 
# 755
return *this; 
# 756
} 
# 758
math::vector< T, 4U>  &operator-=(T a) 
# 759
{ 
# 760
(x) -= a; 
# 761
(y) -= a; 
# 762
(z) -= a; 
# 763
(w) -= a; 
# 764
return *this; 
# 765
} 
# 767
math::vector< T, 4U>  &operator*=(T a) 
# 768
{ 
# 769
(x) *= a; 
# 770
(y) *= a; 
# 771
(z) *= a; 
# 772
(w) *= a; 
# 773
return *this; 
# 774
} 
# 776
math::vector< T, 4U>  &operator*=(const math::vector< T, 4U>  &v) 
# 777
{ 
# 778
(x) *= (v.x); 
# 779
(y) *= (v.y); 
# 780
(z) *= (v.z); 
# 781
(w) *= (v.w); 
# 782
return *this; 
# 783
} 
# 785
math::vector< T, 4U>  &operator/=(T a) 
# 786
{ 
# 787
(x) /= a; 
# 788
(y) /= a; 
# 789
(z) /= a; 
# 790
(w) /= a; 
# 791
return *this; 
# 792
} 
# 794
friend inline const math::vector< T, 4U>  operator+(const math::vector< T, 4U>  &a, const math::vector< T, 4U>  &b) 
# 795
{ 
# 796
return math::vector< T, 4U> ((a.x) + (b.x), (a.y) + (b.y), (a.z) + (b.z), (a.w) + (b.w)); 
# 797
} 
# 799
friend inline const math::vector< T, 4U>  operator+(const math::vector< T, 4U>  &a, T b) 
# 800
{ 
# 801
return math::vector< T, 4U> ((a.x) + b, (a.y) + b, (a.z) + b, (a.w) + b); 
# 802
} 
# 804
friend inline const math::vector< T, 4U>  operator+(T a, const math::vector< T, 4U>  &b) 
# 805
{ 
# 806
return math::vector< T, 4U> (a + (b.x), a + (b.y), a + (b.z), a + (b.w)); 
# 807
} 
# 809
friend inline const math::vector< T, 4U>  operator-(const math::vector< T, 4U>  &a, const math::vector< T, 4U>  &b) 
# 810
{ 
# 811
return math::vector< T, 4U> ((a.x) - (b.x), (a.y) - (b.y), (a.z) - (b.z), (a.w) - (b.w)); 
# 812
} 
# 814
friend inline const math::vector< T, 4U>  operator-(const math::vector< T, 4U>  &a, T b) 
# 815
{ 
# 816
return math::vector< T, 4U> ((a.x) - b, (a.y) - b, (a.z) - b, (a.w) - b); 
# 817
} 
# 819
friend inline const math::vector< T, 4U>  operator-(T b, const math::vector< T, 4U>  &a) 
# 820
{ 
# 821
return math::vector< T, 4U> (b - (a.x), b - (a.y), b - (a.z), b - (a.w)); 
# 822
} 
# 824
friend inline const math::vector< T, 4U>  operator*(T a, const math::vector< T, 4U>  &v) 
# 825
{ 
# 826
return math::vector< T, 4U> (a * (v.x), a * (v.y), a * (v.z), a * (v.w)); 
# 827
} 
# 829
friend inline const math::vector< T, 4U>  operator/(const math::vector< T, 4U>  &v, T a) 
# 830
{ 
# 831
return math::vector< T, 4U> ((v.x) / a, (v.y) / a, (v.z) / a, (v.w) / a); 
# 832
} 
# 834
friend inline const math::vector< T, 4U>  operator*(const math::vector< T, 4U>  &v, T a) 
# 835
{ 
# 836
return a * v; 
# 837
} 
# 839
friend inline const math::vector< T, 4U>  operator*(const math::vector< T, 4U>  &a, const math::vector< T, 4U>  &b) 
# 840
{ 
# 841
return math::vector< T, 4U> ((a.x) * (b.x), (a.y) * (b.y), (a.z) * (b.z), (a.w) * (b.w)); 
# 842
} 
# 844
friend inline T dot(const math::vector< T, 4U>  &a, const math::vector< T, 4U>  &b) 
# 845
{ 
# 846
return ((((a.x) * (b.x)) + ((a.y) * (b.y))) + ((a.z) * (b.z))) + ((a.w) * (b.w)); 
# 847
} 
# 849
friend inline math::vector< T, 4U>  abs(const math::vector< T, 4U>  &v) 
# 850
{ 
# 851
return math::vector< T, 4U> (abs(v.x), abs(v.y), abs(v.z), abs(v.w)); 
# 852
} 
# 854
friend inline math::vector< T, 4U>  floor(const math::vector< T, 4U>  &v) 
# 855
{ 
# 856
return math::vector< T, 4U> (floor(v.x), floor(v.y), floor(v.z), floor(v.w)); 
# 857
} 
# 859
friend inline math::vector< T, 4U>  ceil(const math::vector< T, 4U>  &v) 
# 860
{ 
# 861
return math::vector< T, 4U> (ceil(v.x), ceil(v.y), ceil(v.z), ceil(v.w)); 
# 862
} 
# 864
friend inline math::vector< T, 4U>  max(const math::vector< T, 4U>  &v, T c) 
# 865
{ 
# 866
return math::vector< T, 4U> (max(v.x, c), max(v.y, c), max(v.z, c), max(v.w, c)); 
# 867
} 
# 868
friend inline math::vector< T, 4U>  max(T c, const math::vector< T, 4U>  &v) 
# 869
{ 
# 870
return math::vector< T, 4U> (max(v.x, c), max(v.y, c), max(v.z, c), max(v.w, c)); 
# 871
} 
# 872
friend inline math::vector< T, 4U>  min(const math::vector< T, 4U>  &v, T c) 
# 873
{ 
# 874
return math::vector< T, 4U> (min(v.x, c), min(v.y, c), min(v.z, c), min(v.w, c)); 
# 875
} 
# 876
friend inline math::vector< T, 4U>  min(T c, const math::vector< T, 4U>  &v) 
# 877
{ 
# 878
return math::vector< T, 4U> (min(v.x, c), min(v.y, c), min(v.z, c), min(v.w, c)); 
# 879
} 
# 881
friend inline math::vector< T, 4U>  min(const math::vector< T, 4U>  &a, const math::vector< T, 4U>  &b) 
# 882
{ 
# 883
return math::vector< T, 4U> (min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); 
# 884
} 
# 885
friend inline math::vector< T, 4U>  max(const math::vector< T, 4U>  &a, const math::vector< T, 4U>  &b) 
# 886
{ 
# 887
return math::vector< T, 4U> (max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); 
# 888
} 
# 890
friend inline T min(const math::vector< T, 4U>  &a) 
# 891
{ 
# 892
return min(min(a.x, a.y), min(a.z, a.w)); 
# 893
} 
# 894
friend inline T max(const math::vector< T, 4U>  &a) 
# 895
{ 
# 896
return max(max(a.x, a.y), max(a.z, a.w)); 
# 897
} 
# 899
friend inline T length(const math::vector< T, 4U>  &v) 
# 900
{ 
# 901
return sqrt(dot(v, v)); 
# 902
} 
# 904
friend inline math::vector< T, 4U>  normalize(const math::vector< T, 4U>  &v) 
# 905
{ 
# 906
return v * rcp(length(v)); 
# 907
} 
# 909
friend inline math::vector< T, 4U>  pow(const math::vector< T, 4U>  &v, T exponent) 
# 910
{ 
# 911
return math::vector< T, 4U> (pow(v.x, exponent), pow(v.y, exponent), pow(v.z, exponent), pow(v.w, exponent)); 
# 912
} 
# 914
friend inline math::vector< T, 4U>  rcp(const math::vector< T, 4U>  &v) 
# 915
{ 
# 916
return math::vector< T, 4U> (rcp(v.x), rcp(v.y), rcp(v.z), rcp(v.w)); 
# 917
} 
# 919
friend inline math::vector< T, 4U>  lerp(const math::vector< T, 4U>  &a, const math::vector< T, 4U>  &b, T t) 
# 920
{ 
# 921
return math::vector< T, 4U> (lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t), lerp(a.w, b.w, t)); 
# 922
} 
# 924
bool operator==(const math::vector< T, 4U>  &v) const 
# 925
{ 
# 926
return ((x) == (v.x)) && ((y) == (v.y)) && ((z) == (v.z)) && ((w) == (v.w)); 
# 927
} 
# 928
bool operator!=(const math::vector< T, 4U>  &v) const 
# 929
{ 
# 930
return ((((x) != (v.x)) || ((y) == (v.y))) || ((z) == (v.z))) || ((w) == (v.w)); 
# 931
} 
# 933
friend inline std::ostream &operator<<(std::ostream &stream, const math::vector< T, 4U>  &v) 
# 934
{ 
# 935
(stream << (v.x)) << ", "; 
# 936
(stream << (v.y)) << ", "; 
# 937
(stream << (v.z)) << ", "; 
# 938
(stream << (v.w)) << ", "; 
# 939
return stream; 
# 940
} 
# 942
friend inline std::istream &operator>>(std::istream &stream, math::vector< T, 4U>  &v) 
# 943
{ 
# 944
char c; 
# 945
stream >> (v.x); 
# 946
(stream >> c); 
# 947
if (c != (',')) { 
# 948
return stream; }  
# 949
stream >> (v.y); 
# 950
(stream >> c); 
# 951
if (c != (',')) { 
# 952
return stream; }  
# 953
stream >> (v.z); 
# 954
(stream >> c); 
# 955
if (c != (',')) { 
# 956
return stream; }  
# 957
stream >> (v.w); 
# 958
(stream >> c); 
# 959
if (c != (',')) { 
# 960
return stream; }  
# 961
return stream; 
# 962
} 
# 963
}; 
# 965
template< class T, unsigned D> inline vector< T, D>  
# 966
clamp(const vector< T, D>  &v, T lower, T upper) 
# 967
{ 
# 968
return min(max(v, lower), upper); 
# 969
} 
# 971
template< class T, unsigned D> inline T 
# 972
length2(const vector< T, D>  &v) 
# 973
{ 
# 974
return dot(v, v); 
# 975
} 
# 989
typedef vector< float, 2U>  float2; 
# 990
typedef vector< float, 3U>  float3; 
# 991
typedef vector< float, 4U>  float4; 
# 993
typedef vector< short, 2U>  short2; 
# 994
typedef vector< short, 3U>  short3; 
# 995
typedef vector< short, 4U>  short4; 
# 997
typedef vector< unsigned short, 2U>  ushort2; 
# 998
typedef vector< unsigned short, 3U>  ushort3; 
# 999
typedef vector< unsigned short, 4U>  ushort4; 
# 1001
typedef vector< int, 2U>  int2; 
# 1002
typedef vector< int, 3U>  int3; 
# 1003
typedef vector< int, 4U>  int4; 
# 1005
typedef vector< unsigned, 2U>  uint2; 
# 1006
typedef vector< unsigned, 3U>  uint3; 
# 1007
typedef vector< unsigned, 4U>  uint4; 
# 1008
}
# 1030
#pragma warning (pop)
# 7 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) float luminance(const math::float3 &color) 
# 8
{int volatile ___ = 1;(void)color;
# 10
::exit(___);}
#if 0
# 8
{ 
# 9
return dot(color, math::float3{(0.2125999928F), (0.715200007F), (0.07220000029F)}); 
# 10
} 
#endif
# 12 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) float Uncharted2Tonemap(float x) 
# 13
{int volatile ___ = 1;(void)x;
# 22
::exit(___);}
#if 0
# 13
{ 
# 15
constexpr float A = ((0.1499999999999999944)); 
# 16
constexpr float B = ((0.5)); 
# 17
constexpr float C = ((0.1000000000000000056)); 
# 18
constexpr float D = ((0.2000000000000000111)); 
# 19
constexpr float E = ((0.02000000000000000042)); 
# 20
constexpr float F = ((0.2999999999999999889)); 
# 21
return (((x * ((A * x) + (C * B))) + (D * E)) / ((x * ((A * x) + B)) + (D * F))) - (E / F); 
# 22
} 
#endif
# 24 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) float tonemap(float c, float exposure) 
# 25
{int volatile ___ = 1;(void)c;(void)exposure;
# 28
::exit(___);}
#if 0
# 25
{ 
# 26
constexpr float W = ((11.19999999999999929)); 
# 27
return Uncharted2Tonemap((c * exposure) * (2.0F)) / Uncharted2Tonemap(W); 
# 28
} 
#endif
# 30 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) math::float3 tonemap(const math::float3 &c, float exposure) 
# 31
{int volatile ___ = 1;(void)c;(void)exposure;
# 33
::exit(___);}
#if 0
# 31
{ 
# 32
return {tonemap(c.x, exposure), tonemap(c.y, exposure), tonemap(c.z, exposure)}; 
# 33
} 
#endif
# 35 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) unsigned char toLinear8(float c) 
# 36
{int volatile ___ = 1;(void)c;
# 38
::exit(___);}
#if 0
# 36
{ 
# 37
return static_cast< unsigned char>(saturate(c) * (255.0F)); 
# 38
} 
#endif
# 40 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) unsigned char toSRGB8(float c) 
# 41
{int volatile ___ = 1;(void)c;
# 43
::exit(___);}
#if 0
# 41
{ 
# 42
return toLinear8(powf(c, (1.0F) / (2.200000048F))); 
# 43
} 
#endif
# 45 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) float fromLinear8(unsigned char c) 
# 46
{int volatile ___ = 1;(void)c;
# 48
::exit(___);}
#if 0
# 46
{ 
# 47
return c * ((1.0F) / (255.0F)); 
# 48
} 
#endif
# 50 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/color.cuh"
__attribute__((unused)) float fromSRGB8(unsigned char c) 
# 51
{int volatile ___ = 1;(void)c;
# 53
::exit(___);}
#if 0
# 51
{ 
# 52
return powf(fromLinear8(c), (2.200000048F)); 
# 53
} 
#endif
# 10 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
namespace _GLOBAL__N__31_hdr_pipeline_compute_52_cpp1_ii_b5585b97 { }; using namespace ::_GLOBAL__N__31_hdr_pipeline_compute_52_cpp1_ii_b5585b97; namespace _GLOBAL__N__31_hdr_pipeline_compute_52_cpp1_ii_b5585b97 { 
# 11
constexpr unsigned divup(unsigned a, unsigned b) 
# 12
{ 
# 13
return ((a + b) - (1)) / b; 
# 14
} 
# 16
template< class T> void 
# 17
swap(T &a, T &b) 
# 18
{ 
# 19
T c = b; 
# 20
b = a; 
# 21
a = c; 
# 22
} 
# 23
}
# 26
void luminance_kernel(float *dest, const float *input, unsigned width, unsigned height) ;
#if 0
# 27
{ 
# 29
unsigned x = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 30
unsigned y = ((__device_builtin_variable_blockIdx.y) * (__device_builtin_variable_blockDim.y)) + (__device_builtin_variable_threadIdx.y); 
# 35
if ((x < width) && (y < height)) { 
# 36
const float *input_pixel = input + ((3) * ((width * y) + x)); 
# 38
float lum = (((0.2099999934F) * (input_pixel[0])) + ((0.7200000286F) * (input_pixel[1]))) + ((0.0700000003F) * (input_pixel[2])); 
# 40
(dest[(width * y) + x]) = lum; 
# 41
}  
# 42
} 
#endif
# 45 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
void luminance(float *dest, const float *input, unsigned width, unsigned height) { 
# 46
const dim3 block_size = {32, 32}; 
# 48
const dim3 num_blocks = {divup(width, block_size.x), divup(height, block_size.y)}; 
# 54
(cudaConfigureCall(num_blocks, block_size)) ? (void)0 : (luminance_kernel)(dest, input, width, height); 
# 60
} 
# 62
void downsample_kernel(float *dest, float *input, unsigned width, unsigned height, unsigned outputPitch, unsigned inputPitch) ;
#if 0
# 62
{ 
# 64
unsigned x = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 65
unsigned y = ((__device_builtin_variable_blockIdx.y) * (__device_builtin_variable_blockDim.y)) + (__device_builtin_variable_threadIdx.y); 
# 67
int F = 2; 
# 70
if (((x * F) > (width - (1))) || ((y * F) > (height - (1)))) { 
# 71
return; }  
# 73
float sum = (0.0F); 
# 75
int nb_counted = F * F; 
# 77
int xDim = min(F, width - (x * F)); 
# 78
int yDim = min(F, height - (y * F)); 
# 80
nb_counted = (xDim * yDim); 
# 84
for (int j = 0; j < F; j++) { 
# 85
for (int i = 0; i < F; i++) { 
# 87
if ((((y * F) + j) < height) && (((x * F) + i) < width)) { 
# 88
sum += (input[((((y * F) + j) * inputPitch) + (x * F)) + i]); 
# 89
}  
# 90
}  
# 91
}  
# 92
(dest[(y * outputPitch) + x]) = (sum / nb_counted); 
# 93
} 
#endif
# 96 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
float downsample(float *dest, float *luminance, unsigned width, unsigned height) { 
# 97
const dim3 block_size = {32, 32}; 
# 99
const dim3 num_blocks = {divup(width, block_size.x), divup(height, block_size.y)}; 
# 105
const unsigned pitchBuf = width / (2); 
# 106
const unsigned pitchLuminance = width; 
# 109
(cudaConfigureCall(num_blocks, block_size)) ? (void)0 : (downsample_kernel)(dest, luminance, width, height, pitchBuf, pitchLuminance); 
# 110
int ping = 0; 
# 112
while ((width != (1)) || (height != (1))) { 
# 114
width = (width / (2)); 
# 115
height = (height / (2)); 
# 116
if (width < (1)) { 
# 117
width = (1); 
# 118
printf("width < 1 \n"); 
# 119
}  
# 120
if (height < (1)) { 
# 121
height = (1); 
# 122
printf("height < 1 \n"); 
# 123
}  
# 125
printf(" width %d | height %d \n", width, height); 
# 126
if (ping) { 
# 127
(cudaConfigureCall(num_blocks, block_size)) ? (void)0 : (downsample_kernel)(dest, luminance, width, height, pitchBuf, pitchLuminance); 
# 128
} else 
# 129
{ 
# 131
(cudaConfigureCall(num_blocks, block_size)) ? (void)0 : (downsample_kernel)(luminance, dest, width, height, pitchLuminance, pitchBuf); 
# 132
}  
# 133
ping = (!ping); 
# 134
}  
# 137
if (ping) { 
# 138
cudaMemcpy(luminance, dest, 1, cudaMemcpyDeviceToDevice); 
# 139
}  
# 142
float average; 
# 143
cudaMemcpy(&average, dest, sizeof(float), cudaMemcpyDeviceToHost); 
# 144
return average; 
# 145
} 
# 150
void blur_kernel_x(float *dest, const float *src, unsigned width, unsigned height, unsigned inputPitch, unsigned outputPitch) ;
#if 0
# 151
{ 
# 152
constexpr float weights[] = {(0.002882040106F), (0.004183189943F), (0.005927539896F), (0.008199799806F), (0.01107368991F), (0.01459965017F), (0.01879115961F), (0.02361161076F), (0.02896397933F), (0.03468580917F), (0.04055143893F), (0.04628301039F), (0.05157006904F), (0.05609637126F), (0.05957068875F), (0.06175772846F), (0.06250444055F), (0.06175772846F), (0.05957068875F), (0.05609637126F), (0.05157006904F), (0.04628301039F), (0.04055143893F), (0.03468580917F), (0.02896397933F), (0.02361161076F), (0.01879115961F), (0.01459965017F), (0.01107368991F), (0.008199799806F), (0.005927539896F), (0.004183189943F), (0.002882040106F)}; 
# 162
unsigned x = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 163
unsigned y = ((__device_builtin_variable_blockIdx.y) * (__device_builtin_variable_blockDim.y)) + (__device_builtin_variable_threadIdx.y); 
# 165
float sum = (0.0F); 
# 166
for (int i = x - (16); i <= 16; i++) { 
# 167
if ((x + i) < width) { 
# 168
printf("weight: %d \n", (weights)[i + 16]); 
# 169
sum += ((src[((y * inputPitch) + x) + i]) * ((weights)[i + 16])); 
# 170
}  
# 171
}  
# 172
printf("sum: %d \n", sum); 
# 173
(dest[(y * outputPitch) + x]) = sum; 
# 174
} 
#endif
# 176 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
void gaussian_blur(float *dest, const float *src, unsigned width, unsigned height) 
# 177
{ 
# 178
const dim3 block_size = {32, 32}; 
# 180
const dim3 num_blocks = {divup(width, block_size.x), divup(height, block_size.y)}; 
# 185
int inputPitch = width; 
# 186
int outputPitch = width; 
# 187
(cudaConfigureCall(num_blocks, block_size)) ? (void)0 : (blur_kernel_x)(dest, src, width, height, inputPitch, outputPitch); 
# 188
} 
# 192
void compose(float *output, const float *tonemapped, const float *blurred, unsigned width, unsigned height) 
# 193
{ 
# 195
} 
# 199
void tonemap_kernel(float *tonemapped, float *brightpass, const float *src, unsigned width, unsigned height, float exposure, float brightpass_threshold) ;
#if 0
# 200
{ 
# 201
unsigned x = ((__device_builtin_variable_blockIdx.x) * (__device_builtin_variable_blockDim.x)) + (__device_builtin_variable_threadIdx.x); 
# 202
unsigned y = ((__device_builtin_variable_blockIdx.y) * (__device_builtin_variable_blockDim.y)) + (__device_builtin_variable_threadIdx.y); 
# 204
if ((x < width) && (y < height)) 
# 205
{ 
# 207
math::float3 c = {src[((3) * ((y * width) + x)) + (0)], src[((3) * ((y * width) + x)) + (1)], src[((3) * ((y * width) + x)) + (2)]}; 
# 210
math::float3 c_t = tonemap(c, exposure); 
# 213
(tonemapped[((3) * ((y * width) + x)) + (0)]) = (c_t.x); 
# 214
(tonemapped[((3) * ((y * width) + x)) + (1)]) = (c_t.y); 
# 215
(tonemapped[((3) * ((y * width) + x)) + (2)]) = (c_t.z); 
# 218
math::float3 c_b = ((luminance(c_t) > brightpass_threshold) ? c_t : math::float3{(0.0F), (0.0F), (0.0F)}); 
# 219
(brightpass[((3) * ((y * width) + x)) + (0)]) = (c_b.x); 
# 220
(brightpass[((3) * ((y * width) + x)) + (1)]) = (c_b.y); 
# 221
(brightpass[((3) * ((y * width) + x)) + (2)]) = (c_b.z); 
# 222
}  
# 223
} 
#endif
# 225 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
void tonemap(float *tonemapped, float *brightpass, const float *src, unsigned width, unsigned height, float exposure, float brightpass_threshold) 
# 226
{ 
# 227
const auto block_size = dim3{32U, 32U}; 
# 229
auto num_blocks = dim3{divup(width, block_size.x), divup(height, block_size.y)}; 
# 231
(cudaConfigureCall(num_blocks, block_size)) ? (void)0 : (tonemap_kernel)(tonemapped, brightpass, src, width, height, exposure, brightpass_threshold); 
# 232
} 

# 1 "hdr_pipeline.compute_52.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__31_hdr_pipeline_compute_52_cpp1_ii_b5585b97
# 1 "hdr_pipeline.compute_52.cudafe1.stub.c"
#include "hdr_pipeline.compute_52.cudafe1.stub.c"
# 1 "hdr_pipeline.compute_52.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
