#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "hdr_pipeline.fatbin.c"
extern void __device_stub__Z16luminance_kernelPfPKfjj(float *, const float *, unsigned, unsigned);
extern void __device_stub__Z17downsample_kernelPfS_jjjj(float *, float *, unsigned, unsigned, unsigned, unsigned);
extern void __device_stub__Z14tonemap_kernelP6uchar4S0_PKfjjff(struct uchar4 *, struct uchar4 *, const float *, unsigned, unsigned, float, float);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97(void) __attribute__((__constructor__));
void __device_stub__Z16luminance_kernelPfPKfjj(float *__par0, const float *__par1, unsigned __par2, unsigned __par3){__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 20UL);__cudaLaunch(((char *)((void ( *)(float *, const float *, unsigned, unsigned))luminance_kernel)));}
# 62 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
void luminance_kernel( float *__cuda_0,const float *__cuda_1,unsigned __cuda_2,unsigned __cuda_3)
# 63 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{__device_stub__Z16luminance_kernelPfPKfjj( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 78 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}
# 1 "hdr_pipeline.compute_35.cudafe1.stub.c"
void __device_stub__Z17downsample_kernelPfS_jjjj( float *__par0,  float *__par1,  unsigned __par2,  unsigned __par3,  unsigned __par4,  unsigned __par5) {  __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaLaunch(((char *)((void ( *)(float *, float *, unsigned, unsigned, unsigned, unsigned))downsample_kernel))); }
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
void downsample_kernel( float *__cuda_0,float *__cuda_1,unsigned __cuda_2,unsigned __cuda_3,unsigned __cuda_4,unsigned __cuda_5)
# 98 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{__device_stub__Z17downsample_kernelPfS_jjjj( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 120 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}
# 1 "hdr_pipeline.compute_35.cudafe1.stub.c"
void __device_stub__Z14tonemap_kernelP6uchar4S0_PKfjjff( struct uchar4 *__par0,  struct uchar4 *__par1,  const float *__par2,  unsigned __par3,  unsigned __par4,  float __par5,  float __par6) {  __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 28UL); __cudaSetupArgSimple(__par5, 32UL); __cudaSetupArgSimple(__par6, 36UL); __cudaLaunch(((char *)((void ( *)(struct uchar4 *, struct uchar4 *, const float *, unsigned, unsigned, float, float))tonemap_kernel))); }
# 178 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
void tonemap_kernel( struct uchar4 *__cuda_0,struct uchar4 *__cuda_1,const float *__cuda_2,unsigned __cuda_3,unsigned __cuda_4,float __cuda_5,float __cuda_6)
# 179 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
{__device_stub__Z14tonemap_kernelP6uchar4S0_PKfjjff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 196 "/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/CUDA/Projects/HDR2/build/cmake/hdr_pipeline/../../../source/hdr_pipeline/hdr_pipeline.cu"
}
# 1 "hdr_pipeline.compute_35.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T216) {  __nv_dummy_param_ref(__T216); __nv_save_fatbinhandle_for_managed_rt(__T216); __cudaRegisterEntry(__T216, ((void ( *)(struct uchar4 *, struct uchar4 *, const float *, unsigned, unsigned, float, float))tonemap_kernel), _Z14tonemap_kernelP6uchar4S0_PKfjjff, (-1)); __cudaRegisterEntry(__T216, ((void ( *)(float *, float *, unsigned, unsigned, unsigned, unsigned))downsample_kernel), _Z17downsample_kernelPfS_jjjj, (-1)); __cudaRegisterEntry(__T216, ((void ( *)(float *, const float *, unsigned, unsigned))luminance_kernel), _Z16luminance_kernelPfPKfjj, (-1)); }
static void __sti____cudaRegisterAll_31_hdr_pipeline_compute_52_cpp1_ii_b5585b97(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
