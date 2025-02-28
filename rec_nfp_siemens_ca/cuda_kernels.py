import cupy as cp
# placing/extracting patches is better to do with CUDA C
Efast_kernel = cp.RawKernel(
    r"""                              
extern "C" 
void __global__ Efast(float2* res, float2 *psi, int* stx, int* sty, int npos, int npatch, int npsi)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= npatch || ty >= npatch || tz >= npos)
        return;        
    int ind_out = tz*npatch*npatch+ty*npatch+tx;    
    int ind_in = (sty[tz]+ty)*npsi+stx[tz]+tx;                                                           
    res[ind_out] = psi[ind_in];
}
""",
    "Efast",
)

ETfast_kernel = cp.RawKernel(
    r"""                              
extern "C" 
void __global__ ETfast(float2* res, float2 *psi, int* stx, int* sty, int npos, int npatch, int npsi)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= npatch || ty >= npatch || tz >= npos)
        return;        
    int ind_out = tz*npatch*npatch+ty*npatch+tx;    
    int ind_in = (sty[tz]+ty)*npsi+stx[tz]+tx;                                                           
    atomicAdd(&psi[ind_in].x,res[ind_out].x);
    atomicAdd(&psi[ind_in].y,res[ind_out].y);    
}
""",
    "ETfast",
)
