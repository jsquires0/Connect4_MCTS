
kernel_c_code = """
extern "C"
{

    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }




}
"""
