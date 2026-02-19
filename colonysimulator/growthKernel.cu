extern "C"
{
    __global__ void growthKernel(float* growthDistribution, 
                                float* postGrowthTemporal, 
                                float* foodRatio,
                                float* binomialCoefficients,
                                int maxGrowth,
                                int maxPerish,
                                int maturityBrackets,
                                int length, 
                                int width)
    {
        int xpos = blockIdx.x * blockDim.x + threadIdx.x;
        int ypos = blockIdx.y * blockDim.y + threadIdx.y;
        if (xpos >= length || ypos >= width)
            return;
        int index = xpos * width + ypos;
        int area = length * width;
        int binomialBufferDimension = maxGrowth + maxPerish + 1;
        
        float food = foodRatio[index];

        int center = round((maxGrowth + maxPerish) * food);
        int stepsLeft = center;
        int stepsRight = maxGrowth + maxPerish - center;


        int binomialStart = 0;
        int binomialSize = 2 * stepsLeft + 1;
        binomialCoefficients += binomialBufferDimension * (binomialSize - 1);

        if (stepsLeft > stepsRight)
        {
            binomialStart = stepsLeft - stepsRight;
            binomialSize = 2 * stepsRight + 1;
        }

        for (int i = 0; i < binomialSize; i++)
        {
            for (int j = 0; j < maturityBrackets; j++)
            {
                postGrowthTemporal[(binomialStart + i + j) * area + index] += growthDistribution[index + j * area] * binomialCoefficients[i];
            }
        }
    }
}