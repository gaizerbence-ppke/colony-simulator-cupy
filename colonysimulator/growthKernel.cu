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
        int index = ypos * width + xpos;
        int area = length * width;
        int binomialBufferDimension = maxGrowth + maxPerish + 1;
        
        float food = foodRatio[index];

        int center = round((maxGrowth + maxPerish) * food);
        int stepsLeft = center;
        int stepsRight = maxGrowth + maxPerish - center;


        int binomialStart = 0;
        int binomialSize = 2 * stepsLeft + 1;

        if (stepsLeft > stepsRight)
        {
            binomialStart = center - stepsRight;
            binomialSize = 2 * stepsRight + 1;
        }
        binomialCoefficients += binomialBufferDimension * (binomialSize - 1);
        postGrowthTemporal += area * binomialStart + index;
        growthDistribution += index;

        int temporalLocation, sourceLocation;
        for (int i = 0; i < binomialSize; i++)
        {
            for (int j = 0; j < maturityBrackets; j++)
            {
                temporalLocation = (i + j) * area;
                sourceLocation = j * area;

                postGrowthTemporal[temporalLocation] += growthDistribution[sourceLocation] * binomialCoefficients[i];
            }
        }
    }
}