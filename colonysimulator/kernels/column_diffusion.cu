// Vertical nutrient transport in every (x, y) column of the plate: one thread per column.
//
// A column is a stack of rows. Rows 0 .. agarDepth-1 are agar voxels (row 0 is the bottom of
// the dish, row agarDepth-1 touches the surface); rows agarDepth .. agarDepth+colonyRows-1
// are colony (ECM) layers, layer 0 directly on the agar. Arrays are C-ordered
// (row, x, y), so `row * area + column` addresses one voxel and neighbouring threads read
// neighbouring memory.
//
// For every row r the kernel solves the theta-scheme of
//
//     dz_r dC_r/dt = g_{r-1/2} (C_{r-1} - C_r) + g_{r+1/2} (C_{r+1} - C_r) - dz_r k_r C_r
//
// where g is the conductance between two rows (harmonic mean of D/dz, so an agar/ECM
// interface is handled correctly), and k_r is the Michaelis-Menten sink coefficient
// sum_s N_s m_max_s / (Km_s + C_old) / V, linearised around the concentration at the start of
// the step. Diffusion uses weight theta on the new time level (0.5 = Crank-Nicolson,
// 1 = backward Euler); the sink is always taken fully at the new time level, which keeps it
// from ever removing more nutrient than the voxel holds. Rows that are not part of the
// medium (colony voxels without ECM) are decoupled: no flux, concentration unchanged.
// The bottom of the agar and the top of the colony are closed (zero flux).
//
// The tridiagonal system is solved with the Thomas algorithm in double precision; the two
// sweep coefficients are kept in the scratch arrays.

extern "C"
{
    __global__ void columnDiffusionKernel(float* concentration,            // (agarDepth + colonyRows, area), in/out
                                          const float* sinkRate,           // (colonyRows, area), 1/h
                                          const bool* medium,              // (colonyRows, area), true where ECM is present
                                          const bool* plateMask,           // (area)
                                          double* sweepC,                  // scratch, (agarDepth + colonyRows, area)
                                          double* sweepD,                  // scratch, (agarDepth + colonyRows, area)
                                          int agarDepth,
                                          int colonyRows,
                                          int area,
                                          double agarDz,                   // mm, height of one agar voxel
                                          double colonyDz,                 // mm, height of one colony layer
                                          double agarDiffusion,            // mm^2/h
                                          double ecmDiffusion,             // mm^2/h
                                          double dt,                       // h
                                          double theta)
    {
        int column = blockIdx.x * blockDim.x + threadIdx.x;
        if (column >= area || !plateMask[column])
            return;

        int rows = agarDepth + colonyRows;

        // Properties of the current row and the conductance to the row below, carried along.
        double gBelow = 0.0;         // conductance between row r-1 and row r
        double cBelow = 0.0;         // C_{r-1} at the old time level
        double sweepCBelow = 0.0;    // Thomas coefficients of row r-1
        double sweepDBelow = 0.0;

        double c = concentration[column];
        bool isMedium = true;        // agar rows are always medium
        double diffusion = agarDiffusion;
        double dz = agarDz;

        for (int r = 0; r < rows; r++)
        {
            // Look one row up to get the conductance across the upper face of row r.
            double gAbove = 0.0;
            double cAbove = 0.0;
            bool aboveIsMedium = false;
            double aboveDiffusion = 0.0;
            double aboveDz = 0.0;
            if (r + 1 < rows)
            {
                int above = r + 1;
                // Agar rows are always medium. || stops before the read when the row is agar,
                // so `medium` is never indexed with a negative row.
                aboveIsMedium = above < agarDepth || medium[(above - agarDepth) * area + column];
                aboveDiffusion = above < agarDepth ? agarDiffusion : ecmDiffusion;
                aboveDz = above < agarDepth ? agarDz : colonyDz;
                cAbove = concentration[above * area + column];
                if (isMedium && aboveIsMedium)
                    gAbove = 1.0 / (0.5 * dz / diffusion + 0.5 * aboveDz / aboveDiffusion);
            }

            double sink = 0.0;
            if (r >= agarDepth && isMedium)
                sink = sinkRate[(r - agarDepth) * area + column];

            // Row r of the system:  lower * C_{r-1} + diag * C_r + upper * C_{r+1} = rhs
            double ratio = dt / dz;
            double lower = -theta * ratio * gBelow;
            double upper = -theta * ratio * gAbove;
            double diag = 1.0 + theta * ratio * (gBelow + gAbove) + dt * sink;
            double rhs = c + (1.0 - theta) * ratio * (gBelow * (cBelow - c) + gAbove * (cAbove - c));

            // Forward sweep.
            double denominator = diag - lower * sweepCBelow;
            double sweepCHere = upper / denominator;
            double sweepDHere = (rhs - lower * sweepDBelow) / denominator;
            sweepC[r * area + column] = sweepCHere;
            sweepD[r * area + column] = sweepDHere;

            // Shift the window up by one row.
            gBelow = gAbove;
            cBelow = c;
            sweepCBelow = sweepCHere;
            sweepDBelow = sweepDHere;
            c = cAbove;
            isMedium = aboveIsMedium;
            diffusion = aboveDiffusion;
            dz = aboveDz;
        }

        // Back substitution, top row first.
        double next = 0.0;
        for (int r = rows - 1; r >= 0; r--)
        {
            double value = sweepD[r * area + column] - sweepC[r * area + column] * next;
            concentration[r * area + column] = (float)value;
            next = value;
        }
    }
}
