#include <MyFunctions.H>

using namespace amrex;

void initializeFlowField(FlowField& init_state)
{
    // profiling block for Tiny Profiling
    BL_PROFILE("<Setup> initializeFlowField()");

    // extracting data from FlowField object
    const amrex::Geometry& geom = init_state.getGeom();

    // extracting physical dx, physical domain lo for computing x,y,z
    GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // data extracted from "isotropic finite volume discretizations - RK Shukla, P Giri"
    amrex::Real r0 = 0.05;
    amrex::Real omega_0 = 2.0 * std::sqrt(2.0 * exp(1.0)) / r0;

    // defining the vortex centers
    amrex::Real dTheta = M_PI / 32.0;
    amrex::Vector<amrex::Real> thetas = {0.0, 2.0*M_PI/3.0, 4.0*M_PI/3.0};

    // defining GPU arrays for centers to be used inside ParallelFor's
    amrex::GpuArray<amrex::Real, 6> cx;
    amrex::GpuArray<amrex::Real, 6> cy;
    
    int idx = 0;
    for (int t = 0; t < 3; ++t) 
    {
        cx[idx] = 0.5 * std::cos(thetas[t] - dTheta);
        cy[idx] = 0.5 * std::sin(thetas[t] - dTheta);
        idx++;
        cx[idx] = 0.5 * std::cos(thetas[t] + dTheta);
        cy[idx] = 0.5 * std::sin(thetas[t] + dTheta);
        idx++;
    }

    // initializing x-velocity
    for (amrex::MFIter mfi(init_state.getVel(0), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& u_arr = init_state.getVel(0).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // x-face 'i' is exactly on the edge, 'j' is cell-centered
            amrex::Real x = prob_lo[0] + i * dx[0]; 
            amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1]; 

            amrex::Real u_val = 0.0;
            
            // linear combination of all 6 vortices
            for (int v = 0; v < 6; ++v) 
            {
                amrex::Real xc = cx[v];
                amrex::Real yc = cy[v];
                amrex::Real r2 = (x - xc)*(x - xc) + (y - yc)*(y - yc);
                
                u_val += (-0.5 * omega_0) * (y - yc) * std::exp(-r2 / (r0 * r0));
            }
            u_arr(i,j,k) = u_val;
        });
    }

    // initializing y-velocity
    for (amrex::MFIter mfi(init_state.getVel(1), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& v_arr = init_state.getVel(1).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // y-face 'i' is cell-centered, 'j' is exactly on the edge
            amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0]; 
            amrex::Real y = prob_lo[1] + j * dx[1]; 

            amrex::Real v_val = 0.0;
            
            for (int v = 0; v < 6; ++v) 
            {
                amrex::Real xc = cx[v];
                amrex::Real yc = cy[v];
                amrex::Real r2 = (x - xc)*(x - xc) + (y - yc)*(y - yc);
                
                v_val += (0.5 * omega_0) * (x - xc) * std::exp(-r2 / (r0 * r0));
            }
            v_arr(i,j,k) = v_val;
        });
    }

}