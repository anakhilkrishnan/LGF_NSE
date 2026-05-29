#include <ProjectionWorkspace.H>
#include <spatialDiscretization.H>
#include <RKCoefficients.H>
#include <LGFCore.H>

ProjectionWorkspace::ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const int n_comp, const int n_ghost)
    : geom(geom_in), ba(ba_in), dm(dm_in)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // convert the box array to face centered
        amrex::BoxArray ba_face = amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));

        // declare the specific velocity component
        rhs_vel[idim].define(ba_face, dm, n_comp, n_ghost);
        rhs_vel_corr[idim].define(ba_face, dm, n_comp, n_ghost);

        // initialize velocities upon creation
        rhs_vel[idim].setVal(0.0);
        rhs_vel_corr[idim].setVal(0.0);
    }

    // initialize corr_pres upon creation
    corr_pres.define(ba, dm, n_comp, n_ghost);
    corr_pres.setVal(0.0);

    divU_max_norm = 0.0;
}

void ProjectionWorkspace::computeConvectiveFluxes(const FlowField& stage, amrex::Real Re)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeConvectiveFluxes()");
    // compute the right hand side which is of the form
    // 1/Re(laplacian(u)) - grad(P) - u.divergence(u)
    // all taken at the n^th timestep
    // discretized using a second order finite difference KEP scheme
    // as outlined in Morinishi et al.

    // extracting physical dx for computations
    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // for each velocity direction, rhs is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(stage.getVel(idim), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) 
            {
                vel_arr[d] = stage.getVel(d).const_array(mfi);
            }
            auto const& pres_arr = stage.getPres().const_array(mfi);
            auto const& rhs = rhs_vel[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because these happen at compile time
                if (idim == 0) 
                {
                    rhs(i,j,k) = morinishiFlux<0>(i, j, k, vel_arr, pres_arr, dx, Re);
                }
                #if AMREX_SPACEDIM >= 2
                    else if (idim == 1) 
                    {
                        rhs(i,j,k) = morinishiFlux<1>(i, j, k, vel_arr, pres_arr, dx, Re);
                    }
                #endif
                #if AMREX_SPACEDIM == 3
                    else if (idim == 2) 
                    {
                        rhs(i,j,k) = morinishiFlux<2>(i, j, k, vel_arr, pres_arr, dx, Re);
                    }
                #endif
            });
        }
    }
}

void ProjectionWorkspace::predictVelocity(const FlowField& state_n, FlowField& stage, amrex::Real dt, amrex::Real alpha, amrex::Real beta, amrex::Real gamma)
{
    BL_PROFILE("<Compute> advanceTimeStep(): predictVelocity");
    // use the right hand side to predict velocity at the next time step,
    // before enforcing divergence free condition
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // using amrex's linalg functions for this step
        // not updating any ghost cell data here, those are updated by BCs
        amrex::MultiFab::LinComb(stage.getVel(idim), alpha, state_n.getVel(idim), 0, beta, stage.getVel(idim), 0, 0, stage.getVel(idim).nComp(), 0);
        amrex::Real dt_by_gam = dt / gamma;
        amrex::MultiFab::Saxpy(stage.getVel(idim), dt_by_gam, rhs_vel[idim], 0, 0, stage.getVel(idim).nComp(), 0);
    }
}

void ProjectionWorkspace::computePressure(FlowField& stage, amrex::Real source_tag_thresh, amrex::Vector<int>& box_tag_arr)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computePressure()");

    // extracting physical dx for computations
    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // compute divU and store in stage
    for(amrex::MFIter mfi(stage.getDivU(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divU_arr = stage.getDivU().array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            vel_arr[d] = stage.getVel(d).const_array(mfi);
        }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divU_arr(i,j,k) = discreteDivergence(i, j, k, dx, vel_arr);
        });
    }

    // use the custom lgf solver to compute the pressure at the next time step
    // running the tagging algorithmn and obtaining the box tags as an array of 0s and 1s
    box_tag_arr = tagSource(stage.getDivU(), source_tag_thresh);
    
    // write out divU_max_norm
    divU_max_norm = stage.getDivU().norm0(0, 0, false);

    // performing addition of box values
    addEverySourceBox(stage.getDivU(), corr_pres, geom, box_tag_arr);

    corr_pres.FillBoundary(geom.periodicity());
}

void ProjectionWorkspace::computeVelocityCorrection(FlowField& stage)
{
    BL_PROFILE("<Compute> advanceTimeStep(): computeVelocityCorrection");

    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // for each velocity direction, vel_corr is computed accordingly
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        for (amrex::MFIter mfi(stage.getVel(idim), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            
            auto const& corr_pres_arr = corr_pres.const_array(mfi);
            auto const& rhs_corr = rhs_vel_corr[idim].array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // evaluating one at a time for template variable idim, because these happen at compile time
                if (idim == 0) 
                {
                    rhs_corr(i,j,k) = discreteGradient<0>(i, j, k, corr_pres_arr, dx);
                }
                #if AMREX_SPACEDIM >= 2
                    else if (idim == 1) 
                    {
                        rhs_corr(i,j,k) = discreteGradient<1>(i, j, k, corr_pres_arr, dx);
                    }
                #endif
                #if AMREX_SPACEDIM == 3
                    else if (idim == 2) 
                    {
                        rhs_corr(i,j,k) = discreteGradient<2>(i, j, k, corr_pres_arr, dx);
                    }
                #endif
            });
        }
    }
}

void ProjectionWorkspace::correctVelocityandPressure(FlowField& stage, amrex::Real gamma, amrex::Real dt)
{
    BL_PROFILE("<Compute> advanceTimeStep(): correctVelocity()");

    // use the updated pressure to correct velocity to a divergence free field
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // updating velocity correctly
        // BCs are not updated here
        amrex::MultiFab::Subtract(stage.getVel(idim), rhs_vel_corr[idim], 0, 0, stage.getVel(idim).nComp(), 0);
    }

    // updating pressure to reflect base state + corrected
    amrex::Real gam_by_dt = gamma/dt;
    amrex::MultiFab::Saxpy(stage.getPres(), gam_by_dt, corr_pres, 0, 0, stage.getPres().nComp(), stage.getPres().nGrow());

    // additional checker for divergence at end of step
    // extracting physical dx for computations
    const amrex::Geometry& geom = stage.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // compute divU_at_end
    for(amrex::MFIter mfi(stage.getDivUAtEnd(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const& divU_at_end_arr = stage.getDivUAtEnd().array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> vel_arr;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) 
        {
            vel_arr[d] = stage.getVel(d).const_array(mfi);
        }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            divU_at_end_arr(i,j,k) = discreteDivergence(i, j, k, dx, vel_arr);
        });
    }
}

void ProjectionWorkspace::advanceTimeStep(FlowField& state_n, amrex::Real dt, amrex::Real Re, int rk_order, amrex::Real source_tag_thresh)
{

    // perform low-storage RK method for specified order, which can be reduced 
    // to a set of Forward Euler like stages with the final sum having appropriate
    // coefficients alpha, beta, gamma

    BL_PROFILE("<Compute> advanceTimeStep()");
    FlowField stage = state_n;
    amrex::Vector<RKCoeffs> coeffs = getRKCoeffs(rk_order);
    amrex::Vector<int> tag_region;

    for(int k = 0; k < rk_order; ++k)
    {
        // extracting RK coefficients
        amrex::Real alpha = coeffs[k].alp;
        amrex::Real beta = coeffs[k].bet;
        amrex::Real gamma = coeffs[k].gam;

        // compute and store fluxes in workspace
        computeConvectiveFluxes(stage, Re);

        // compute predicted velocity without divergence free condition
        // store predicted velocity within stage
        predictVelocity(state_n, stage, dt, alpha, beta, gamma);
        stage.setBoundary(geom);

        // find divergence of predicted velocity, store in workspace
        // use custom LGF solver to find pressure correction delta
        // update pressure stored in stage
        computePressure(stage, source_tag_thresh, tag_region);

        // use pressure to compute velocity correction
        // store correction in workspace
        computeVelocityCorrection(stage);

        // correct stage using correction from workspace
        correctVelocityandPressure(stage, gamma, dt);
        stage.setBoundary(geom);
    }

    // export divergence at the end of each time step for confidence
    for (MFIter mfi(stage.getTagRegion()); mfi.isValid(); ++mfi) 
    {
        if (tag_region[mfi.LocalIndex()] == 1) 
        {
            // If active, fill the entire box with 1.0 (on the GPU)
            stage.getTagRegion()[mfi].setVal<RunOn::Device>(1.0); 
        } 
        else 
        {
            // If inactive, fill the entire box with 0.0 (on the GPU)
            stage.getTagRegion()[mfi].setVal<RunOn::Device>(0.0); 
        }
    }

    state_n = stage;
}