#include <MyFunctions.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    testPrint();
    extendedMain();

    amrex::Finalize();
    return 0;
}

void testPrint()
{
    amrex::Print() << "This is the start of my first NSE solver." << "\n";
}

amrex::MultiFab computeCellCenteredVorticity(const FlowField& state)
{
    BL_PROFILE("computeCellCenteredVorticity()");

    // get geometry and grid info using your safely encapsulated getter!
    const amrex::Geometry& geom = state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    amrex::BoxArray ba = state.getPres().boxArray();
    amrex::DistributionMapping dm = state.getPres().DistributionMap();

    // allocate the cell-centered vorticity MultiFab
    // In 2D: 1 component (omega_z). In 3D: 3 components (omega_x, omega_y, omega_z)
    int ncomp = (AMREX_SPACEDIM == 2) ? 1 : 3;
    amrex::MultiFab vort(ba, dm, ncomp, 0);

    // compute the averaged gradients on the GPU
    for (amrex::MFIter mfi(vort, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();

        auto const& vort_arr = vort.array(mfi);
        auto const& u_arr    = state.getVel(0).const_array(mfi);
        auto const& v_arr    = state.getVel(1).const_array(mfi);
        
        #if AMREX_SPACEDIM == 3
        auto const& w_arr    = state.getVel(2).const_array(mfi);
        #endif

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Note: A face-centered array at index (i,j,k) represents the 'left' or 'bottom' face.
            // (i+1,j,k) represents the 'right' face, etc.
            
            #if AMREX_SPACEDIM == 2
                // dv/dx averaged from top and bottom y-faces
                amrex::Real dvdx = (v_arr(i+1,j,k) + v_arr(i+1,j+1,k) - v_arr(i-1,j,k) - v_arr(i-1,j+1,k)) / (4.0 * dx[0]);
                // du/dy averaged from left and right x-faces
                amrex::Real dudy = (u_arr(i,j+1,k) + u_arr(i+1,j+1,k) - u_arr(i,j-1,k) - u_arr(i+1,j-1,k)) / (4.0 * dx[1]);
                
                vort_arr(i,j,k) = dvdx - dudy;

            #elif AMREX_SPACEDIM == 3
                // omega_x = dw/dy - dv/dz
                amrex::Real dwdy = (w_arr(i,j+1,k) + w_arr(i,j+1,k+1) - w_arr(i,j-1,k) - w_arr(i,j-1,k+1)) / (4.0 * dx[1]);
                amrex::Real dvdz = (v_arr(i,j,k+1) + v_arr(i,j+1,k+1) - v_arr(i,j,k-1) - v_arr(i,j+1,k-1)) / (4.0 * dx[2]);
                vort_arr(i,j,k,0) = dwdy - dvdz;

                // omega_y = du/dz - dw/dx
                amrex::Real dudz = (u_arr(i,j,k+1) + u_arr(i+1,j,k+1) - u_arr(i,j,k-1) - u_arr(i+1,j,k-1)) / (4.0 * dx[2]);
                amrex::Real dwdx = (w_arr(i+1,j,k) + w_arr(i+1,j,k+1) - w_arr(i-1,j,k) - w_arr(i-1,j,k+1)) / (4.0 * dx[0]);
                vort_arr(i,j,k,1) = dudz - dwdx;

                // omega_z = dv/dx - du/dy
                amrex::Real dvdx = (v_arr(i+1,j,k) + v_arr(i+1,j+1,k) - v_arr(i-1,j,k) - v_arr(i-1,j+1,k)) / (4.0 * dx[0]);
                amrex::Real dudy = (u_arr(i,j+1,k) + u_arr(i+1,j+1,k) - u_arr(i,j-1,k) - u_arr(i+1,j-1,k)) / (4.0 * dx[1]);
                vort_arr(i,j,k,2) = dvdx - dudy;
            #endif
        });
    }

    return vort;
}

void writeStaggeredPlotFile(int step, amrex::Real time, const FlowField& state, const amrex::BoxArray ba, const amrex::DistributionMapping dm, const amrex::Geometry& geom, int n_cell, std::string plot_prefix)
{
    // checking total components for plotfile
    int ncomp_vort = (AMREX_SPACEDIM == 2) ? 1 : 3;
    int ncomp_plot = AMREX_SPACEDIM + ncomp_vort + 4 ;

    // building a multiFab with n dim + 2 components for plotting
   amrex::MultiFab plotFab(ba, dm, ncomp_plot, 0);

   // converting face-centered data to cell-centered data
    #if AMREX_SPACEDIM == 1
        amrex::average_face_to_cellcenter(plotFab, 0, amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>{&state.getVel(0)});
    #elif AMREX_SPACEDIM == 2
        amrex::average_face_to_cellcenter(plotFab, 0, amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>{&state.getVel(0), &state.getVel(1)});
    #elif AMREX_SPACEDIM == 3
        amrex::average_face_to_cellcenter(plotFab, 0, amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>{&state.getVel(0), &state.getVel(1), &state.getVel(2)});
    #endif
    
    amrex::MultiFab::Copy(plotFab, state.getPres(), 0, AMREX_SPACEDIM, 1, 0); 
    amrex::MultiFab::Copy(plotFab, state.getTagRegion(), 0, AMREX_SPACEDIM + 1, 1, 0);
    amrex::MultiFab::Copy(plotFab, state.getDivU(), 0, AMREX_SPACEDIM + 2, 1, 0);
    amrex::MultiFab::Copy(plotFab, state.getDivUAtEnd(), 0, AMREX_SPACEDIM + 3, 1, 0);
    amrex::MultiFab::Copy(plotFab, computeCellCenteredVorticity(state), 0, AMREX_SPACEDIM + 4, ncomp_vort, 0);


    // exporting the names of the MultiFabs
    amrex::Vector<std::string> varnames = {AMREX_D_DECL("x_velocity", "y_velocity", "z_velocity"), "pressure", "active_box_tag", "divU", "divUAtEnd"};
    #if AMREX_SPACEDIM == 2
        varnames.push_back("z_vorticity");
    #elif AMREX_SPACEDIM == 3
        varnames.push_back("x_vorticity");
        varnames.push_back("y_vorticity");
        varnames.push_back("z_vorticity");
    #endif

    // writing a simple plotfile
    const std::string& plotfile_name = amrex::Concatenate(plot_prefix, step);
    amrex::Print() << "Writing plotfile to: " << plotfile_name << "\n";
    WriteSingleLevelPlotfile(plotfile_name, plotFab, varnames, geom, time, step);
    amrex::Print() << "Plotfile written to: " << plotfile_name << "\n";
}

amrex::Real computeDt(const FlowField& state, amrex::Real cfl, amrex::Real Re)
{
    const amrex::Geometry& geom = state.getGeom();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // cfl constraint: advective limit on dt
    amrex::Real u_max = state.getVel(0).norm0(0, 0, false);
    amrex::Real adv_metric = u_max / dx[0];

    #if AMREX_SPACEDIM >= 2
        amrex::Real v_max = state.getVel(1).norm0(0, 0, false);
        adv_metric += v_max / dx[1];
    #endif

    #if AMREX_SPACEDIM == 3
        amrex::Real w_max = state.getVel(2).norm0(0, 0, false);
        adv_metric += w_max / dx[2];
    #endif

    // dt_adv = CFL / ( |u|/dx + |v|/dy + |w|/dz )
    amrex::Real dt_adv = cfl / (adv_metric + 1.0e-12); // epsilon to prevent div-by-zero

    // diffusive limit on dt
    amrex::Real diff_metric = 1.0 / (dx[0] * dx[0]);
    
    #if AMREX_SPACEDIM >= 2
        diff_metric += 1.0 / (dx[1] * dx[1]);
    #endif

    #if AMREX_SPACEDIM == 3
        diff_metric += 1.0 / (dx[2] * dx[2]);
    #endif

    // for explicit schemes, Fourier number <= 0.5
    // dt_diff <= 0.5 * Re / ( 1/dx^2 + 1/dy^2 + 1/dz^2 )
    amrex::Real dt_diff = 0.5 * Re / diff_metric;

    return amrex::min(dt_adv, dt_diff);
}

void extendedMain()
{
    BL_PROFILE("extendedMain()");

    auto overall_start_time = amrex::second();

    // necessary ParmParse parameters
    int n_cell, max_grid_size, rk_order, plot_int, max_steps, n_ghost;
    amrex::Real t_start, t_stop, cfl, source_tag_thresh, Re;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dom_lo, dom_hi;

    // pre-defined ParmParse parameters
    int n_comp = 1;
    bool write_plot = true;
    std::string plot_prefix = "./Results/plt";

    // reading inputs file
    amrex::ParmParse pp;

    // domain data
    pp.get("n_cell",n_cell);
    pp.get("max_grid_size",max_grid_size);
    pp.get("domain_lo", dom_lo);
    pp.get("domain_hi", dom_hi);
    pp.get("n_ghost", n_ghost);
    pp.query("n_comp", n_comp);


    // solver data
    pp.get("Re", Re);
    pp.get("tagging_threshold", source_tag_thresh);

    // time stepping data
    pp.get("t_start", t_start);
    pp.get("t_stop", t_stop);
    pp.get("cfl", cfl);        
    pp.get("rk_order", rk_order);
    pp.get("max_steps", max_steps);

    // plotting data
    pp.query("write_plot", write_plot);
    pp.get("plot_int", plot_int);
    pp.query("plot_prefix", plot_prefix);


    // creating domain data objects
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    amrex::BoxArray ba(domain);
    ba.maxSize(max_grid_size);

    amrex::DistributionMapping dm(ba);

    amrex::RealBox real_box(dom_lo, dom_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // infinite domain using zero-grad BC
    amrex::Geometry geom(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

    // declare solver objects
    FlowField state_n(geom, ba, dm, n_comp, n_ghost);
    ProjectionWorkspace workspace(geom, ba, dm, n_comp, n_ghost);

    initializeFlowField(state_n);

    state_n.setBoundary(geom);
    // PENDING: employ boundary conditions somewhere here
    
    // performing time stepping
    amrex::Real dt;
    amrex::Real time = t_start;
    int step = 0;

    // plotting initial conditions
    if (write_plot)
    {
        BL_PROFILE("<IO> Initial Plot()");
        writeStaggeredPlotFile(step, time, state_n, ba, dm, geom, n_cell, plot_prefix);

    }

    while(time < t_stop && step < max_steps)
    {
        auto step_start_time = amrex::second();
        
        //dt = 0.0001;
        dt = computeDt(state_n, cfl, Re);
        
        // advance time using RK for time, KEP Morinishi for space and LGF for pressure poisson
        workspace.advanceTimeStep(state_n, dt, Re, rk_order, source_tag_thresh);

        // update counters
        time += dt;
        step++;

        //  plot in specified intervals
        if (step % plot_int == 0 && write_plot)
        {
            BL_PROFILE("<IO> Interval Plot()");
            writeStaggeredPlotFile(step, time, state_n, ba, dm, geom, n_cell, plot_prefix);
        }

        // track duration of simulation
        auto step_stop_time = amrex::second();
        auto step_duration = step_stop_time - step_start_time;

        // print to terminal each timestep
        amrex::Print() << "Step: " << step << " | Time: " << time << " | dt: " << dt 
                       << " | WallTime: " << (step_duration) << "s | divU_max: " << workspace.divU_max_norm << "\n";
    }

    // overall code walltime tracking
    auto overall_end_time = amrex::second();
    auto elapsed_time = overall_end_time - overall_start_time;

    // making copies to track slowest and fastest processor
    amrex::Real max_time = elapsed_time;
    amrex::Real min_time = elapsed_time;

    // performing a reduction over all the processors to track the slowest and
    // fastest MPI rank
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::ReduceRealMax(max_time, IOProc);
    amrex::ParallelDescriptor::ReduceRealMin(min_time, IOProc);

    amrex::Print() << "Max compute time (Slowest Rank): " << max_time << " s\n"
                   << "Min compute time (Fastest Rank): " << min_time << " s\n"
                   << "Time spread (Load Imbalance)   : " << (max_time - min_time) << " s\n";
}   

