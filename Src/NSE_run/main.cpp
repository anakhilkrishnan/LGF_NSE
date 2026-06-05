#include <MyFunctions.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    amrex::Print() << "Launching LGF-NSE solver..." << "\n";
    extendedMain();

    amrex::Finalize();
    return 0;
}

void extendedMain()
{
    BL_PROFILE("extendedMain()");

    auto overall_start_time = amrex::second();

    // creating simulation configuration object and reading inputs
    SimConfig cfg;
    cfg.readInputs();
    
    // creating input/output object 
    IOManager io(cfg);

    // creating domain data objects
    amrex::IntVect dom_lo_iv(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi_iv(AMREX_D_DECL(cfg.n_cell-1, cfg.n_cell-1, cfg.n_cell-1));
    amrex::Box domain(dom_lo_iv, dom_hi_iv);

    amrex::BoxArray ba(domain);
    ba.maxSize(cfg.max_grid_size);

    amrex::DistributionMapping dm(ba);

    amrex::RealBox real_box(cfg.dom_lo, cfg.dom_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // infinite domain using zero-grad BC
    amrex::Geometry geom(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

    // create flow field object
    FlowField state_n(geom, ba, dm, cfg.n_comp, cfg.n_ghost);
    // create solver object
    ProjectionWorkspace workspace(geom, ba, dm, cfg.n_comp, cfg.n_ghost);

    initializeFlowField(state_n);

    state_n.setBoundary(geom);
    
    // performing time stepping
    amrex::Real dt;
    amrex::Real time = cfg.t_start;
    int step = 0;

    // plotting initial conditions
    if (cfg.write_plot)
    {
        BL_PROFILE("<IO> Initial Plot()");
        io.writeMyPlotFile(step, time, state_n, ba, dm, geom, cfg.n_cell, cfg.plot_prefix);

    }

    while(time < cfg.t_stop && step < cfg.max_steps)
    {
        auto step_start_time = amrex::second();
        
        //dt = 0.0001;
        dt = workspace.computeDt(state_n, cfg.cfl, cfg.Re);
        
        // advance time using RK for time, KEP Morinishi for space and LGF for
        // pressure poisson
        workspace.advanceTimeStep(state_n, dt, cfg.Re, cfg.rk_order, cfg.source_tag_thresh);

        // update counters
        time += dt;
        step++;

        //  plot in specified intervals
        if (step % cfg.plot_int == 0 && cfg.write_plot)
        {
            BL_PROFILE("<IO> Interval Plot()");
            io.writeMyPlotFile(step, time, state_n, ba, dm, geom, cfg.n_cell, cfg.plot_prefix);
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

