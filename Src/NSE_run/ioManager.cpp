#include <IOManager.H>

IOManager::IOManager(const SimConfig& config) : cfg(config)
{
    
}

void IOManager::writeMyPlotFile(int step, amrex::Real time, const FlowField& state, const amrex::BoxArray ba, const amrex::DistributionMapping dm, const amrex::Geometry& geom, int n_cell, std::string plot_prefix)
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