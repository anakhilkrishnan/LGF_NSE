#ifndef IOMANAGER_H_
#define IOMANAGER_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_BLProfiler.H>
#include <string>

#include <SimConfig.H>
#include <FlowField.H>
#include <ProjectionWorkspace.H>


amrex::MultiFab computeCellCenteredVorticity(const FlowField& state);

class IOManager
{
    private:

        // simulation configuration object, passed by reference
        const SimConfig& cfg;

    public:
        IOManager(const SimConfig& config);
        void writeMyPlotFile(int step, amrex::Real time, const FlowField& state, const amrex::BoxArray ba, const amrex::DistributionMapping dm, const amrex::Geometry& geom, int n_cell, std::string plot_prefix);
 
};

#endif