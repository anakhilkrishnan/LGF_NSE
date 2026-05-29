#ifndef MYFUNCTIONS_H_
#define MYFUNCTIONS_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_BCRec.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BLProfiler.H>
#include <string>

#include <FlowField.H>
#include <ProjectionWorkspace.H>

using namespace amrex;

void testPrint();
amrex::MultiFab computeCellCenteredVorticity(const FlowField& state);
void writeStaggeredPlotFile(int step, amrex::Real time, const FlowField& state, const amrex::BoxArray ba, const amrex::DistributionMapping dm, const amrex::Geometry& geom, int n_cell, std::string plot_prefix);
void extendedMain();

void initializeFlowField(FlowField& init_state);

#endif