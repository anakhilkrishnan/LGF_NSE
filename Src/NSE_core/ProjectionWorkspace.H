#ifndef PROJECTIONWORKSPACE_H_
#define PROJECTIONWORKSPACE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>
#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>

#include <FlowField.H>
#include <SpatialDiscretization.H>
#include <RKCoefficients.H>
#include <LGFCore.H>

using namespace amrex;

class ProjectionWorkspace
{
    private:
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel;
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> rhs_vel_corr;
        amrex::MultiFab corr_pres;
        
    public: 
        amrex::Real divU_max_norm;
        ProjectionWorkspace(const amrex::Geometry& geom_in, const amrex::BoxArray& ba_in, const amrex::DistributionMapping& dm_in, const int n_comp, const int n_ghost);
        amrex::Real computeDt(const FlowField& state, amrex::Real cfl, amrex::Real Re);
        void computeConvectiveFluxes(const FlowField& stage, amrex::Real Re);
        void predictVelocity(const FlowField& state_n, FlowField& stage, amrex::Real dt, amrex::Real alpha, amrex::Real beta, amrex::Real gamma);
        void computePressure(FlowField& stage, amrex::Real source_tag_thresh, amrex::Vector<int>& box_tag_arr);
        void computeVelocityCorrection(FlowField& stage);
        void correctVelocityandPressure(FlowField& stage, amrex::Real gamma, amrex::Real dt);
        void advanceTimeStep(FlowField& state_n, amrex::Real dt, amrex::Real Re, int rk_order, amrex::Real source_tag_thresh);
};

#endif