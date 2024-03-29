//
// Created by justin on 3/11/21.
//

#ifndef SAND_INPUT_INFORMATION_H
#define SAND_INPUT_INFORMATION_H
#include <deal.II/grid/tria_accessor.h>
#include "parameters_and_components.h"

namespace SAND {
    using namespace dealii;

    namespace Input
    {
        constexpr double volume_percentage = .5;

        //geometry options
        constexpr unsigned int geometry_base = GeometryOptions::mbb;
        constexpr unsigned int dim = 2;
        constexpr unsigned int refinements =  2;

        //nonlinear algorithm options
        constexpr double initial_barrier_size = 25;
        constexpr double min_barrier_size = .00000;
        constexpr double fraction_to_boundary = .9;
        constexpr unsigned int max_steps=5;
        constexpr unsigned int barrier_reduction=BarrierOptions::loqo;
        constexpr double required_norm = .0001;

        //density filter options
        constexpr double filter_r = .251;

        //other options
        constexpr double density_penalty_exponent = 3;

        //output options
        constexpr bool output_full_preconditioned_matrix = false;
        constexpr bool output_full_matrix = false;
        constexpr bool output_parts_of_matrix = false;

        //Linear solver options
        constexpr unsigned int solver_choice = SolverOptions::inexact_K_with_exact_A_gmres;
        constexpr bool use_eisenstat_walker = false;
        constexpr double default_gmres_tolerance = 1e-6;


    }
}
#endif //SAND_INPUT_INFORMATION_H
