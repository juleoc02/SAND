//
// Created by justin on 3/11/21.
//

#ifndef SAND_INPUT_INFORMATION_H
#define SAND_INPUT_INFORMATION_H
#include <deal.II/grid/tria_accessor.h>

namespace SAND {
    using namespace dealii;

    namespace Input
    {
        constexpr double volume_percentage = .5;

        //geometry options
        constexpr unsigned int dim = 2;
        constexpr unsigned int height = 1;
        constexpr unsigned int width = 6;
        constexpr unsigned int depth = 0;
        constexpr unsigned int refinements = 2;

        //BC Options
        constexpr double downforce_x = 3;
        constexpr double downforce_y = 1;
        constexpr double downforce_size = .3;
        /* NEED SOME WAY FOR CORNER BCs */


        //nonlinear algorithm options
        constexpr double initial_barrier_size = 25;
        constexpr double min_barrier_size = .0000001;
        constexpr double fraction_to_boundary = .8;

        //density filter options
        constexpr double filter_r = .251;

        //other options
        constexpr unsigned int density_penalty_exponent = 3;

    }
}
#endif //SAND_INPUT_INFORMATION_H
