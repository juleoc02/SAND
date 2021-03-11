//
// Created by justin on 3/11/21.
//

#ifndef SAND_PARAMETERS_AND_COMPONENTS_H
#define SAND_PARAMETERS_AND_COMPONENTS_H
#include <deal.II/grid/tria_accessor.h>

template <int dim>
struct SolutionComponents
{
public:
    static constexpr unsigned int density = 0;
    static constexpr unsigned int displacement =1;
    static constexpr unsigned int unfiltered_density = 1+dim;
    static constexpr unsigned int displacement_multiplier = 2+dim;
    static constexpr unsigned int unfiltered_density_multiplier = 2+2*dim;
    static constexpr unsigned int density_lower_slack =  3+2*dim;
    static constexpr unsigned int density_lower_slack_multiplier =  4+2*dim;
    static constexpr unsigned int density_upper_slack =  5+2*dim;
    static constexpr unsigned int density_upper_slack_multiplier =  6+2*dim;
};

struct SolutionBlocks
{
public:
    static constexpr unsigned int density = 0;
    static constexpr unsigned int displacement = 1;
    static constexpr unsigned int unfiltered_density = 2;
    static constexpr unsigned int displacement_multiplier = 3;
    static constexpr unsigned int unfiltered_density_multiplier = 4;
    static constexpr unsigned int density_lower_slack = 5;
    static constexpr unsigned int density_lower_slack_multiplier = 6;
    static constexpr unsigned int density_upper_slack = 7;
    static constexpr unsigned int density_upper_slack_multiplier = 8;
};

struct BoundaryIds
{
public:
    static constexpr types::boundary_id no_force = 101;
    static constexpr types::boundary_id down_force = 102;
    static constexpr types::boundary_id held_still = 103;
};


#endif //SAND_PARAMETERS_AND_COMPONENTS_H
