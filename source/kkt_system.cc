//
// Created by justin on 2/17/21.
//
#include "../include/kkt_system.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/matrix_out.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include "../include/input_information.h"
#include "../include/sand_tools.h"

#include <iostream>
#include <algorithm>

// This problem initializes with a FESystem composed of 2×dim FE_Q(1) elements, and 8 FE_DGQ(0)  elements.
// The  piecewise  constant  functions  are  for  density-related  variables,and displacement-related variables are assigned to the FE_Q(1) elements.
namespace SAND {
    template<int dim>
    KktSystem<dim>::KktSystem()
            :
            dof_handler(triangulation),
            /*fe should have 1 FE_DGQ<dim>(0) element for density, dim FE_Q finite elements for displacement,
             * another dim FE_Q elements for the lagrange multiplier on the FE constraint, and 2 more FE_DGQ<dim>(0)
             * elements for the upper and lower bound constraints */
            fe_nine(FE_DGQ<dim>(0) ^ 5,
                    (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 2,
                    FE_DGQ<dim>(0) ^ 2,
                    FE_Nothing<dim>() ^ 1),
            fe_ten(FE_DGQ<dim>(0) ^ 5,
                   (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 2,
                   FE_DGQ<dim>(0) ^ 2,
                   FE_DGQ<dim>(0) ^ 1),
            density_ratio(Input::volume_percentage),
            density_penalty_exponent(Input::density_penalty_exponent),
            density_filter() {
        fe_collection.push_back(fe_nine);
        fe_collection.push_back(fe_ten);
    }


//A  function  used  once  at  the  beginning  of  the  program,  this  creates  a  matrix  H  so  that H* unfiltered density = filtered density

    template<int dim>
    void
    KktSystem<dim>::setup_filter_matrix() {

        density_filter.initialize(triangulation);
    }

    //This triangulation matches the problem description -
    // a 6-by-1 rectangle where a force will be applied in the top center.

    template<int dim>
    void
    KktSystem<dim>::create_triangulation() {
        std::vector<unsigned int> sub_blocks(2*dim+8, 0);


                sub_blocks[0]=0;
                sub_blocks[1]=1;
                sub_blocks[2]=2;
                sub_blocks[3]=3;
                sub_blocks[4]=4;
                for(int i=0; i<dim; i++)
                {
                    sub_blocks[5+i]=5;
                }

                for(int i=0; i<dim; i++)
                {
                    sub_blocks[5+dim+i]=6;
                }
                sub_blocks[5+2*dim]=7;
                sub_blocks[6+2*dim]=8;
                sub_blocks[7+2*dim]=9;

        if (Input::geometry_base == GeometryOptions::mbb) {
            const double width = 6;
            const unsigned int width_refine = 6;
            const double height = 1;
            const unsigned int height_refine = 1;
            const double depth = 1;
            const unsigned int depth_refine = 1;
            const double downforce_y = 1;
            const double downforce_x = 3;
            const double downforce_size = .3;

            if (dim == 2) {
                GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          {width_refine, height_refine},
                                                          Point<dim>(0, 0),
                                                          Point<dim>(width, height));

                triangulation.refine_global(Input::refinements);

                /*Set BCIDs   */
                for (const auto &cell: dof_handler.active_cell_iterators()) {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(1) - downforce_y) < 1e-12) {
                                if (std::fabs(center(0) - downforce_x) < downforce_size) {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                } else {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1)) <
                            1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }

                dof_handler.distribute_dofs(fe_collection);

                DoFRenumbering::component_wise(dof_handler,sub_blocks);
            } else if (dim == 3) {
                GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          {width_refine, height_refine, depth_refine},
                                                          Point<dim>(0, 0, 0),
                                                          Point<dim>(width, height, depth));

                triangulation.refine_global(Input::refinements);

                for (const auto &cell: dof_handler.active_cell_iterators()) {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(1) - downforce_y) < 1e-12) {
                                if (std::fabs(center(0) - downforce_x) < downforce_size) {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                } else {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1))
                            + std::abs(cell->vertex(vertex_number)(2)) < 1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }

                dof_handler.distribute_dofs(fe_collection);

                DoFRenumbering::component_wise(dof_handler,sub_blocks);

            } else {
                throw;
            }
        } else if (Input::geometry_base == GeometryOptions::l_shape) {
            const double width = 2;
            const unsigned int width_refine = 2;
            const double height = 2;
            const unsigned int height_refine = 2;
            const double depth = 1;
            const unsigned int depth_refine = 1;
            const double downforce_x = 2;
            const double downforce_y = 1;
            const double downforce_z = .5;
            const double downforce_size = .3;

            if (dim == 2) {
                GridGenerator::subdivided_hyper_L(triangulation,
                                                  {width_refine, height_refine},
                                                  Point<dim>(0, 0),
                                                  Point<dim>(width, height),
                                                  {-1, -1});

                triangulation.refine_global(Input::refinements);

                /*Set BCIDs   */
                for (const auto &cell: dof_handler.active_cell_iterators()) {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(0) - downforce_x) < 1e-12) {
                                if (std::fabs(center(1) - downforce_y) < downforce_size) {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                } else {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1)) <
                            1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }

                dof_handler.distribute_dofs(fe_collection);

                DoFRenumbering::component_wise(dof_handler,sub_blocks);

            } else if (dim == 3) {
                GridGenerator::subdivided_hyper_L(triangulation,
                                                  {width_refine, height_refine, depth_refine},
                                                  Point<dim>(0, 0, 0),
                                                  Point<dim>(width, height, depth),
                                                  {-1, -1, depth_refine});

                triangulation.refine_global(Input::refinements);

                /*Set BCIDs   */
                for (const auto &cell: dof_handler.active_cell_iterators()) {
                    cell->set_active_fe_index(0);
                    cell->set_material_id(MaterialIds::without_multiplier);
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            const auto center = cell->face(face_number)->center();

                            if (std::fabs(center(0) - downforce_x) < 1e-12) {
                                if (std::fabs(center(1) - downforce_y) < downforce_size) {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                    if (std::fabs(center(2) - downforce_z) < downforce_size) {
                                        cell->face(face_number)->set_boundary_id(BoundaryIds::down_force);
                                    } else {
                                        cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                    }
                                } else {
                                    cell->face(face_number)->set_boundary_id(BoundaryIds::no_force);
                                }
                            }
                        }
                    }
                    for (unsigned int vertex_number = 0;
                         vertex_number < GeometryInfo<dim>::vertices_per_cell;
                         ++vertex_number) {
                        if (std::abs(cell->vertex(vertex_number)(0)) + std::abs(cell->vertex(vertex_number)(1)) <
                            1e-10) {
                            cell->set_active_fe_index(1);
                            cell->set_material_id(MaterialIds::with_multiplier);
                        }
                    }
                }

                dof_handler.distribute_dofs(fe_collection);

                DoFRenumbering::component_wise(dof_handler,sub_blocks);
            } else {
                throw;
            }
        } else {
            throw;
        }

    }

// The  bottom  corners  are  kept  in  place  in  the  y  direction  -  the  bottom  left  also  in  the  x direction.
// Because deal.ii is formulated to enforce boundary conditions along regions of the boundary,
// we do this to ensure these BCs are only enforced at points.
    template<int dim>
    void
    KktSystem<dim>::setup_boundary_values() {
        if (Input::geometry_base == GeometryOptions::mbb) {
            if (dim == 2) {
                for (const auto &cell: dof_handler.active_cell_iterators()) {

                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number) {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                        vert(1) - 0) < 1e-12) {

                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    /*set bottom left BC*/
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 6) < 1e-12 && std::fabs(
                                        vert(1) - 0) < 1e-12) {
//                            const unsigned int x_displacement =
//                                    cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
//                            const unsigned int x_displacement_multiplier =
//                                    cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
//                            boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
//                            boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }
            } else if (dim == 3) {
                for (const auto &cell: dof_handler.active_cell_iterators()) {
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number) {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                        vert(1) - 0) < 1e-12 && ((std::fabs(
                                        vert(2) - 0) < 1e-12) || (std::fabs(
                                        vert(2) - 1) < 1e-12))) {


                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());

                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 6) < 1e-12 && std::fabs(
                                        vert(1) - 0) < 1e-12 && ((std::fabs(
                                        vert(2) - 0) < 1e-12) || (std::fabs(
                                        vert(2) - 1) < 1e-12))) {
//                              const unsigned int x_displacement =
//                                    cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
//                              const unsigned int x_displacement_multiplier =
//                                    cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());
//                              boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
//                              boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }
            } else {
                throw;
            }
        } else if (Input::geometry_base == GeometryOptions::l_shape) {
            if (dim == 2) {
                for (const auto &cell: dof_handler.active_cell_iterators()) {

                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number) {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find top left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                        vert(1) - 2) < 1e-12) {

                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    /*set bottom left BC*/
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                                /*Find top right corner*/
                                if (std::fabs(vert(0) - 1) < 1e-12 && std::fabs(
                                        vert(1) - 2) < 1e-12) {
                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }
            } else if (dim == 3) {
                for (const auto &cell: dof_handler.active_cell_iterators()) {
                    for (unsigned int face_number = 0;
                         face_number < GeometryInfo<dim>::faces_per_cell;
                         ++face_number) {
                        if (cell->face(face_number)->at_boundary()) {
                            for (unsigned int vertex_number = 0;
                                 vertex_number < GeometryInfo<dim>::vertices_per_cell;
                                 ++vertex_number) {
                                const auto vert = cell->vertex(vertex_number);
                                /*Find bottom left corner*/
                                if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                        vert(1) - 2) < 1e-12 && ((std::fabs(
                                        vert(2) - 0) < 1e-12) || (std::fabs(
                                        vert(2) - 1) < 1e-12))) {


                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());

                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                                /*Find bottom right corner*/
                                if (std::fabs(vert(0) - 1) < 1e-12 && std::fabs(
                                        vert(1) - 2) < 1e-12 && ((std::fabs(
                                        vert(2) - 0) < 1e-12) || (std::fabs(
                                        vert(2) - 1) < 1e-12))) {
                                    const unsigned int x_displacement =
                                            cell->vertex_dof_index(vertex_number, 0, cell->active_fe_index());
                                    const unsigned int y_displacement =
                                            cell->vertex_dof_index(vertex_number, 1, cell->active_fe_index());
                                    const unsigned int z_displacement =
                                            cell->vertex_dof_index(vertex_number, 2, cell->active_fe_index());
                                    const unsigned int x_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 3, cell->active_fe_index());
                                    const unsigned int y_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 4, cell->active_fe_index());
                                    const unsigned int z_displacement_multiplier =
                                            cell->vertex_dof_index(vertex_number, 5, cell->active_fe_index());
                                    boundary_values[x_displacement] = 0;
                                    boundary_values[y_displacement] = 0;
                                    boundary_values[z_displacement] = 0;
                                    boundary_values[x_displacement_multiplier] = 0;
                                    boundary_values[y_displacement_multiplier] = 0;
                                    boundary_values[z_displacement_multiplier] = 0;
                                }
                            }
                        }
                    }
                }
            } else {
                throw;
            }
        }


    }


    //This makes a giant 10-by-10 block matrix, and also sets up the necessary block vectors.  The
    // sparsity pattern for this matrix includes the sparsity pattern for the filter matrix. It also initializes
    // any block vectors we will use.
    template<int dim>
    void
    KktSystem<dim>::setup_block_system() {
        const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);

        //MAKE n_u and n_P*****************************************************************

        /*Setup 10 by 10 block matrix*/

        std::vector<unsigned int> block_component(10, 2);
        block_component[0] = 0;
        block_component[5] = 1;
        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

        const unsigned int n_p = dofs_per_block[0];
        const unsigned int n_u = dofs_per_block[1];
        std::cout << "n_p:  " << n_p << "   n_u:  " << n_u << std::endl;
        const std::vector<unsigned int> block_sizes = {n_p, n_p, n_p, n_p, n_p, n_u, n_u, n_p, n_p, 1};

        BlockDynamicSparsityPattern dsp(10, 10);

        for (unsigned int k = 0; k < 10; k++) {
            for (unsigned int j = 0; j < 10; j++) {
                dsp.block(j, k).reinit(block_sizes[j], block_sizes[k]);
            }
        }

        dsp.collect_sizes();

        Table<2, DoFTools::Coupling> coupling(2 * dim + 8, 2 * dim + 8);
//Coupling for density
        coupling[SolutionComponents::density<dim>][SolutionComponents::density<dim>] = DoFTools::always;

        for (unsigned int i = 0; i < dim; i++) {
            coupling[SolutionComponents::density<dim>][SolutionComponents::displacement<dim> + i] = DoFTools::always;
            coupling[SolutionComponents::displacement<dim> + i][SolutionComponents::density<dim>] = DoFTools::always;
        }

        coupling[SolutionComponents::density<dim>][SolutionComponents::unfiltered_density_multiplier<dim>] = DoFTools::always;
        coupling[SolutionComponents::unfiltered_density_multiplier<dim>][SolutionComponents::density<dim>] = DoFTools::always;

        for (unsigned int i = 0; i < dim; i++) {
            coupling[SolutionComponents::density<dim>][SolutionComponents::displacement_multiplier<dim> +
                                                       i] = DoFTools::always;
            coupling[SolutionComponents::displacement_multiplier<dim> +
                     i][SolutionComponents::density<dim>] = DoFTools::always;
        }

//Coupling for displacement
        for (unsigned int i = 0; i < dim; i++) {

            for (unsigned int k = 0; k < dim; k++) {
                coupling[SolutionComponents::displacement<dim> + i][SolutionComponents::displacement_multiplier<dim> +
                                                                    k] = DoFTools::always;
                coupling[SolutionComponents::displacement_multiplier<dim> + k][SolutionComponents::displacement<dim> +
                                                                               i] = DoFTools::always;
            }
        }

// coupling for unfiltered density
        coupling[SolutionComponents::unfiltered_density<dim>][SolutionComponents::density_lower_slack_multiplier<dim>] = DoFTools::always;
        coupling[SolutionComponents::density_lower_slack_multiplier<dim>][SolutionComponents::unfiltered_density<dim>] = DoFTools::always;

        coupling[SolutionComponents::unfiltered_density<dim>][SolutionComponents::density_upper_slack_multiplier<dim>] = DoFTools::always;
        coupling[SolutionComponents::density_upper_slack_multiplier<dim>][SolutionComponents::unfiltered_density<dim>] = DoFTools::always;

        coupling[SolutionComponents::unfiltered_density<dim>][SolutionComponents::unfiltered_density_multiplier<dim>] = DoFTools::always;
        coupling[SolutionComponents::unfiltered_density_multiplier<dim>][SolutionComponents::unfiltered_density<dim>] = DoFTools::always;




//        Coupling for lower slack
        coupling[SolutionComponents::density_lower_slack<dim>][SolutionComponents::density_lower_slack<dim>] = DoFTools::always;

        coupling[SolutionComponents::density_lower_slack<dim>][SolutionComponents::density_lower_slack_multiplier<dim>] = DoFTools::always;
        coupling[SolutionComponents::density_lower_slack_multiplier<dim>][SolutionComponents::density_lower_slack<dim>] = DoFTools::always;

//
        coupling[SolutionComponents::density_upper_slack<dim>][SolutionComponents::density_upper_slack<dim>] = DoFTools::always;
        coupling[SolutionComponents::density_upper_slack<dim>][SolutionComponents::density_upper_slack_multiplier<dim>] = DoFTools::always;
        coupling[SolutionComponents::density_upper_slack_multiplier<dim>][SolutionComponents::density_upper_slack<dim>] = DoFTools::always;

        coupling[SolutionComponents::density_upper_slack_multiplier<dim>][SolutionComponents::density_upper_slack_multiplier<dim>] = DoFTools::always;

        constraints.clear();
        constraints.close();

        DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp, constraints);

        //adds the row into the sparsity pattern for the total volume constraint
        for (const auto &cell: dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            dsp.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier).add(i, 0);
            dsp.block(SolutionBlocks::total_volume_multiplier, SolutionBlocks::density).add(0, i);
        }

        constraints.condense(dsp);
        sparsity_pattern.copy_from(dsp);
        setup_filter_matrix();
        //adds the row into the sparsity pattern for the total volume constraint
        sparsity_pattern.block(SolutionBlocks::unfiltered_density,
                               SolutionBlocks::unfiltered_density_multiplier).copy_from(
                density_filter.filter_sparsity_pattern);
        sparsity_pattern.block(SolutionBlocks::unfiltered_density_multiplier,
                               SolutionBlocks::unfiltered_density).copy_from(density_filter.filter_sparsity_pattern);

        std::ofstream out("sparsity.plt");
        sparsity_pattern.print_gnuplot(out);

        system_matrix.reinit(sparsity_pattern);


        linear_solution.reinit(block_sizes);
        system_rhs.reinit(block_sizes);

        for (unsigned int j = 0; j < 10; j++) {
            linear_solution.block(j).reinit(block_sizes[j]);
            system_rhs.block(j).reinit(block_sizes[j]);
        }

        linear_solution.collect_sizes();
        system_rhs.collect_sizes();
    }

    ///This  is  where  the  magic  happens.   The  equations  describing  the newtons method for finding 0s in the KKT conditions are implemented here.


    template<int dim>
    void
    KktSystem<dim>::assemble_block_system(const BlockVector<double> &state, const double barrier_size) {
        /*Remove any values from old iterations*/
        system_matrix.reinit(sparsity_pattern);
        system_matrix = 0;
        linear_solution = 0;
        system_rhs = 0;

        QGauss<dim> nine_quadrature(fe_nine.degree + 1);
        QGauss<dim> ten_quadrature(fe_ten.degree + 1);

        hp::QCollection<dim> q_collection;
        q_collection.push_back(nine_quadrature);
        q_collection.push_back(ten_quadrature);

        hp::FEValues<dim> hp_fe_values(fe_collection,
                                       q_collection,
                                       update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);

        QGauss<dim - 1> common_face_quadrature(fe_ten.degree + 1);

        FEFaceValues<dim> fe_nine_face_values(fe_nine,
                                              common_face_quadrature,
                                              update_JxW_values |
                                              update_gradients | update_values);
        FEFaceValues<dim> fe_ten_face_values(fe_ten,
                                             common_face_quadrature,
                                             update_normal_vectors |
                                             update_values);

        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);
        const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);
        const FEValuesExtractors::Scalar unfiltered_densities(SolutionComponents::unfiltered_density<dim>);
        const FEValuesExtractors::Vector displacement_multipliers(SolutionComponents::displacement_multiplier<dim>);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(
                SolutionComponents::unfiltered_density_multiplier<dim>);
        const FEValuesExtractors::Scalar density_lower_slacks(SolutionComponents::density_lower_slack<dim>);
        const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                SolutionComponents::density_lower_slack_multiplier<dim>);
        const FEValuesExtractors::Scalar density_upper_slacks(SolutionComponents::density_upper_slack<dim>);
        const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                SolutionComponents::density_upper_slack_multiplier<dim>);
        const FEValuesExtractors::Scalar total_volume_multiplier(
                SolutionComponents::total_volume_multiplier<dim>);

        const Functions::ConstantFunction<dim> lambda(1.), mu(1.);

        BlockVector<double> filtered_unfiltered_density_solution = state;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = state;
        filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(SolutionBlocks::unfiltered_density_multiplier) = 0;

        density_filter.filter_matrix.vmult(
                filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density),
                state.block(SolutionBlocks::unfiltered_density));
        density_filter.filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(
                                                    SolutionBlocks::unfiltered_density_multiplier),
                                            state.block(SolutionBlocks::unfiltered_density_multiplier));


        for (const auto &cell: dof_handler.active_cell_iterators()) {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            cell_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                               cell->get_fe().n_dofs_per_cell());
            cell_rhs.reinit(cell->get_fe().n_dofs_per_cell());

            const unsigned int n_q_points = fe_values.n_quadrature_points;

            std::vector<double> old_density_values(n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
            std::vector<double> old_displacement_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
                    n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
                    n_q_points);
            std::vector<double> old_displacement_multiplier_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
                    n_q_points);
            std::vector<double> old_lower_slack_multiplier_values(n_q_points);
            std::vector<double> old_upper_slack_multiplier_values(n_q_points);
            std::vector<double> old_lower_slack_values(n_q_points);
            std::vector<double> old_upper_slack_values(n_q_points);
            std::vector<double> old_unfiltered_density_values(n_q_points);
            std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> filtered_unfiltered_density_values(n_q_points);
            std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> lambda_values(n_q_points);
            std::vector<double> mu_values(n_q_points);

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

            cell_matrix = 0;
            cell_rhs = 0;
            local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(state,
                                                     old_density_values);
            fe_values[displacements].get_function_values(state,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(state,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    state, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    state, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    state, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    state, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    state, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    state, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    state, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    state, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    state, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    state, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    filter_adjoint_unfiltered_density_multiplier_solution,
                    filter_adjoint_unfiltered_density_multiplier_values);

            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);


                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const SymmetricTensor<2, dim> displacement_phi_j_symmgrad =
                                fe_values[displacements].symmetric_gradient(j,
                                                                            q_point);
                        const double displacement_phi_j_div =
                                fe_values[displacements].divergence(j, q_point);

                        const SymmetricTensor<2, dim> displacement_multiplier_phi_j_symmgrad =
                                fe_values[displacement_multipliers].symmetric_gradient(
                                        j, q_point);
                        const double displacement_multiplier_phi_j_div =
                                fe_values[displacement_multipliers].divergence(j,
                                                                               q_point);

                        const double density_phi_j = fe_values[densities].value(
                                j, q_point);

                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(j,
                                                                                                      q_point);
                        const double unfiltered_density_multiplier_phi_j = fe_values[unfiltered_density_multipliers].value(
                                j, q_point);


                        const double lower_slack_phi_j =
                                fe_values[density_lower_slacks].value(j, q_point);

                        const double upper_slack_phi_j =
                                fe_values[density_upper_slacks].value(j, q_point);

                        const double lower_slack_multiplier_phi_j =
                                fe_values[density_lower_slack_multipliers].value(j,
                                                                                 q_point);

                        const double upper_slack_multiplier_phi_j =
                                fe_values[density_upper_slack_multipliers].value(j,
                                                                                 q_point);

                        //Equation 0
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) *
                                (
                                        -density_phi_i * unfiltered_density_multiplier_phi_j

                                        - density_penalty_exponent * (density_penalty_exponent - 1)
                                          * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 2)
                                          * density_phi_i
                                          * density_phi_j
                                          * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (old_displacement_symmgrads[q_point] *
                                                  old_displacement_multiplier_symmgrads[q_point]))

                                        - density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                          * density_phi_i
                                          * (displacement_multiplier_phi_j_div * old_displacement_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               *
                                               (old_displacement_symmgrads[q_point] *
                                                displacement_multiplier_phi_j_symmgrad))

                                        - density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                          * density_phi_i
                                          * (displacement_phi_j_div * old_displacement_multiplier_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (old_displacement_multiplier_symmgrads[q_point] *
                                                  displacement_phi_j_symmgrad)));
                        //Equation 1

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                        -density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                        * density_phi_j
                                        * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                           * lambda_values[q_point]
                                           + 2 * mu_values[q_point]
                                             * (old_displacement_multiplier_symmgrads[q_point] *
                                                displacement_phi_i_symmgrad))

                                        - std::pow(old_density_values[q_point],
                                                   density_penalty_exponent)
                                          * (displacement_multiplier_phi_j_div * displacement_phi_i_div
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (displacement_multiplier_phi_j_symmgrad * displacement_phi_i_symmgrad))

                                );

                        //Equation 2 has to do with the filter, which is calculated elsewhere.
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                        -1 * unfiltered_density_phi_i * lower_slack_multiplier_phi_j
                                        + unfiltered_density_phi_i * upper_slack_multiplier_phi_j);

                        //Equation 3 - Primal Feasibility

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (

                                        -1 * density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                        * density_phi_j
                                        * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                           * lambda_values[q_point]
                                           + 2 * mu_values[q_point]
                                             * (old_displacement_symmgrads[q_point] *
                                                displacement_multiplier_phi_i_symmgrad))

                                        + -1 * std::pow(old_density_values[q_point],
                                                        density_penalty_exponent)
                                          * (displacement_phi_j_div * displacement_multiplier_phi_i_div
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               *
                                               (displacement_phi_j_symmgrad * displacement_multiplier_phi_i_symmgrad)));

                        //Equation 4 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * lower_slack_multiplier_phi_i *
                                (unfiltered_density_phi_j - lower_slack_phi_j);

                        //Equation 5 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * upper_slack_multiplier_phi_i * (
                                        -1 * unfiltered_density_phi_j - upper_slack_phi_j);

                        //Equation 6 - more primal feasibility - part with filter added later
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * unfiltered_density_multiplier_phi_i * (
                                        density_phi_j);

                        //Equation 7 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point) *
                                             (lower_slack_phi_i * lower_slack_multiplier_phi_j
                                              + lower_slack_phi_i * lower_slack_phi_j *
                                                old_lower_slack_multiplier_values[q_point] /
                                                old_lower_slack_values[q_point]);
                        //Equation 8 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point) *
                                             (upper_slack_phi_i * upper_slack_multiplier_phi_j
                                              + upper_slack_phi_i * upper_slack_phi_j *
                                                old_upper_slack_multiplier_values[q_point] /
                                                old_upper_slack_values[q_point]);
                    }

                }

            }

            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     cell_matrix, cell_rhs, true);

            constraints.distribute_local_to_global(
                    cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);


        }


        for (const auto &cell: dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();

            typename SparseMatrix<double>::iterator iter = density_filter.filter_matrix.begin(
                    i);
            for (; iter != density_filter.filter_matrix.end(i); iter++) {
                unsigned int j = iter->column();
                double value = iter->value() * cell->measure();

                system_matrix.block(SolutionBlocks::unfiltered_density_multiplier,
                                    SolutionBlocks::unfiltered_density).add(i, j, value);
                system_matrix.block(SolutionBlocks::unfiltered_density,
                                    SolutionBlocks::unfiltered_density_multiplier).add(j, i, value);
            }

            system_matrix.block(SolutionBlocks::total_volume_multiplier, SolutionBlocks::density).add(0, i,
                                                                                                      cell->measure());
            system_matrix.block(SolutionBlocks::density, SolutionBlocks::total_volume_multiplier).add(i, 0,
                                                                                                      cell->measure());
        }


        system_rhs = calculate_rhs(state, barrier_size);
        std::cout << "assembled" << std::endl;
    }

    template<int dim>
    double
    KktSystem<dim>::calculate_objective_value(const BlockVector<double> &state) const {
        /*Remove any values from old iterations*/

        QGauss<dim> nine_quadrature(fe_nine.degree + 1);
        QGauss<dim> ten_quadrature(fe_ten.degree + 1);

        hp::QCollection<dim> q_collection;
        q_collection.push_back(nine_quadrature);
        q_collection.push_back(ten_quadrature);

        hp::FEValues<dim> hp_fe_values(fe_collection,
                                       q_collection,
                                       update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);

        QGauss<dim - 1> common_face_quadrature(fe_ten.degree + 1);

        FEFaceValues<dim> fe_nine_face_values(fe_nine,
                                              common_face_quadrature,
                                              update_JxW_values |
                                              update_gradients | update_values);
        FEFaceValues<dim> fe_ten_face_values(fe_ten,
                                             common_face_quadrature,
                                             update_normal_vectors |
                                             update_values);

        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;

        const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);

        Tensor<1, dim> traction;
        traction[1] = -1;

        double objective_value = 0;
        for (const auto &cell: dof_handler.active_cell_iterators()) {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            const unsigned int n_q_points = fe_values.n_quadrature_points;
            const unsigned int n_face_q_points = common_face_quadrature.size();

            std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
            fe_values[displacements].get_function_values(
                    state, old_displacement_values);

            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id()
                                                              == BoundaryIds::down_force) {


                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            if (cell->material_id() == MaterialIds::without_multiplier) {
                                fe_nine_face_values.reinit(cell, face_number);
                                objective_value += traction
                                                   * fe_nine_face_values[displacements].value(i,
                                                                                              face_q_point)
                                                   * fe_nine_face_values.JxW(face_q_point);
                            } else {
                                fe_ten_face_values.reinit(cell, face_number);
                                objective_value += traction
                                                   * fe_ten_face_values[displacements].value(i,
                                                                                             face_q_point)
                                                   * fe_ten_face_values.JxW(face_q_point);
                            }
                        }
                    }
                }

            }
        }
        std::cout << "objective value: " << objective_value << std::endl;
        return objective_value;
    }


    //As the KKT System know which vectors correspond to the slack variables, the sum of the logs of the slacks is computed here for use in the filter.
    template<int dim>
    double
    KktSystem<dim>::calculate_barrier_distance(const BlockVector<double> &state) const {
        double barrier_distance_log_sum = 0;
        unsigned int vect_size = state.block(SolutionBlocks::density_lower_slack).size();
        for (unsigned int k = 0; k < vect_size; k++) {
            barrier_distance_log_sum += std::log(state.block(SolutionBlocks::density_lower_slack)[k]);
        }
        for (unsigned int k = 0; k < vect_size; k++) {
            barrier_distance_log_sum += std::log(state.block(SolutionBlocks::density_upper_slack)[k]);
        }
        std::cout << "barrier distance log sum: " << barrier_distance_log_sum << std::endl;
        return barrier_distance_log_sum;
    }

    template<int dim>
    double
    KktSystem<dim>::calculate_rhs_norm(const BlockVector<double> &state, const double barrier_size) const {
        return calculate_rhs(state, barrier_size).l2_norm();
    }


    //Feasibility conditions appear on the RHS of the linear system, so I compute the RHS to find it. Could probably be combined with the objective value finding part to make it faster.
    template<int dim>
    double
    KktSystem<dim>::calculate_feasibility(const BlockVector<double> &state, const double barrier_size) const {
        BlockVector<double> test_rhs = calculate_rhs(state, barrier_size);

        double norm = 0;

        for (unsigned int k = 0; k < state.block(SolutionBlocks::density_lower_slack).size(); k++) {
            norm += state.block(SolutionBlocks::density_lower_slack)[k] *
                    state.block(SolutionBlocks::density_lower_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_lower_slack)[k] *
                    state.block(SolutionBlocks::density_lower_slack_multiplier)[k];
        }
        for (unsigned int k = 0; k < state.block(SolutionBlocks::density_upper_slack).size(); k++) {
            norm += state.block(SolutionBlocks::density_upper_slack)[k] *
                    state.block(SolutionBlocks::density_upper_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_upper_slack)[k] *
                    state.block(SolutionBlocks::density_upper_slack_multiplier)[k];
        }

        std::cout << "pre-norm: " << norm << std::endl;
        norm += std::pow(test_rhs.block(SolutionBlocks::displacement).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::density).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::displacement_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::total_volume_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::density_upper_slack_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::density_lower_slack_multiplier).l2_norm(), 2);

        std::cout << "norm: " << norm << std::endl;
        return norm;
    }

    template<int dim>
    double
    KktSystem<dim>::calculate_convergence(const BlockVector<double> &state) const {
        BlockVector<double> test_rhs = calculate_rhs(state, Input::min_barrier_size);
        double norm = 0;

        norm += std::pow(test_rhs.block(SolutionBlocks::displacement).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::density).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::displacement_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::unfiltered_density_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::total_volume_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::density_upper_slack_multiplier).l2_norm(), 2);
        norm += std::pow(test_rhs.block(SolutionBlocks::density_lower_slack_multiplier).l2_norm(), 2);

        for (unsigned int k = 0; k < state.block(SolutionBlocks::density_upper_slack).size(); k++) {
            norm += state.block(SolutionBlocks::density_upper_slack)[k] *
                    state.block(SolutionBlocks::density_upper_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_upper_slack)[k] *
                    state.block(SolutionBlocks::density_upper_slack_multiplier)[k];
        }
        for (unsigned int k = 0; k < state.block(SolutionBlocks::density_lower_slack).size(); k++) {
            norm += state.block(SolutionBlocks::density_lower_slack)[k] *
                    state.block(SolutionBlocks::density_lower_slack_multiplier)[k]
                    * state.block(SolutionBlocks::density_lower_slack)[k] *
                    state.block(SolutionBlocks::density_lower_slack_multiplier)[k];
        }
        norm = std::pow(norm, .5);

        std::cout << "l2 norm: " << system_rhs.l2_norm() << std::endl;
        std::cout << "KKT norm: " << norm << std::endl;
        return norm;
    }

    template<int dim>
    BlockVector<double>
    KktSystem<dim>::calculate_rhs(const BlockVector<double> &state, const double barrier_size) const {
        BlockVector<double> test_rhs;
        test_rhs = system_rhs;
        test_rhs = 0;


        QGauss<dim> nine_quadrature(fe_nine.degree + 1);
        QGauss<dim> ten_quadrature(fe_ten.degree + 1);

        hp::QCollection<dim> q_collection;
        q_collection.push_back(nine_quadrature);
        q_collection.push_back(ten_quadrature);

        hp::FEValues<dim> hp_fe_values(fe_collection,
                                       q_collection,
                                       update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);

        QGauss<dim - 1> common_face_quadrature(fe_ten.degree + 1);

        FEFaceValues<dim> fe_nine_face_values(fe_nine,
                                              common_face_quadrature,
                                              update_JxW_values |
                                              update_gradients | update_values);
        FEFaceValues<dim> fe_ten_face_values(fe_ten,
                                             common_face_quadrature,
                                             update_normal_vectors |
                                             update_values);

        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        const FEValuesExtractors::Scalar densities(SolutionComponents::density<dim>);
        const FEValuesExtractors::Vector displacements(SolutionComponents::displacement<dim>);
        const FEValuesExtractors::Scalar unfiltered_densities(SolutionComponents::unfiltered_density<dim>);
        const FEValuesExtractors::Vector displacement_multipliers(SolutionComponents::displacement_multiplier<dim>);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(
                SolutionComponents::unfiltered_density_multiplier<dim>);
        const FEValuesExtractors::Scalar density_lower_slacks(SolutionComponents::density_lower_slack<dim>);
        const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                SolutionComponents::density_lower_slack_multiplier<dim>);
        const FEValuesExtractors::Scalar density_upper_slacks(SolutionComponents::density_upper_slack<dim>);
        const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                SolutionComponents::density_upper_slack_multiplier<dim>);
        const FEValuesExtractors::Scalar total_volume_multiplier(
                SolutionComponents::total_volume_multiplier<dim>);


        const unsigned int n_face_q_points = common_face_quadrature.size();

        const Functions::ConstantFunction<dim> lambda(1.), mu(1.);

        BlockVector<double> filtered_unfiltered_density_solution = state;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = state;
        filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(SolutionBlocks::unfiltered_density_multiplier) = 0;

        density_filter.filter_matrix.vmult(
                filtered_unfiltered_density_solution.block(SolutionBlocks::unfiltered_density),
                state.block(SolutionBlocks::unfiltered_density));
        density_filter.filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(
                                                    SolutionBlocks::unfiltered_density_multiplier),
                                            state.block(SolutionBlocks::unfiltered_density_multiplier));
        const double old_volume_multiplier = state.block(SolutionBlocks::total_volume_multiplier)[0];

        for (const auto &cell: dof_handler.active_cell_iterators()) {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            cell_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                               cell->get_fe().n_dofs_per_cell());
            cell_rhs.reinit(cell->get_fe().n_dofs_per_cell());

            const unsigned int n_q_points = fe_values.n_quadrature_points;

            std::vector<double> old_density_values(n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
            std::vector<double> old_displacement_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
                    n_q_points);
            std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
                    n_q_points);
            std::vector<double> old_displacement_multiplier_divs(n_q_points);
            std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
                    n_q_points);
            std::vector<double> old_lower_slack_multiplier_values(n_q_points);
            std::vector<double> old_upper_slack_multiplier_values(n_q_points);
            std::vector<double> old_lower_slack_values(n_q_points);
            std::vector<double> old_upper_slack_values(n_q_points);
            std::vector<double> old_unfiltered_density_values(n_q_points);
            std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> filtered_unfiltered_density_values(n_q_points);
            std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);
            std::vector<double> lambda_values(n_q_points);
            std::vector<double> mu_values(n_q_points);

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

            cell_matrix = 0;
            cell_rhs = 0;
            local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());

            cell->get_dof_indices(local_dof_indices);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(state,
                                                     old_density_values);
            fe_values[displacements].get_function_values(state,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(state,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    state, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    state, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    state, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    state, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    state, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    state, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    state, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    state, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    state, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    state, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    filter_adjoint_unfiltered_density_multiplier_solution,
                    filter_adjoint_unfiltered_density_multiplier_values);


            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);


                    //rhs eqn 0
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    -1 * density_penalty_exponent *
                                    std::pow(old_density_values[q_point], density_penalty_exponent - 1) * density_phi_i
                                    * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_symmgrads[q_point]
                                                                   * old_displacement_multiplier_symmgrads[q_point]))
                                    - density_phi_i * old_unfiltered_density_multiplier_values[q_point]
                                    + old_volume_multiplier * density_phi_i
                            );

                    //rhs eqn 1 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    -1 * std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                                                                   * displacement_phi_i_symmgrad))
                            );

                    //rhs eqn 2
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    unfiltered_density_phi_i *
                                    filter_adjoint_unfiltered_density_multiplier_values[q_point]
                                    + unfiltered_density_phi_i * old_upper_slack_multiplier_values[q_point]
                                    + -1 * unfiltered_density_phi_i * old_lower_slack_multiplier_values[q_point]
                            );




                    //rhs eqn 3 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    -1 * std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (displacement_multiplier_phi_i_symmgrad
                                                                   * old_displacement_symmgrads[q_point]))
                            );

                    //rhs eqn 4
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) *
                            (-1 * lower_slack_multiplier_phi_i
                             * (old_unfiltered_density_values[q_point] - old_lower_slack_values[q_point])
                            );

                    //rhs eqn 5
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    -1 * upper_slack_multiplier_phi_i
                                    * (1 - old_unfiltered_density_values[q_point]
                                       - old_upper_slack_values[q_point]));

                    //rhs eqn 6
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    -1 * unfiltered_density_multiplier_phi_i
                                    * (old_density_values[q_point] - filtered_unfiltered_density_values[q_point])
                            );

                    //rhs eqn 7
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) *
                            (lower_slack_phi_i *
                             (old_lower_slack_multiplier_values[q_point] -
                              barrier_size / old_lower_slack_values[q_point]));

                    //rhs eqn 8
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) *
                            (upper_slack_phi_i *
                             (old_upper_slack_multiplier_values[q_point] -
                              barrier_size / old_upper_slack_values[q_point]));

                }

            }


            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary() && cell->face(
                        face_number)->boundary_id() == BoundaryIds::down_force) {
                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            if (cell->material_id() == MaterialIds::without_multiplier) {
                                fe_nine_face_values.reinit(cell, face_number);
                                cell_rhs(i) += -1
                                               * traction
                                               * fe_nine_face_values[displacements].value(i,
                                                                                          face_q_point)
                                               * fe_nine_face_values.JxW(face_q_point);

                                cell_rhs(i) += -1 * traction
                                               * fe_nine_face_values[displacement_multipliers].value(
                                        i, face_q_point)
                                               * fe_nine_face_values.JxW(face_q_point);
                            } else {
                                fe_ten_face_values.reinit(cell, face_number);
                                cell_rhs(i) += -1
                                               * traction
                                               * fe_ten_face_values[displacements].value(i,
                                                                                         face_q_point)
                                               * fe_ten_face_values.JxW(face_q_point);

                                cell_rhs(i) += -1 * traction
                                               * fe_ten_face_values[displacement_multipliers].value(
                                        i, face_q_point)
                                               * fe_ten_face_values.JxW(face_q_point);
                            }
                        }
                    }
                }
            }


            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     cell_matrix, cell_rhs, true);

            constraints.distribute_local_to_global(
                    cell_rhs, local_dof_indices, test_rhs);

        }

        double total_volume = 0;
        double goal_volume = 0;
        for (const auto &cell: dof_handler.active_cell_iterators()) {
            total_volume += cell->measure() * state.block(SolutionBlocks::density)[cell->active_cell_index()];
            goal_volume += cell->measure() * Input::volume_percentage;
        }

        test_rhs.block(SolutionBlocks::total_volume_multiplier)[0] = goal_volume - total_volume;

        return test_rhs;

    }

    template<int dim>
    BlockVector<double>
    KktSystem<dim>::solve(const BlockVector<double> &state, double barrier_size) {
        constraints.condense(system_matrix);
        double gmres_tolerance;
        if (Input::use_eisenstat_walker) {
            gmres_tolerance = std::max(
                    std::min(
                            .1 * system_rhs.l2_norm() / (initial_rhs_error),
                            .001
                    ),
                    Input::default_gmres_tolerance);
        } else {
            gmres_tolerance = Input::default_gmres_tolerance;
        }
        SolverControl solver_control(10000, gmres_tolerance * system_rhs.l2_norm());
        TopOptSchurPreconditioner<dim> preconditioner(system_matrix);

        for(unsigned int i=0; i<10; i++)
        {
            for(unsigned int j=0; j<10; j++)
            {
                std::cout << i << ", " << j << std::endl;
                std::cout << system_matrix.block(i,j).frobenius_norm() << std::endl;
            }
        }



        switch (Input::solver_choice) {
            case SolverOptions::direct_solve: {
                SparseDirectUMFPACK A_direct;
                A_direct.initialize(system_matrix);
                A_direct.vmult(linear_solution, system_rhs);
                break;
            }
            case SolverOptions::exact_preconditioner_with_gmres: {
                preconditioner.initialize(system_matrix, boundary_values, dof_handler, barrier_size, state);
                SolverFGMRES<BlockVector<double>> A_fgmres(solver_control);
                A_fgmres.solve(system_matrix, linear_solution, system_rhs, preconditioner);
                std::cout << solver_control.last_step() << " steps to solve with GMRES" << std::endl;
                break;
            }
            case SolverOptions::inexact_K_with_exact_A_gmres: {
                preconditioner.initialize(system_matrix, boundary_values, dof_handler, barrier_size, state);
                SolverFGMRES<BlockVector<double>> B_fgmres(solver_control);
                B_fgmres.solve(system_matrix, linear_solution, system_rhs, preconditioner);
                std::cout << solver_control.last_step() << " steps to solve with GMRES" << std::endl;
                break;
            }
            case SolverOptions::inexact_K_with_inexact_A_gmres: {
                preconditioner.initialize(system_matrix, boundary_values, dof_handler, barrier_size, state);
                SolverFGMRES<BlockVector<double>> C_fgmres(solver_control);
                C_fgmres.solve(system_matrix, linear_solution, system_rhs, preconditioner);
                std::cout << solver_control.last_step() << " steps to solve with GMRES" << std::endl;
                break;
            }
            default:
                throw;
        }

        std::cout << "rhs" << std::endl;
        system_rhs.print(std::cout);
        std::cout << std::endl;

        preconditioner.initialize(system_matrix, boundary_values, dof_handler, barrier_size, state);
        preconditioner.vmult(linear_solution,system_rhs);

        std::cout << "linear solution" << std::endl;
        linear_solution.print(std::cout);
        std::cout << std::endl;

//        switch (Input::solver_choice) {
//            case SolverOptions::direct_solve: {
//                SparseDirectUMFPACK A_direct;
//                A_direct.initialize(system_matrix);
//                A_direct.vmult(linear_solution, system_rhs);
//                break;
//            }
//            case SolverOptions::exact_preconditioner_with_gmres: {
//                preconditioner.initialize(system_matrix, boundary_values, dof_handler, barrier_size, state);
//                SolverFGMRES<BlockVector<double>> A_fgmres(solver_control);
//                A_fgmres.solve(system_matrix, linear_solution, system_rhs, preconditioner);
//                std::cout << solver_control.last_step() << " steps to solve with GMRES" << std::endl;
//                break;
//            }
//            case SolverOptions::inexact_K_with_exact_A_gmres: {
//                preconditioner.initialize(system_matrix, boundary_values, dof_handler, barrier_size, state);
//                SolverFGMRES<BlockVector<double>> B_fgmres(solver_control);
//                B_fgmres.solve(system_matrix, linear_solution, system_rhs, preconditioner);
//                std::cout << solver_control.last_step() << " steps to solve with GMRES" << std::endl;
//                break;
//            }
//            case SolverOptions::inexact_K_with_inexact_A_gmres: {
//                preconditioner.initialize(system_matrix, boundary_values, dof_handler, barrier_size, state);
//                SolverFGMRES<BlockVector<double>> C_fgmres(solver_control);
//                C_fgmres.solve(system_matrix, linear_solution, system_rhs, preconditioner);
//                std::cout << solver_control.last_step() << " steps to solve with GMRES" << std::endl;
//                break;
//            }
//            default:
//                throw;
//        }

        constraints.distribute(linear_solution);

        if (Input::output_parts_of_matrix) {
            preconditioner.print_stuff(system_matrix);
        }

        if (Input::output_full_preconditioned_matrix) {
//            FullMatrix<double> preconditioned_full_mat(system_matrix.n(), system_matrix.n());
//            const auto op_preconditioned_full_mat = linear_operator(preconditioner) * linear_operator(system_matrix);
//            build_matrix_element_by_element(op_preconditioned_full_mat, preconditioned_full_mat);
//            print_matrix("preconditioned_full_block_matrix.csv",preconditioned_full_mat);
        }

        if (Input::output_full_matrix) {
//            const unsigned int vec_size = system_matrix.n();
//            FullMatrix<double> full_mat(vec_size, vec_size);
//            build_matrix_element_by_element(system_matrix,full_mat);
//            std::ofstream Mat("full_block_matrix.csv");
//            for (unsigned int i = 0; i < vec_size; i++) {
//                Mat << full_mat(i, 0);
//                for (unsigned int j = 1; j < vec_size; j++) {
//                    Mat << "," << full_mat(i, j);
//                }
//                Mat << "\n";
//            }
//            Mat.close();
        }


        std::cout << "solution" << std::endl;
        linear_solution.print(std::cout);
        std::cout << std::endl;

        return linear_solution;
    }

    template<int dim>
    void
    KktSystem<dim>::calculate_initial_rhs_error() {
        initial_rhs_error = system_rhs.l2_norm();
    }

    template<int dim>
    BlockVector<double>
    KktSystem<dim>::get_initial_state() {

        std::vector<unsigned int> block_component(10, 2);
        block_component[0] = 0;
        block_component[5] = 1;
        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_p = dofs_per_block[0];
        const unsigned int n_u = dofs_per_block[1];
        const std::vector<unsigned int> block_sizes = {n_p, n_p, n_p, n_p, n_p, n_u, n_u, n_p, n_p, 1};

        BlockVector<double> state(block_sizes);
        {
            using namespace SolutionBlocks;
            state.block(density).add(density_ratio);
            state.block(unfiltered_density).add(density_ratio);
            state.block(unfiltered_density_multiplier)
                    .add(density_ratio);
            state.block(density_lower_slack).add(density_ratio);
            state.block(density_lower_slack_multiplier).add(50);
            state.block(density_upper_slack).add(1 - density_ratio);
            state.block(density_upper_slack_multiplier).add(50);
            state.block(total_volume_multiplier).add(1);
            state.block(displacement).add(0);
            state.block(displacement_multiplier).add(0);
        }
        return state;

    }

    template<int dim>
    void
    KktSystem<dim>::output(const BlockVector<double> &state, const unsigned int j) const {
        std::vector<std::string> solution_names(1, "low_slack_multiplier");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
                1, DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("upper_slack_multiplier");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("low_slack");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("upper_slack");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("unfiltered_density");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        for (unsigned int i = 0; i < dim; i++) {
            solution_names.emplace_back("displacement");
            data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }
        for (unsigned int i = 0; i < dim; i++) {
            solution_names.emplace_back("displacement_multiplier");
            data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }
        solution_names.emplace_back("density_multiplier");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("density");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("volume_multiplier");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(state, solution_names,
                                 DataOut<dim>::type_dof_data, data_component_interpretation);
        data_out.build_patches();
        std::ofstream output("solution" + std::to_string(j) + ".vtk");
        data_out.write_vtk(output);

    }

    template<>
    void
    KktSystem<2>::output_stl(const BlockVector<double> &state) {
        double height = .25;
        const int dim = 2;
        std::ofstream stlfile;
        stlfile.open("bridge.stl");
        stlfile << "solid bridge\n" << std::scientific;

        for (const auto &cell: dof_handler.active_cell_iterators()) {
            if (state.block(
                    SolutionBlocks::density)[cell->active_cell_index()] > 0.5) {
                const Tensor<1, dim> edge_directions[2] = {cell->vertex(1) -
                                                           cell->vertex(0),
                                                           cell->vertex(2) -
                                                           cell->vertex(0)};
                const Tensor<2, dim> edge_tensor(
                        {{edge_directions[0][0], edge_directions[0][1]},
                         {edge_directions[1][0], edge_directions[1][1]}});
                const bool is_right_handed_cell = (determinant(edge_tensor) > 0);
                if (is_right_handed_cell) {
                    /* Write one side at z = 0. */
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                    /* Write one side at z = height. */
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                } else /* The cell has a left-handed set up */
                {
                    /* Write one side at z = 0. */
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                    /* Write one side at z = height. */
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(0)[0] << " "
                            << cell->vertex(0)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                    stlfile << "   facet normal " << 0.000000e+00 << " "
                            << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                    stlfile << "      outer loop\n";
                    stlfile << "         vertex " << cell->vertex(1)[0] << " "
                            << cell->vertex(1)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(2)[0] << " "
                            << cell->vertex(2)[1] << " " << height << "\n";
                    stlfile << "         vertex " << cell->vertex(3)[0] << " "
                            << cell->vertex(3)[1] << " " << height << "\n";
                    stlfile << "      endloop\n";
                    stlfile << "   endfacet\n";
                }
                for (unsigned int face_number = 0;
                     face_number < GeometryInfo<dim>::faces_per_cell;
                     ++face_number) {
                    const typename DoFHandler<dim>::face_iterator face =
                            cell->face(face_number);
                    if ((face->at_boundary()) ||
                        (!face->at_boundary() &&
                         (state.block(
                                 SolutionBlocks::density)[cell->neighbor(face_number)->active_cell_index()] <
                          0.5))) {
                        const Tensor<1, dim> normal_vector =
                                (face->center() - cell->center());
                        const double normal_norm = normal_vector.norm();
                        if ((face->vertex(0)[0] - face->vertex(0)[0]) *
                            (face->vertex(1)[1] - face->vertex(0)[1]) *
                            0.000000e+00 +
                            (face->vertex(0)[1] - face->vertex(0)[1]) * (0 - 0) *
                            normal_vector[0] +
                            (height - 0) *
                            (face->vertex(1)[0] - face->vertex(0)[0]) *
                            normal_vector[1] -
                            (face->vertex(0)[0] - face->vertex(0)[0]) * (0 - 0) *
                            normal_vector[1] -
                            (face->vertex(0)[1] - face->vertex(0)[1]) *
                            (face->vertex(1)[0] - face->vertex(0)[0]) *
                            normal_vector[0] -
                            (height - 0) *
                            (face->vertex(1)[1] - face->vertex(0)[1]) * 0 >
                            0) {
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " " << height
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " " << height
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " " << height
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                        } else {
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " " << height
                                    << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " " << height
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " "
                                    << 0.000000e+00 << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " " << height
                                    << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                        }
                    }
                }
            }
        }
        stlfile << "endsolid bridge";
    }


    template<>
    void
    KktSystem<3>::output_stl(const BlockVector<double> &state)
    {
        std::ofstream stlfile;
        stlfile.open("bridge.stl");
        stlfile << "solid bridge\n" << std::scientific;
        const int dim = 3;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (state.block(
                    SolutionBlocks::unfiltered_density)[cell->active_cell_index()] > 0.5)
            {
                for (const auto n : cell->face_indices())
                {
                    bool create_boundary = false;
                    if (cell->at_boundary(n))
                    {
                        create_boundary = true;
                    }
                    else if (state.block(
                            SolutionBlocks::unfiltered_density)[cell->neighbor(n)->active_cell_index()] <= 0.5)
                    {
                        create_boundary = true;
                    }

                    if (create_boundary)
                    {
                        const auto face = cell->face(n);
                        const Tensor<1,dim> normal_vector = face->center() -
                                                          cell->center();
                        double normal_norm = normal_vector.norm();
                        const Tensor<1,dim> edge_vectors_1 = face->vertex(1) - face->vertex(0);
                        const Tensor<1,dim> edge_vectors_2 = face->vertex(2) - face->vertex(0);

                        const Tensor<2, dim> edge_tensor (
                                 {{edge_vectors_1[0], edge_vectors_1[1],edge_vectors_1[2]},
                                 {edge_vectors_2[0], edge_vectors_2[1],edge_vectors_2[2]},
                                 {normal_vector[0], normal_vector[1], normal_vector[2]}});
                        const bool is_right_handed_cell = (determinant(edge_tensor) > 0);

                        if (is_right_handed_cell)
                        {
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " "
                                    << face->vertex(0)[2] << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " "
                                    << face->vertex(1)[2] << "\n";
                            stlfile << "         vertex " << face->vertex(2)[0]
                                    << " " << face->vertex(2)[1] << " "
                                    << face->vertex(2)[2] << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " " << face->vertex(1)[2]
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(3)[0]
                                    << " " << face->vertex(3)[1] << " " << face->vertex(3)[2]
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(2)[0]
                                    << " " << face->vertex(2)[1] << " "
                                    << face->vertex(2)[2] << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                        }
                        else
                        {
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(0)[0]
                                    << " " << face->vertex(0)[1] << " "
                                    << face->vertex(0)[2] << "\n";
                            stlfile << "         vertex " << face->vertex(2)[0]
                                    << " " << face->vertex(2)[1] << " "
                                    << face->vertex(2)[2] << "\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " "
                                    << face->vertex(1)[2] << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                            stlfile << "   facet normal "
                                    << normal_vector[0] / normal_norm << " "
                                    << normal_vector[1] / normal_norm << " "
                                    << normal_vector[2] / normal_norm << "\n";
                            stlfile << "      outer loop\n";
                            stlfile << "         vertex " << face->vertex(1)[0]
                                    << " " << face->vertex(1)[1] << " " << face->vertex(1)[2]
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(2)[0]
                                    << " " << face->vertex(2)[1] << " " << face->vertex(2)[2]
                                    << "\n";
                            stlfile << "         vertex " << face->vertex(3)[0]
                                    << " " << face->vertex(3)[1] << " "
                                    << face->vertex(3)[2] << "\n";
                            stlfile << "      endloop\n";
                            stlfile << "   endfacet\n";
                        }

                    }

                }
            }
        }
    }
}

template class SAND::KktSystem<2>;
template class SAND::KktSystem<3>;
