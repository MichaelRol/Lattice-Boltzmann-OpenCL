#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  bool condition = !obstacles[ii + jj* nx]
      && (cells[3*(nx * ny) + ii + jj*nx] - w1) > 0.f
      && (cells[6*(nx * ny) + ii + jj*nx] - w2) > 0.f
      && (cells[7*(nx * ny) + ii + jj*nx] - w2) > 0.f
      && jj == ny - 2;

    /* increase 'east-side' densities */
    cells[1*(nx * ny) + ii + jj*nx] += condition ? w1 : 0.f;
    cells[5*(nx * ny) + ii + jj*nx] += condition ? w2 : 0.f;
    cells[8*(nx * ny) + ii + jj*nx] += condition ? w2 : 0.f;
    /* decrease 'west-side' densities */
    cells[3*(nx * ny) + ii + jj*nx] -= condition ? w1 : 0.f;
    cells[6*(nx * ny) + ii + jj*nx] -= condition ? w2 : 0.f;
    cells[7*(nx * ny) + ii + jj*nx] -= condition ? w2 : 0.f;
    
}

kernel void propagate(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      const int nx, const int ny,
                      const float omega, 
                      local float* local_u,
                      global float* partial_u)
                      // float density, float accel)
{
  /* get column and row indices */
  const int ii = get_global_id(0);
  const int jj = get_global_id(1);
  // float accel_w1 = density * accel / 9.0;
  // float accel_w2 = density * accel / 36.0;

  // bool condition = !obstacles[ii + jj* nx]
  //     && (cells[3*(nx * ny) + ii + jj*nx] - accel_w1) > 0.f
  //     && (cells[6*(nx * ny) + ii + jj*nx] - accel_w2) > 0.f
  //     && (cells[7*(nx * ny) + ii + jj*nx] - accel_w2) > 0.f
  //     && jj == ny - 2;
  //   /* increase 'east-side' densities */
  // cells[1*(nx * ny) + ii + jj*nx] += condition ? accel_w1 : 0.f;
  // cells[5*(nx * ny) + ii + jj*nx] += condition ? accel_w2 : 0.f;
  // cells[8*(nx * ny) + ii + jj*nx] += condition ? accel_w2 : 0.f;
  // /* decrease 'west-side' densities */
  // cells[3*(nx * ny) + ii + jj*nx] -= condition ? accel_w1 : 0.f;
  // cells[6*(nx * ny) + ii + jj*nx] -= condition ? accel_w2 : 0.f;
  // cells[7*(nx * ny) + ii + jj*nx] -= condition ? accel_w2 : 0.f;

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  const int y_n = (jj + 1) % ny;
  const int x_e = (ii + 1) % nx;
  const int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  const int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  const float c_sq = 3.f; /* square of speed of sound */
  const float halfc_sqsq = 4.5f;
  const float halfc_sq = 1.5f;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */


  const int local_idX = get_local_id(0);
  const int local_idY = get_local_id(1);
  const int num_wrk_itemsX = get_local_size(0);
  const int num_wrk_itemsY = get_local_size(1);
  const int group_idX = get_group_id(0);
  const int group_idY = get_group_id(1);

  const float speed0 = cells[ii + jj*nx];
  const float speed1 = cells[(nx*ny) + x_w + jj*nx];
  const float speed2 = cells[2*(nx*ny) + ii + y_s*nx];
  const float speed3 = cells[3*(nx*ny) + x_e + jj*nx];
  const float speed4 = cells[4*(nx*ny) + ii + y_n*nx];
  const float speed5 = cells[5*(nx*ny) + x_w + y_s*nx];
  const float speed6 = cells[6*(nx*ny) + x_e + y_s*nx];
  const float speed7 = cells[7*(nx*ny) + x_e + y_n*nx];
  const float speed8 = cells[8*(nx*ny) + x_w + y_n*nx];

  /* compute local density total */
  float local_density = 0.f;
  local_density += speed0;
  local_density += speed1;
  local_density += speed2;
  local_density += speed3;
  local_density += speed4;
  local_density += speed5;
  local_density += speed6;
  local_density += speed7;
  local_density += speed8;
  /* compute x velocity component */
  const float u_x = (speed1
                  + speed5
                  + speed8
                  - (speed3
                      + speed6
                      + speed7))
                  / local_density;
    /* compute y velocity component */
  const float u_y = (speed2
                  + speed5
                  + speed6
                  - (speed4
                      + speed7
                      + speed8))
                  / local_density;
 

  /* velocity squared */
  const float u_sq = u_x * u_x + u_y * u_y;
  const float u_sqhalfc_sq = u_sq * halfc_sq;
  /* directional velocity components */
  float u[NSPEEDS];
  u[1] =   u_x;        /* east */
  u[2] =         u_y;  /* north */
  u[3] = - u_x;        /* west */
  u[4] =       - u_y;  /* south */
  u[5] =   u_x + u_y;  /* north-east */
  u[6] = - u_x + u_y;  /* north-west */
  u[7] = - u_x - u_y;  /* south-west */
  u[8] =   u_x - u_y;  /* south-east */

  /* equilibrium densities */
  float d_equ[NSPEEDS];
  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density
              * (1.f - u_sq * halfc_sq);
  /* axis speeds: weight w1 */
  d_equ[1] = w1 * local_density * (1.f + u[1] * c_sq
                                    + (u[1] * u[1]) * halfc_sqsq
                                    - u_sqhalfc_sq);
  d_equ[2] = w1 * local_density * (1.f + u[2] * c_sq
                                    + (u[2] * u[2]) * halfc_sqsq
                                    - u_sqhalfc_sq);
  d_equ[3] = w1 * local_density * (1.f + u[3] * c_sq
                                    + (u[3] * u[3]) * halfc_sqsq
                                    - u_sqhalfc_sq);
  d_equ[4] = w1 * local_density * (1.f + u[4] * c_sq
                                    + (u[4] * u[4]) * halfc_sqsq
                                    - u_sqhalfc_sq);
  /* diagonal speeds: weight w2 */
  d_equ[5] = w2 * local_density * (1.f + u[5] * c_sq
                                    + (u[5] * u[5]) * halfc_sqsq
                                    - u_sqhalfc_sq);
  d_equ[6] = w2 * local_density * (1.f + u[6] * c_sq
                                    + (u[6] * u[6]) * halfc_sqsq
                                    - u_sqhalfc_sq);
  d_equ[7] = w2 * local_density * (1.f + u[7] * c_sq
                                    + (u[7] * u[7]) * halfc_sqsq
                                    - u_sqhalfc_sq);
  d_equ[8] = w2 * local_density * (1.f + u[8] * c_sq
                                    + (u[8] * u[8]) * halfc_sqsq
                                    - u_sqhalfc_sq);

    /* relaxation step */
  tmp_cells[ii + jj*nx] = obstacles[ii + jj*nx] ? speed0 : speed0
                                          + omega
                                          * (d_equ[0] - speed0);
  tmp_cells[(nx*ny) + ii + jj*nx] =  obstacles[ii + jj*nx] ? speed3 : speed1
                                          + omega
                                          * (d_equ[1] - speed1);
  tmp_cells[2*(nx * ny) + ii + jj*nx] =  obstacles[ii + jj*nx] ? speed4 : speed2
                                          + omega
                                          * (d_equ[2] - speed2);
  tmp_cells[3*(nx * ny) + ii + jj*nx] = obstacles[ii + jj*nx] ? speed1 : speed3
                                          + omega
                                          * (d_equ[3] - speed3);
  tmp_cells[4*(nx * ny) + ii + jj*nx] =  obstacles[ii + jj*nx] ? speed2 : speed4
                                          + omega
                                          * (d_equ[4] - speed4);
  tmp_cells[5*(nx * ny) + ii + jj*nx] =  obstacles[ii + jj*nx] ? speed7 : speed5
                                          + omega
                                          * (d_equ[5] - speed5);
  tmp_cells[6*(nx * ny) + ii + jj*nx] =  obstacles[ii + jj*nx] ? speed8 : speed6
                                          + omega
                                          * (d_equ[6] - speed6);
  tmp_cells[7*(nx * ny) + ii + jj*nx] =  obstacles[ii + jj*nx] ? speed5 : speed7
                                          + omega
                                          * (d_equ[7] - speed7);
  tmp_cells[8*(nx * ny) + ii + jj*nx] =  obstacles[ii + jj*nx] ? speed6 : speed8
                                          + omega
                                          * (d_equ[8] - speed8);

  local_u[local_idX + (num_wrk_itemsX * local_idY)] = obstacles[ii + jj*nx] ? 0.f : (float)pow(((u_x * u_x) + (u_y * u_y)), 0.5f);
    
  for (int stride = (num_wrk_itemsX * num_wrk_itemsY)/2; stride>0; stride /=2)
  {
      barrier(CLK_LOCAL_MEM_FENCE);

      if (local_idX + (num_wrk_itemsX * local_idY) < stride){
        local_u[local_idX + (num_wrk_itemsX * local_idY)] +=  local_u[local_idX + stride + (num_wrk_itemsX * (local_idY))];
      }
  }

  if (local_idX == 0 && local_idY == 0){
    partial_u[group_idX + ((nx / num_wrk_itemsX) * group_idY)] = local_u[0]; 
  }
 
}

kernel void av_velocity(global float* cells,
                      global int* obstacles,
                      int nx, int ny,
                      local float* local_u,
                      global float* partial_u)
{
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int local_idX = get_local_id(0);
  int local_idY = get_local_id(1);
  int num_wrk_itemsX = get_local_size(0);
  int num_wrk_itemsY = get_local_size(1);
  int group_idX = get_group_id(0);
  int group_idY = get_group_id(1);
  
 /* ignore occupied cells */
  if (!obstacles[ii + jj*nx])
  {
    /* local density total */
    float local_density = 0.f;

    for (int kk = 0; kk < NSPEEDS; kk++)
    {

      local_density += cells[kk*(nx*ny) + ii + jj*nx];
    }

    /* x-component of velocity */
    float u_x = (cells[(nx * ny) + ii + jj*nx]
                  + cells[5*(nx * ny) + ii + jj*nx]
                  + cells[8*(nx * ny) + ii + jj*nx]
                  - (cells[3*(nx * ny) + ii + jj*nx]
                      + cells[6*(nx * ny) + ii + jj*nx]
                      + cells[7*(nx * ny) + ii + jj*nx]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (cells[2*(nx * ny) + ii + jj*nx]
                  + cells[5*(nx * ny) + ii + jj*nx]
                  + cells[6*(nx * ny) + ii + jj*nx]
                  - (cells[4*(nx * ny) + ii + jj*nx]
                      + cells[7*(nx * ny) + ii + jj*nx]
                      + cells[8*(nx * ny) + ii + jj*nx]))
                  / local_density;
    /* accumulate the norm of x- and y- velocity components */
    local_u[local_idX + (num_wrk_itemsX * local_idY)] = (float)pow(((u_x * u_x) + (u_y * u_y)), 0.5f);
    for (uint stride = (num_wrk_itemsX * num_wrk_itemsY)/2; stride>0; stride /=2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_idX + (num_wrk_itemsX * local_idY) < stride){
          local_u[local_idX + (num_wrk_itemsX * local_idY)] +=  local_u[local_idX + stride + (num_wrk_itemsX * (local_idY))];
        }
    }

    if (local_idX + (num_wrk_itemsX * local_idY) == 0){
      partial_u[group_idX + ((nx / num_wrk_itemsX) * group_idY)] = local_u[0]; 
    }
  }
}