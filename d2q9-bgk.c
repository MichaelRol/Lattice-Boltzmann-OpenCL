/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"

#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16


/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int   tot_cells;
  size_t size_wkg;   /*size of work groups*/
  size_t num_wkg;      /*no, of work groups*/
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  propagate;
  cl_kernel  av_velocity;
  // cl_kernel  global_reduction;

  cl_mem partial_cells;
  cl_mem partial_u;
  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
  // cl_mem av_vels
} t_ocl;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep_first(const t_param params, t_ocl ocl);
float timestep_second(const t_param params, t_ocl ocl);
int accelerate_flow_first(const t_param params, t_ocl ocl);
float propagate_first(const t_param params, t_ocl ocl);
int accelerate_flow_second(const t_param params, t_ocl ocl);
float propagate_second(const t_param params, t_ocl ocl);
// int global_reduction(const t_params params, t_ocl ocl);
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_ocl ocl, int tot_cells);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_ocl ocl, int tot_cells);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl    ocl;                 /* struct to hold OpenCL objects */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }
  
  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // Write cells to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(float) * NSPEEDS * params.nx * params.ny, cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);

  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.obstacles, CL_TRUE, 0,
    sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  for (int tt = 0; tt < params.maxIters; tt += 2)
  {
    av_vels[tt] = timestep_first(params, ocl)/(float)params.tot_cells;
    av_vels[tt+1] = timestep_second(params, ocl)/(float)params.tot_cells;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  
  err = clFinish(ocl.queue);
  checkError(err, "waiting for queue", __LINE__);

  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(float) * NSPEEDS * params.nx * params.ny, cells, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, ocl, params.tot_cells));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

float timestep_first(const t_param params, t_ocl ocl)
{
  cl_int err;
  accelerate_flow_first(params, ocl);
  float av = propagate_first(params, ocl);
  printf("time 1 tot_u: %f\ntot_cells: %d\n", av, tot_cells);

  return av;
}

float timestep_second(const t_param params, t_ocl ocl)
{
  cl_int err;
  accelerate_flow_second(params, ocl);
  float av = propagate_second(params, ocl);
    printf("time 2 tot_u: %f\ntot_cells: %d\n", av, tot_cells);

  return av;
}

int accelerate_flow_first(const t_param params, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
  checkError(err, "setting accelerate_flow arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;
}

int accelerate_flow_second(const t_param params, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
  checkError(err, "setting accelerate_flow arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;
}

float propagate_first(const t_param params, t_ocl ocl)
{
  float tot_u = 0.f;    /* accumulated magnitudes of velocity for each cell */
  float* sum_u = (float*)malloc(sizeof(float)  * params.num_wkg);
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.propagate, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting propagate arg 0", __LINE__);
  err = clSetKernelArg(ocl.propagate, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting propagate arg 1", __LINE__);
  err = clSetKernelArg(ocl.propagate, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting propagate arg 2", __LINE__);
  err = clSetKernelArg(ocl.propagate, 3, sizeof(cl_int), &params.nx);
  checkError(err, "setting propagate arg 3", __LINE__);
  err = clSetKernelArg(ocl.propagate, 4, sizeof(cl_int), &params.ny);
  checkError(err, "setting propagate arg 4", __LINE__);
  err = clSetKernelArg(ocl.propagate, 5, sizeof(int), &params.omega);
  checkError(err, "setting propagate arg 5", __LINE__);
  err = clSetKernelArg(ocl.propagate, 6, sizeof(float) * params.size_wkg, NULL);
  checkError(err, "setting propagate arg 6", __LINE__);
  err = clSetKernelArg(ocl.propagate, 7, sizeof(cl_mem), &ocl.partial_u);
  checkError(err, "setting propagate arg 7", __LINE__);
  // err = clSetKernelArg(ocl.propagate, 8, sizeof(cl_float), &params.density);
  // checkError(err, "setting propagate arg 8", __LINE__);
  // err = clSetKernelArg(ocl.propagate, 9, sizeof(cl_float), &params.accel);
  // checkError(err, "setting propagate arg 9", __LINE__);

  // Enqueue kernel

  size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
  size_t global[2] = {params.nx, params.ny};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.propagate,
                               2, NULL, global, local, 0, NULL, NULL);
  checkError(err, "enqueueing propagate kernel", __LINE__);

  // // Wait for kernel to finish
  // err = clFinish(ocl.queue);
  // checkError(err, "waiting for propagate kernel", __LINE__);

  err = clEnqueueReadBuffer(
    ocl.queue, ocl.partial_u, CL_TRUE, 0,
    sizeof(float) * params.num_wkg, sum_u, 0, NULL, NULL);
  checkError(err, "Reading partial_u", __LINE__);
    
  for (int x = 0; x < params.num_wkg; x++)
  {
      tot_u += sum_u[x];
  }

  free(sum_u);
  
  return tot_u;

}

float propagate_second(const t_param params, t_ocl ocl)
{
  float tot_u = 0.f;    /* accumulated magnitudes of velocity for each cell */

  float* sum_u = (float*)malloc(sizeof(float)  * params.num_wkg);
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.propagate, 0, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting propagate arg 0", __LINE__);
  err = clSetKernelArg(ocl.propagate, 1, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting propagate arg 1", __LINE__);
  err = clSetKernelArg(ocl.propagate, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting propagate arg 2", __LINE__);
  err = clSetKernelArg(ocl.propagate, 3, sizeof(cl_int), &params.nx);
  checkError(err, "setting propagate arg 3", __LINE__);
  err = clSetKernelArg(ocl.propagate, 4, sizeof(cl_int), &params.ny);
  checkError(err, "setting propagate arg 4", __LINE__);
  err = clSetKernelArg(ocl.propagate, 5, sizeof(int), &params.omega);
  checkError(err, "setting propagate arg 5", __LINE__);
  err = clSetKernelArg(ocl.propagate, 6, sizeof(float) * params.size_wkg, NULL);
  checkError(err, "setting propagate arg 6", __LINE__);
  err = clSetKernelArg(ocl.propagate, 7, sizeof(cl_mem), &ocl.partial_u);
  checkError(err, "setting propagate arg 7", __LINE__);
  // err = clSetKernelArg(ocl.propagate, 8, sizeof(cl_float), &params.density);
  // checkError(err, "setting propagate arg 8", __LINE__);
  // err = clSetKernelArg(ocl.propagate, 9, sizeof(cl_float), &params.accel);
  // checkError(err, "setting propagate arg 9", __LINE__);

  // Enqueue kernel

  size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
  size_t global[2] = {params.nx, params.ny};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.propagate,
                               2, NULL, global, local, 0, NULL, NULL);
  checkError(err, "enqueueing propagate kernel", __LINE__);

  err = clEnqueueReadBuffer(
    ocl.queue, ocl.partial_u, CL_TRUE, 0,
    sizeof(float) * params.num_wkg, sum_u, 0, NULL, NULL);
  checkError(err, "Reading partial_u", __LINE__);
    
  for (int x = 0; x < params.num_wkg; x++)
  {
      tot_u += sum_u[x];
  }

  free(sum_u);
  
  return tot_u;
}

// int global_reduction(const t_param params, t_ocl ocl) {
//   cl_int err;

//   // Set kernel arguments
//   err = clSetKernelArg(ocl.global_reduction, 0, sizeof(cl_mem), &ocl.partial_u);
//   checkError(err, "setting global_reduction arg 0", __LINE__);
//   err = clSetKernelArg(ocl.global_reduction, 1, sizeof(cl_mem), &ocl.av_vels);
//   checkError(err, "setting global_reduction arg 1", __LINE__);
//   err = clSetKernelArg(ocl.global_reduction, 2, sizeof(cl_int), &params.tot_cells);
//   checkError(err, "setting global_reduction arg 2", __LINE__);
//   err = clSetKernelArg(ocl.global_reduction, 3, sizeof(cl_int), &params.num_wkg);
//   checkError(err, "setting global_reduction arg 3", __LINE__);


//   // Enqueue kernel
//   size_t global[2] = {params.num_wkg, params.maxIters};
//   size_t local[1] = {params.num_wkg};
//   err = clEnqueueNDRangeKernel(ocl.queue, ocl.global_reduction,
//                                2, NULL, global, local, 0, NULL, NULL);
//   checkError(err, "enqueueing global_reduction kernel", __LINE__);

//   // Wait for kernel to finish
//   err = clFinish(ocl.queue);
//   checkError(err, "waiting for global_reduction kernel", __LINE__);

//   return EXIT_SUCCESS;
// }

float av_velocity(const t_param params, t_ocl ocl, int tot_cells)
{
  float tot_u = 0.f;    /* accumulated magnitudes of velocity for each cell */
  float* sum_u = (float*)malloc(sizeof(float)  * params.num_wkg);


  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.av_velocity, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting av_velocity arg 0", __LINE__);
  err = clSetKernelArg(ocl.av_velocity, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting av_velocity arg 1", __LINE__);
  err = clSetKernelArg(ocl.av_velocity, 2, sizeof(int), &params.nx);
  checkError(err, "setting av_velocity arg 2", __LINE__);
  err = clSetKernelArg(ocl.av_velocity, 3, sizeof(int), &params.ny);
  checkError(err, "setting av_velocity arg 3", __LINE__);
  err = clSetKernelArg(ocl.av_velocity, 4, sizeof(int) * params.size_wkg, NULL);
  checkError(err, "setting av_velocity arg 4", __LINE__);
  err = clSetKernelArg(ocl.av_velocity, 5, sizeof(cl_mem), &ocl.partial_u);
  checkError(err, "setting av_velocity arg 5", __LINE__);

  // Enqueue kernel
  size_t global[2] = {params.nx, params.ny};
  size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.av_velocity,
                               2, NULL, global, local, 0, NULL, NULL);
  checkError(err, "enqueueing av_velocity kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for av_velocity kernel", __LINE__);

  err = clEnqueueReadBuffer(
    ocl.queue, ocl.partial_u, CL_TRUE, 0,
    sizeof(float) * params.num_wkg, sum_u, 0, NULL, NULL);
  checkError(err, "Reading partial_u", __LINE__);
    
  for (int x = 0; x < params.num_wkg; x++)
  {
      tot_u += sum_u[x];
  }

  free(sum_u);
    printf("tot_u: %f\ntot_cells: %d\n", tot_u, tot_cells);

  return tot_u/(float)tot_cells;

}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (float*)malloc(sizeof(float) * NSPEEDS * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (float*)malloc(sizeof(float) * NSPEEDS * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr)[1*(params->nx * params->ny) + ii + jj*params->nx] = w1;
      (*cells_ptr)[2*(params->nx * params->ny) + ii + jj*params->nx] = w1;
      (*cells_ptr)[3*(params->nx * params->ny) + ii + jj*params->nx] = w1;
      (*cells_ptr)[4*(params->nx * params->ny) + ii + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr)[5*(params->nx * params->ny) + ii + jj*params->nx] = w2;
      (*cells_ptr)[6*(params->nx * params->ny) + ii + jj*params->nx] = w2;
      (*cells_ptr)[7*(params->nx * params->ny) + ii + jj*params->nx] = w2;
      (*cells_ptr)[8*(params->nx * params->ny) + ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }
  params->tot_cells = params->nx*params->ny;
  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    params->tot_cells--;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);


  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->propagate = clCreateKernel(ocl->program, "propagate", &err);
  checkError(err, "creating propagate kernel", __LINE__);
  ocl->av_velocity = clCreateKernel(ocl->program, "av_velocity", &err);
  checkError(err, "creating av_velocity kernel", __LINE__);
  // ocl->av_velocity = clCreateKernel(ocl->program, "global_reduction", &err);
  // checkError(err, "creating global_recutiom kernel", __LINE__);

  params->size_wkg = LOCAL_SIZE_X * LOCAL_SIZE_Y;
  params->num_wkg = (params->nx * params->ny) / (LOCAL_SIZE_X * LOCAL_SIZE_Y);
  

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * NSPEEDS * params->nx * params->ny, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(float) * NSPEEDS * params->nx * params->ny, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);
  ocl->partial_cells = clCreateBuffer(
    ocl->context, CL_MEM_WRITE_ONLY,
    sizeof(int) * params->num_wkg, NULL, &err);
  checkError(err, "creating buffer for partial_cells", __LINE__);
  ocl->partial_u = clCreateBuffer(
    ocl->context, CL_MEM_WRITE_ONLY,
    sizeof(float) * params->num_wkg, NULL, &err);
  checkError(err, "creating buffer for partial_u", __LINE__);


  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.propagate);
  // clReleaseKernel(ocl.global_reduction);
  clReleaseKernel(ocl.av_velocity);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);
  clReleaseMemObject(ocl.partial_cells);
  clReleaseMemObject(ocl.partial_u);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_ocl ocl, int tot_cells)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, ocl, tot_cells) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[kk*(params.nx * params.ny) + ii + jj*params.nx];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[kk*(params.nx * params.ny) + ii + jj*params.nx];
        }

        /* compute x velocity component */
        u_x = (cells[1*(params.nx * params.ny) + ii + jj*params.nx]
               + cells[5*(params.nx * params.ny) + ii + jj*params.nx]
               + cells[8*(params.nx * params.ny) + ii + jj*params.nx]
               - (cells[3*(params.nx * params.ny) + ii + jj*params.nx]
                  + cells[6*(params.nx * params.ny) + ii + jj*params.nx]
                  + cells[7*(params.nx * params.ny) + ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[2*(params.nx * params.ny) + ii + jj*params.nx]
               + cells[5*(params.nx * params.ny) + ii + jj*params.nx]
               + cells[6*(params.nx * params.ny) + ii + jj*params.nx]
               - (cells[4*(params.nx * params.ny) + ii + jj*params.nx]
                  + cells[7*(params.nx * params.ny) + ii + jj*params.nx]
                  + cells[8*(params.nx * params.ny) + ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
