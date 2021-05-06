#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <vector>
#include <iostream>

// GPU map calls
#include "gpuKernels_util.cu"

#define DISCRETE_GPU

// for computation
#define COUNT 4

// thread block size
#define THREADS_PER_BLOCK 128
#define BLOCK_SIZE THREADS_PER_BLOCK
#define NUM_PHYSICAL_CORES 15

// maximum number of threads *including* the main thread
#define MAX_THREADS 16

// input processing
#define NUM_ARGS 6

pthread_barrier_t barrier;

void inline checkError(cudaError_t cudaErr, const char * functWithError)
{
  if ( cudaErr != cudaSuccess )
  {
    fprintf(stderr, "ERROR %s - %s\n", functWithError,
            cudaGetErrorString(cudaErr));
    exit(-1);
  }
}

//UTS variables
struct uts_node {
  unsigned int value;
  unsigned int depth;
  unsigned int to_add;
  int child_a;
  int child_b;
};

#define TASK_FRONT_SIZE 32
int p_fertility;
int global_stack_size;
int local_stack_size;
uts_node *uts_tree;
int cores_used;
int* work_stacks;
int* stack_sizes;
int* stack_heads;
unsigned int* stack_locks;
int num_nodes;
unsigned int nodes_processed;
int leaves_found;
unsigned int leaves_found_golden;
unsigned int leaf_sum;
unsigned int leaf_sum_golden;

#ifdef DISCRETE_GPU
uts_node *d_uts_tree;
int* d_work_stacks;
int* d_stack_sizes;
int* d_stack_heads;
unsigned int* d_stack_locks;
// these need to be pointers - will be allocated on GPU
unsigned int *d_nodes_processed;
int* d_leaves_found;
unsigned int *d_leaf_sum;
#endif

// thread work routine
void *doWork(void *thread_id);

// helper function that rounds x to the next power of two
int getNextPowerOfTwo(int x);

unsigned int __num_work_threads__; // number of CPU threads that do work
unsigned int __num_real_threads__; // number of CPU threads that are launched
unsigned int __cpu_block_size__;   // elements per CPU thread
unsigned int __extra_elements__;   // excessive elements if N % num_threads != 0

unsigned int N = 0;
int N_active_blocks = 0;

__global__ void access_shared(int n_active_blocks, uts_node* utsTree, int* work_stacks, int global_stack_size, int local_stack_size, int* stack_head, unsigned int* stack_lock, int total_nodes, unsigned int* nodes_processed, int* leaves_found, unsigned int* leaf_sum)
{
  // local variables
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int corex = bx % NUM_PHYSICAL_CORES;

  __shared__ unsigned int my_nodes_processed [BLOCK_SIZE/TASK_FRONT_SIZE];
  __shared__ bool got_global[BLOCK_SIZE/TASK_FRONT_SIZE];
  __shared__ int to_enqueue_a[BLOCK_SIZE];
  __shared__ int to_enqueue_b[BLOCK_SIZE];
  int tfid = tx / TASK_FRONT_SIZE; //task front id
  int tfx = tx % TASK_FRONT_SIZE; //task front index

  if(bx < n_active_blocks) {
    int my_local_head;
    int my_global_head;
    int* global_stack = work_stacks;
    int* local_stack = work_stacks + global_stack_size + corex*local_stack_size;
    int* global_head = stack_head;
    int* local_head = stack_head + corex + 1; 
    unsigned int* global_lock = stack_lock;
    unsigned int* local_lock = stack_lock + corex + 1;
    if(tfx==0) {
      my_nodes_processed[tfid] = atomicAdd(nodes_processed, 0);
    }
    while(my_nodes_processed[tfid] < total_nodes) {
      int process_index=-1;
      // begin local critical section
      if(tfx == 0) {
        __gpuLock(local_lock);
      }
      my_local_head = (*local_head);
      // if the local stack is empty, pull from the global stack (if there's anything there)
      // be careful- this is a nested lock
      if(my_local_head==0) {
        // begin global crit section
        if(tfx == 0) {
          got_global[tfid]=__gpuTryLock(global_lock);
        }
        if(got_global[tfid]) {
          int my_global_head = (*global_head);
          // try to fill up about half of the local stack if possible
          for(int i=0; i<(local_stack_size / 2); i+=TASK_FRONT_SIZE) {
            if(my_global_head > tfx) {
              local_stack[i+tfx] = global_stack[my_global_head-tfx-1];
            }
            if(my_global_head > TASK_FRONT_SIZE) {
              my_local_head = my_local_head + TASK_FRONT_SIZE;
              my_global_head = my_global_head - TASK_FRONT_SIZE;
            }
            else {
              my_local_head = my_local_head + my_global_head;
              my_global_head = 0;
              break;
            }
          }
          if(tfx == 0) {
            (*global_head) = my_global_head;
            __gpuUnlock(global_lock);
          }
          // end global crit section
        }
      }
      if(my_local_head > tfx) {
        process_index = local_stack[my_local_head-tfx-1];
      }
      if(tfx == 0) {
        (*local_head) = (my_local_head > TASK_FRONT_SIZE) ? my_local_head - TASK_FRONT_SIZE : 0;
        __gpuUnlock(local_lock);
      }
      // end local critical section
      
      if(process_index>=0) {
        unsigned int childA = utsTree[process_index].child_a;
        unsigned int childB = utsTree[process_index].child_b;
        to_enqueue_a[tx] = childA;
        to_enqueue_b[tx] = childB;
        utsTree[process_index].value += utsTree[process_index].to_add;
        if(childA == 0) {
          atomicAdd(leaf_sum, utsTree[process_index].value);
          atomicAdd(leaves_found, 1);
        }
        else {
          utsTree[childA].depth = utsTree[process_index].depth + 1;
          utsTree[childB].depth = utsTree[process_index].depth + 1;
          utsTree[childA].to_add = 0;
          utsTree[childB].to_add = 0;
        }
        atomicAdd(nodes_processed, 1);
      }
      else {
        to_enqueue_a[tx] = 0;
        to_enqueue_b[tx] = 0;
      }

      // enqueue any children of the nodes we just processed
      // do this serially to avoid gaps in the stack due to leaf nodes
      if(tfx == 0) {
        int next_tf_idx = tx;
        while(next_tf_idx<tx+TASK_FRONT_SIZE) {
          if(to_enqueue_a[next_tf_idx]==0) {
            // this thread's node was a leaf or there was no valid node to process, 
            // either way there's nothing to enqueue on work stack
            next_tf_idx++;
          }
          else {
            bool enqueued = false;
            // begin critical section
            __gpuLock(local_lock);
            my_local_head = (*local_head);
            // only add children nodes to stack if there is space in the local stack. otherwise loop until there's space
            if(my_local_head <= local_stack_size-2) {
              local_stack[my_local_head] = to_enqueue_a[next_tf_idx];
              local_stack[my_local_head+1] = to_enqueue_b[next_tf_idx];
              my_local_head = my_local_head + 2;
              enqueued = true;
            }
            else {
              // if there is not space in the local stack, try to add to the global stack
              // (careful, another nested lock)
              __gpuLockRelaxed(global_lock);
              my_global_head = (*global_head);
              if(my_global_head < global_stack_size - 2) {
                global_stack[my_global_head] = to_enqueue_a[next_tf_idx];
                global_stack[my_global_head+1] = to_enqueue_b[next_tf_idx];
                (*global_head) = my_global_head+2;
              }
              // and donate up to half the local stack if possible
              for(int i=0; my_global_head<global_stack_size && i<local_stack_size / 2; i++) {
                global_stack[my_global_head] = local_stack[my_local_head-1];
                my_global_head++;
                my_local_head--;
              }
              (*global_head) = my_global_head;
              __gpuUnlock(global_lock);
            }
            (*local_head) = my_local_head;
            __gpuUnlock(local_lock);
            // end critical section
            if(enqueued) {
              next_tf_idx++;
            }
          }
        }
      }
      if(tfx == 0) {
        my_nodes_processed[tfid] = atomicAdd(nodes_processed, 0);
      }
    }
  }
}

void callGPU()
{
  /* allocate arrays on CPU */ 
  int restart_count = 0; 

  uts_tree = (uts_node *)calloc(num_nodes, sizeof(uts_node)); 
  cudaError_t cudaErr = cudaGetLastError();
#ifdef DISCRETE_GPU
  checkError(cudaErr, "Before cudaMallocManaged");
  cudaMallocManaged((void**)&d_uts_tree, sizeof(uts_node) * num_nodes); 
  cudaErr = cudaGetLastError();
  checkError(cudaErr, "After cudaMallocManaged1");
#endif
  // need to allocate per-core stack parameters + 1 global stack
  cores_used = (N_active_blocks > NUM_PHYSICAL_CORES) ? NUM_PHYSICAL_CORES : N_active_blocks;
  // global stack size should be size of nodes, but at least 256
  global_stack_size = (num_nodes > 256) ? num_nodes : 256;
  work_stacks = (int *)calloc((global_stack_size + local_stack_size * cores_used), sizeof(int));
#ifdef DISCRETE_GPU
  cudaMallocManaged(&d_work_stacks, sizeof(int) * (global_stack_size + local_stack_size * cores_used));
  cudaErr = cudaGetLastError();
  checkError(cudaErr, "After cudaMallocManaged2");
#endif
  stack_heads = (int *)calloc((cores_used+1), sizeof(int));
#ifdef DISCRETE_GPU
  cudaMallocManaged(&d_stack_heads, sizeof(int*) * (cores_used+1));
  cudaErr = cudaGetLastError();
  checkError(cudaErr, "After cudaMallocManaged3");
#endif
  for(int i=0; i<cores_used+1; i++) {
    stack_heads[i] = 0;
  }
  stack_locks = (unsigned int *)calloc((cores_used+1), sizeof(int));
#ifdef DISCRETE_GPU
  cudaMallocManaged(&d_stack_locks, sizeof(int)*(cores_used+1));
  cudaErr = cudaGetLastError();
  checkError(cudaErr, "After cudaMallocManaged4");
#endif
  for(int i=0; i<cores_used+1; i++) {
    stack_locks[i] = 0;
  }
  nodes_processed = 0;
  leaves_found = 0;
  leaves_found_golden = 0;
  leaf_sum = 0;
  leaf_sum_golden = 0;
  srand(200); 
  std::vector<int> undecided_nodes; 
  std::vector<int> leaf_nodes;
  undecided_nodes.push_back(0); 
  int next_node_idx = 1;
  while(next_node_idx != num_nodes) {
    if(undecided_nodes.size()==0) {
      // ran out of undecided nodes, set all leaf nodes to undecided 
      undecided_nodes = leaf_nodes;
      leaf_nodes.clear();
      restart_count++;
    }
    assert(undecided_nodes.size()>0);
    // process next node
    int next_undecided = undecided_nodes.front();
    uts_tree[next_undecided].value = rand() % 100;
    uts_tree[next_undecided].depth = 0;//init depth to zero
    uts_tree[next_undecided].to_add = 0;//to_add is just a dummy variable to increase memory accesses
    if((rand() % 100) < p_fertility) {
      // if this node is fertile, give it children
      uts_tree[next_undecided].child_a = next_node_idx;
      undecided_nodes.push_back(next_node_idx);
      next_node_idx++;
      uts_tree[next_undecided].child_b = next_node_idx;
      undecided_nodes.push_back(next_node_idx);
      next_node_idx++;
    }
    else {
      // otherwise it is a leaf
      uts_tree[next_undecided].child_a = 0;
      uts_tree[next_undecided].child_b = 0;
      leaf_nodes.insert(leaf_nodes.begin(), next_undecided);
    }
    // pop the node we just processed
    undecided_nodes.erase(undecided_nodes.begin());
  }
  // we've reached the desired node count- make all undecided nodes leaves
  for(std::vector<int>::iterator it = undecided_nodes.begin(); it != undecided_nodes.end(); it++) {
    uts_tree[*it].child_a = 0;
    uts_tree[*it].child_b = 0;
    uts_tree[*it].value = 1;//rand() % 100;
    uts_tree[*it].depth = 0; //init depth to zero
    uts_tree[*it].to_add = 0; //to_add is just a dummy variable to increase memory accesses
    leaf_sum_golden += uts_tree[*it].value;
  }
  // get golden leaf sum and count values
  for(std::vector<int>::iterator it = leaf_nodes.begin(); it != leaf_nodes.end(); it++) {
    leaf_sum_golden += uts_tree[*it].value;
  }
  leaves_found_golden = leaf_nodes.size()+undecided_nodes.size();

  //initialize local stacks with some work
  std::vector<int> init_nodes;
  int num_init_nodes = cores_used * 16;
  init_nodes.push_back(0);
  //while(init_nodes.size() < num_init_nodes) {
  while(init_nodes.size() < num_init_nodes) {
    // pop front node, process it and push its children
    if (init_nodes.size() == 0) {
      printf("Error: not enough nodes to fill stacks (need frontier of size %d, there are %ld leaf nodes)\n", num_init_nodes, leaf_nodes.size());
      assert(0);
    }
    int next_node = init_nodes.front();
    init_nodes.erase(init_nodes.begin());
    if(uts_tree[next_node].child_a > 0) {
      init_nodes.push_back(uts_tree[next_node].child_a);
      init_nodes.push_back(uts_tree[next_node].child_b);
      nodes_processed++;
    }
    else {
      init_nodes.push_back(next_node);
      //leaf_sum += uts_tree[next_node].value;
      //leaves_found++;
    }
  }
  printf("done processing nodes\n");
  int curr_core=0;
  int curr_stack_index=0;
  // push initial nodes onto local stacks
  while(init_nodes.size() > 0) {
    // pop front node and push it on next local stack
    int next_node = init_nodes.front();
    init_nodes.erase(init_nodes.begin());
    work_stacks[global_stack_size+(curr_core*local_stack_size)+curr_stack_index] = next_node;
    stack_heads[curr_core+1] = curr_stack_index+1;
    if(curr_core == cores_used-1) {
      curr_core = 0;
      curr_stack_index++;
    }
    else {
      curr_core++;
    }
  }
  printf("done initing work stack\n");

  /* array for thread ids */
  pthread_t tids[MAX_THREADS];
  int args[MAX_THREADS];

  // Set up and launch threads before starting simulation. There is a barrier in
  // doWork that prevents them from doing work until it's time. Also, sometimes
  // we may launch more threads than were specified but check in doWork if they
  // should do any work. This is beause the tree barrier only works for
  // powers-of-two number of threads.
  tids[0] = pthread_self();
  args[0] = 0;
  for (int i = 1; i < __num_real_threads__; i++) {
    args[i] = i;
    pthread_create(&tids[i], NULL, doWork, &args[i]);
  }
  
  // the main thread should also work!
  doWork((void *) 0);

  printf("Created uts tree size %d with %d leaves, %d leaf sum, %d fertility, and %d restarts:\n", num_nodes, leaves_found_golden, leaf_sum_golden, p_fertility, restart_count);
  if(num_nodes < 512) {
#ifdef DEBUG
    for(int i=0; i<num_nodes; i++) {
      printf("node %d: val=%d children=%d, %d\n", i, uts_tree[i].value, uts_tree[i].child_a, uts_tree[i].child_b);
    }
#endif
  }
  printf("kernel found %d leaf nodes\n", leaves_found);
  printf("kernel found %d leaf sum\n", leaf_sum);
  printf("nodes_processed:%d\n", nodes_processed);
  //verify uts
  if (leaves_found != leaves_found_golden) {
    printf("Error: leaves_found %d != %d\n", leaves_found, leaves_found_golden);
    assert(0);
  }
  if (leaf_sum != leaf_sum_golden) {
    printf("Error: leaf_sum %d != %d\n", leaf_sum, leaf_sum_golden);
    assert(0);
  }
  if (nodes_processed != num_nodes) {
    printf("Error: nodes_processed %d != %d\n", nodes_processed, num_nodes);
    assert(0);
  }
  for(int i=0; i<cores_used+1; i++) {
    if (stack_heads[i] != 0) {
      printf("Error: stack head %d = %d\n", i, stack_heads[i]);
      assert(0);
    }
    if (stack_heads[i] != 0) {
      printf("Error: stack lock %d = %d\n", i, stack_locks[i]);
      assert(0);
    }
  }
  printf("SUCCESS!\n");

  /*free memory on CPU */
#ifdef DISCRETE_GPU
  cudaFreeHost(d_uts_tree);
  cudaFreeHost(d_work_stacks);
  cudaFreeHost(d_stack_heads);
  cudaFreeHost(d_stack_locks);
  cudaFreeHost(d_nodes_processed);
  cudaFreeHost(d_nodes_processed);
  cudaFreeHost(d_leaves_found);
  cudaFreeHost(d_leaf_sum);
#endif
  free(uts_tree);
  free(work_stacks);
  free(stack_heads);
  free(stack_locks);
}

void *doWork(void * thread_id) {
  unsigned long tid = (unsigned long) thread_id;

#ifdef DEBUG
  fprintf(stdout, "Thread ID: %ld\n", tid);
#endif
  // We want to use every CPU core except the one which shares a node with the
  // GPU core. That core's ID is 1. Therefore, don't bind to 1 - bind to the
  // first unallocated core. As a result, however, one of the dummy threads is
  // bound to core 1. Hopefully, this shouldn't affect the GPU (significantly).

  //int start = tid * __cpu_block_size__;
  //int stop  = start + __cpu_block_size__;

  // make sure all threads have reached here before continuing
  pthread_barrier_wait(&barrier);

  // the main thread starts the simulation and launches the GPU kernel
  if (tid == 0) {
    /* GPU thread block and grid size */
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
    dim3 numBlocks(N_active_blocks, 1, 1);

    //uts inputs
#ifdef DISCRETE_GPU
    cudaError_t cudaErr = cudaGetLastError();
    checkError(cudaErr, "Before cudaMallocManageds");
    cudaMallocManaged(&d_nodes_processed, sizeof(unsigned int));
    cudaMallocManaged(&d_leaves_found, sizeof(int));
    cudaMallocManaged(&d_leaf_sum, sizeof(unsigned int));
    cudaErr = cudaGetLastError();
    checkError(cudaErr, "After cudaMallocManageds");

    cudaMemcpy(d_uts_tree, uts_tree, sizeof(uts_node) * num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_work_stacks, work_stacks, sizeof(int) * (global_stack_size + local_stack_size*cores_used), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stack_heads, stack_heads, sizeof(int) * (cores_used+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stack_locks, stack_locks, sizeof(int) * (cores_used+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_processed, &nodes_processed, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leaves_found, &leaves_found, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leaf_sum, &leaf_sum, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaErr = cudaGetLastError();
    checkError(cudaErr, "After cudaMemcpys");

    uts_node * uts_tree_ptr = d_uts_tree;
    int* work_stacks_ptr = d_work_stacks;
    int* stack_head_ptr = d_stack_heads;
    unsigned int* stack_lock_ptr = d_stack_locks;
    unsigned int* nodes_processed_ptr = d_nodes_processed;
    int* leaves_found_ptr = d_leaves_found;
    unsigned int* leaf_sum_ptr = d_leaf_sum;
#else
    uts_node * uts_tree_ptr = uts_tree;
    int* work_stacks_ptr = work_stacks;
    int* stack_head_ptr = stack_heads;
    unsigned int* stack_lock_ptr = stack_locks;
    unsigned int* nodes_processed_ptr = &nodes_processed;
    int* leaves_found_ptr = &leaves_found;
    unsigned int* leaf_sum_ptr = &leaf_sum;
#endif

    printf("Launch uts kernel...\n");
    access_shared<<<numBlocks, threadsPerBlock>>>(N_active_blocks, uts_tree_ptr, work_stacks_ptr, global_stack_size, local_stack_size, stack_head_ptr, stack_lock_ptr, num_nodes, nodes_processed_ptr, leaves_found_ptr, leaf_sum_ptr);
    cudaErr = cudaGetLastError();
    checkError(cudaErr, "After Launch");
    cudaErr = cudaDeviceSynchronize();
    checkError(cudaErr, "cudaDeviceSynchronize");
    printf("Kernel completed.\n");

#ifdef DISCRETE_GPU
    cudaMemcpy(&nodes_processed, d_nodes_processed, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&leaves_found, d_leaves_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&leaf_sum, d_leaf_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stack_heads, d_stack_heads, sizeof(int) * (cores_used+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(stack_locks, d_stack_locks, sizeof(int) * (cores_used+1), cudaMemcpyDeviceToHost);
    cudaErr = cudaGetLastError();
    checkError(cudaErr, "After cudaMemcpy results");
#endif
  }

  // make sure all threads have reached here before continuing
  pthread_barrier_wait(&barrier);
  return 0;
}

// helper function that rounds x up to the next power of two
int getNextPowerOfTwo(int x)
{
  int power = 1;
  while (power < x)
    power *= 2;
  return power;
}

int main(int argc, char *argv[])
{
  if (argc != NUM_ARGS) {
    fprintf(stderr, "Invalid arguments passed in, expected:\n");
    fprintf(stderr, "./uts <numThreads> <numElements> <fertility> <local_stack_size> <N_active_blocks>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numThreads>: Number of CPU threads to launch\n");
    fprintf(stderr, "\t<numElements>: Number of elements in the data array\n");
    fprintf(stderr, "\t<fertility>: 1-100, percent chance that a node has children (for unbalanced tree generation)\n");
    fprintf(stderr, "\t<local_stack_size>: if < 0, how many times to half the num nodes before getting aggregate local stack size. If > 0, what the maximum local stack size should be\n");
    fprintf(stderr, "\t<N_active_blocks>: Number of threadblocks to use (only for inter-core tests)\n");
    exit(-1);
  }

  N = atoi(argv[2]);
  p_fertility = atoi(argv[3]);
  global_stack_size = (num_nodes > 256) ? num_nodes : 256;
  local_stack_size = atoi(argv[4]);
  N_active_blocks = atoi(argv[5]);

  __num_work_threads__ = atoi(argv[1]);
  printf("# CPU threads that do work:%d\n", __num_work_threads__);
  __num_real_threads__ = getNextPowerOfTwo(__num_work_threads__);
  __cpu_block_size__ = (int) N/__num_work_threads__;
  __extra_elements__ = N % __num_work_threads__;

  // number of nodes must be odd
  num_nodes = N + ((N % 2) ? 0 : 1);//TREE_SIZE + ((TREE_SIZE % 2) ? 0 : 1);

  assert(N > 0 && "N must be > 0!\n");
  assert(__num_work_threads__ <= MAX_THREADS && "Maximum number of threads exceeded");

  printf("Running with N_elements=%d N_active_blocks=%d\n", num_nodes, N_active_blocks);
  pthread_barrier_init(&barrier, NULL, __num_real_threads__);
  callGPU();

  return 0;
}
