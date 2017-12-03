#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

#ifdef USE_NCCL

#include <boost/thread.hpp>

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/nccl.hpp"

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient

DISABLE_COPY_AND_ASSIGN(Params);
};

// Params stored in GPU memory.
template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~GPUParams();

  void Configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

template<typename Dtype>
class NCCL : public GPUParams<Dtype>,
             public Solver<Dtype>::Callback,
             public Net<Dtype>::Callback {
 public:
  /**
   * Single process version.
   */
  explicit NCCL(shared_ptr<Solver<Dtype> > solver);
  /**
   * In multi-process settings, first create a NCCL id (new_uid), then
   * pass it to each process to create connected instances.
   */
  NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid);
  ~NCCL();

  boost::barrier* barrier();
  void set_barrier(boost::barrier* value);

  /**
   * In single process settings, create instances without uids and
   * call this to connect them.
   */
  static void InitSingleProcess(vector<NCCL<Dtype>*>* nccls);

  static string new_uid();

  /**
   * Broadcast weights from rank 0 other solvers.
   */
  void Broadcast();

  /**
   * Single process multi-GPU.
   */
  virtual void Run(const vector<int>& gpus, const char* restore);

 protected:
  void Init();
  void on_start() {}
  void run(int layer);  // Net callback
  void on_gradients_ready();

  ncclComm_t comm_;
  cudaStream_t stream_;

  shared_ptr<Solver<Dtype> > solver_;
  // Should not be necessary, https://github.com/NVIDIA/nccl/issues/37
  boost::barrier* barrier_;
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

template<typename Dtype>
class CGDP : public NCCL<Dtype>, public Solver<Dtype>::ICallback {
 public:
  /**
   * Single process version.
   */
  explicit CGDP(shared_ptr<Solver<Dtype> > solver);
  explicit CGDP(shared_ptr<Solver<Dtype> > solver, Dtype* grads,
		vector<BlockingQueue<int>* >* criticals_free,
		int threshold);
  /**
   * In multi-process settings, first create a NCCL id (new_uid), then
   * pass it to each process to create connected instances.
   */
  CGDP(shared_ptr<Solver<Dtype> > solver, const string& uid);
  ~CGDP();

  void set_threshold(int threshold) { threshold_ = threshold; }

  static void InitSingleProcess(vector<CGDP<Dtype>*>* nccls);

#ifndef CPU_ONLY
  static void CUDART_CB callback_grads(cudaStream_t stream,
				       cudaError_t status,
				       void* tp){
    CGDP<Dtype>* sync = (CGDP<Dtype>*)tp;
    sync->accumulate_gradients();
  }
  static void CUDART_CB callback_reset_variables(cudaStream_t stream,
						 cudaError_t status,
						 void* tp){
    CGDP<Dtype>* sync = (CGDP<Dtype>*)tp;
    sync->reset_variables();
  }
#endif

  virtual void Run(const vector<int>& gpus, const char* restore);

 protected:
  void CGInit();
  void on_start();
  void reset_variables();
  void on_inner_iteration(int inner_iter);
  void run(int layer);  // Net callback
  void accumulate_gradients();  // Accumulation on host
  void on_gradients_ready();

  // a shared array on host to store the summation of gradients
  Dtype* grads_;
  // gradients on cpu of a solver
  Dtype* cpu_diff_;
  // blobs of learnable parameters that the solver computed
  // during the backward
  BlockingQueue<vector<int> > ready_blobs_;
  // queues to sync callbacks for each layer
  vector<BlockingQueue<int>* >* criticals_free_;
  // mapping the id of a blob in the blobs of learnable parameters to
  // its id in the shared array grads_ and diff_
  vector<int> pid_aid_;
  vector<int> pid_size_;
  // the number of blobs of learnable parameters
  int blobs_num_;
  // the number of solvers/gpus
  int solvers_num_;
  // a list of layers that has had the updated accumulation on host
  // this layer is ready to send the accum. on host to GPU
  BlockingQueue<int> updated_layers_;
  // these are used to transfer data between host and devices
  #ifndef CPU_ONLY
  cudaStream_t d2h_h_stream_;
  cudaStream_t h2d_stream_;
  #endif
  // iteration index if iter_size is set
  int inner_iter_;

  // overhead of gradient accumulation on host
  float grad_overhead_;

  // command line arguments
  int threshold_;

  using NCCL<Dtype>::solver_;
  using NCCL<Dtype>::barrier_;
  using NCCL<Dtype>::diff_;
  using NCCL<Dtype>::size_;
};
}  // namespace caffe

#endif  // USE_NCCL
#endif  // header
