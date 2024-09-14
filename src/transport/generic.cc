#include "comm.h"
#include "transport.h"

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int c = 0; c < comm->nChannels; c++) {
      // struct ncclChannel* channel = comm->channels + c;
      // NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
      int rank_inter = (comm->rank / NCCL_KPARAM) * NCCL_KPARAM;
      int rank_intra = comm->rank % NCCL_KPARAM;
      int prev[2] = {
        rank_inter                                       + ((rank_intra + NCCL_KPARAM - 1) % NCCL_KPARAM),
        ((rank_inter + comm->nRanks - NCCL_KPARAM) % comm->nRanks) + rank_intra
      };
      int next[2] = {
        rank_inter                        + ((rank_intra + 1) % NCCL_KPARAM),
        ((rank_inter + NCCL_KPARAM) % comm->nRanks) + rank_intra
      };
      // printf("(%d->%d->%d  %d->%d->%d) ", prev[0], comm->rank, next[0], prev[1], comm->rank, next[1]);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 2, prev, 2, next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all rings");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}
