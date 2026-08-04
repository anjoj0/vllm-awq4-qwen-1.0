#pragma once

#include <hip/hip_runtime.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/backend_engine.h"

class W7900HipIpcBackend final : public nixlBackendEngine {
public:
    struct RemoteMapping;

    explicit W7900HipIpcBackend(const nixlBackendInitParams* init_params);
    ~W7900HipIpcBackend() override;

    bool supportsRemote() const override { return true; }
    bool supportsLocal() const override { return false; }
    bool supportsNotif() const override { return true; }
    nixl_mem_list_t getSupportedMems() const override { return {VRAM_SEG}; }

    nixl_status_t registerMem(const nixlBlobDesc& mem, const nixl_mem_t& nixl_mem,
                              nixlBackendMD*& out) override;
    nixl_status_t deregisterMem(nixlBackendMD* meta) override;
    nixl_status_t getPublicData(const nixlBackendMD* meta, std::string& out) const override;
    nixl_status_t loadRemoteMD(const nixlBlobDesc& input, const nixl_mem_t& nixl_mem,
                               const std::string& remote_agent,
                               nixlBackendMD*& output) override;
    nixl_status_t unloadMD(nixlBackendMD* input) override;

    nixl_status_t getConnInfo(std::string& out) const override;
    nixl_status_t loadRemoteConnInfo(const std::string& remote_agent,
                                     const std::string& remote_conn_info) override;
    nixl_status_t connect(const std::string& remote_agent) override;
    nixl_status_t disconnect(const std::string& remote_agent) override;

    nixl_status_t prepXfer(const nixl_xfer_op_t& operation,
                           const nixl_meta_dlist_t& local,
                           const nixl_meta_dlist_t& remote,
                           const std::string& remote_agent,
                           nixlBackendReqH*& handle,
                           const nixl_opt_b_args_t* opt_args = nullptr) const override;
    nixl_status_t postXfer(const nixl_xfer_op_t& operation,
                           const nixl_meta_dlist_t& local,
                           const nixl_meta_dlist_t& remote,
                           const std::string& remote_agent,
                           nixlBackendReqH*& handle,
                           const nixl_opt_b_args_t* opt_args = nullptr) const override;
    nixl_status_t checkXfer(nixlBackendReqH* handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH* handle) const override;

    nixl_status_t getNotifs(notif_list_t& notif_list) override;
    nixl_status_t genNotif(const std::string& remote_agent,
                           const std::string& msg) const override;

private:
    nixl_status_t sendNotif(const std::string& remote_agent,
                            const std::string& msg) const;

    int device_ = 0;
    int notif_fd_ = -1;
    std::string socket_path_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> remote_sockets_;
    std::unordered_map<std::string, std::weak_ptr<RemoteMapping>> remote_mappings_;
};
