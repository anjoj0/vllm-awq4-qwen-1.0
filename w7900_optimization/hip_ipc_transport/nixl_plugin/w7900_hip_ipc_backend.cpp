#include "w7900_hip_ipc_backend.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "common/nixl_log.h"

namespace {

constexpr uint64_t kWireMagic = 0x5748495043504931ULL;  // WHIPCPI1
constexpr uint32_t kWireVersion = 1;
constexpr size_t kMaxNotificationDatagram = 65536;

struct WireMetadata {
    uint64_t magic;
    uint32_t version;
    int32_t exporter_device;
    uint64_t allocation_base;
    uint64_t allocation_size;
    hipIpcMemHandle_t handle;
};

struct LocalMetadata final : nixlBackendMD {
    LocalMetadata() : nixlBackendMD(true) {}
    WireMetadata wire{};
};

struct CopyDesc {
    uintptr_t destination;
    uintptr_t source;
    size_t bytes;
};

enum class TransferState {
    Prepared,
    InProgress,
    DataComplete,
    Complete,
    Failed,
};

struct TransferHandle final : nixlBackendReqH {
    std::vector<CopyDesc> copies;
    nixl_xfer_op_t operation = NIXL_READ;
    int device = 0;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t end = nullptr;
    std::string remote_agent;
    std::string notification;
    bool notification_sent = false;
    TransferState state = TransferState::Prepared;
    nixl_status_t failure_status = NIXL_SUCCESS;
    hipError_t hip_failure = hipSuccess;
    uint64_t post_count = 0;
    mutable std::mutex mutex;
};

nixl_status_t hipStatus(hipError_t status, const char* operation) {
    if (status == hipSuccess) return NIXL_SUCCESS;
    NIXL_ERROR << operation << " failed: " << hipGetErrorString(status);
    return NIXL_ERR_BACKEND;
}

void destroyRequestResources(TransferHandle& request, bool synchronize) {
    (void)hipSetDevice(request.device);
    if (synchronize && request.stream != nullptr)
        (void)hipStreamSynchronize(request.stream);
    if (request.end != nullptr) (void)hipEventDestroy(request.end);
    if (request.start != nullptr) (void)hipEventDestroy(request.start);
    if (request.stream != nullptr) (void)hipStreamDestroy(request.stream);
    request.end = nullptr;
    request.start = nullptr;
    request.stream = nullptr;
}

nixl_status_t createRequestResources(TransferHandle& request) {
    hipError_t status = hipSetDevice(request.device);
    if (status == hipSuccess)
        status = hipStreamCreateWithFlags(&request.stream, hipStreamNonBlocking);
    if (status == hipSuccess) status = hipEventCreate(&request.start);
    if (status == hipSuccess) status = hipEventCreate(&request.end);
    if (status != hipSuccess) {
        request.hip_failure = status;
        destroyRequestResources(request, false);
        return hipStatus(status, "HIP IPC request resource creation");
    }
    return NIXL_SUCCESS;
}

nixl_status_t prepareRequestForPost(TransferHandle& request) {
    if (request.state == TransferState::InProgress ||
        request.state == TransferState::DataComplete)
        return NIXL_ERR_REPOST_ACTIVE;

    if (request.state == TransferState::Failed)
        destroyRequestResources(request, true);
    if (request.stream == nullptr) {
        const nixl_status_t status = createRequestResources(request);
        if (status != NIXL_SUCCESS) {
            request.state = TransferState::Failed;
            request.failure_status = status;
            return status;
        }
    }

    request.failure_status = NIXL_SUCCESS;
    request.hip_failure = hipSuccess;
    request.notification_sent = false;
    request.state = TransferState::Prepared;
    return NIXL_SUCCESS;
}

std::string mappingKey(const WireMetadata& wire) {
    return std::string(reinterpret_cast<const char*>(&wire.handle), sizeof(wire.handle));
}

std::string makeSocketPath(const std::string& agent) {
    const auto hash = std::hash<std::string>{}(agent);
    return "/tmp/nixl-w7900-hipipc-" + std::to_string(getpid()) + "-" +
           std::to_string(hash) + ".sock";
}

std::vector<CopyDesc> coalesce(std::vector<CopyDesc> copies) {
    std::vector<CopyDesc> merged;
    merged.reserve(copies.size());
    for (const auto& copy : copies) {
        if (!merged.empty()) {
            auto& previous = merged.back();
            if (previous.destination + previous.bytes == copy.destination &&
                previous.source + previous.bytes == copy.source) {
                previous.bytes += copy.bytes;
                continue;
            }
        }
        merged.push_back(copy);
    }
    return merged;
}

}  // namespace

struct W7900HipIpcBackend::RemoteMapping {
    int device = 0;
    uintptr_t remote_base = 0;
    size_t allocation_size = 0;
    uintptr_t mapped_base = 0;

    ~RemoteMapping() {
        if (mapped_base == 0) return;
        (void)hipSetDevice(device);
        const hipError_t status =
            hipIpcCloseMemHandle(reinterpret_cast<void*>(mapped_base));
        if (status != hipSuccess) {
            NIXL_ERROR << "hipIpcCloseMemHandle failed: " << hipGetErrorString(status);
        }
    }
};

namespace {

struct RemoteMetadata final : nixlBackendMD {
    explicit RemoteMetadata(std::shared_ptr<W7900HipIpcBackend::RemoteMapping> value)
        : nixlBackendMD(false), mapping(std::move(value)) {}
    std::shared_ptr<W7900HipIpcBackend::RemoteMapping> mapping;
};

}  // namespace

W7900HipIpcBackend::W7900HipIpcBackend(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params), socket_path_(makeSocketPath(init_params->localAgent)) {
    if (hipGetDevice(&device_) != hipSuccess) {
        initErr = true;
        return;
    }

    notif_fd_ = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (notif_fd_ < 0) {
        NIXL_ERROR << "socket(AF_UNIX) failed: " << std::strerror(errno);
        initErr = true;
        return;
    }
    sockaddr_un address{};
    if (socket_path_.size() >= sizeof(address.sun_path)) {
        NIXL_ERROR << "HIP IPC notification socket path is too long";
        initErr = true;
        return;
    }
    (void)unlink(socket_path_.c_str());
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
    if (bind(notif_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        NIXL_ERROR << "bind(" << socket_path_ << ") failed: " << std::strerror(errno);
        initErr = true;
    }
}

W7900HipIpcBackend::~W7900HipIpcBackend() {
    remote_mappings_.clear();
    if (notif_fd_ >= 0) close(notif_fd_);
    if (!socket_path_.empty()) (void)unlink(socket_path_.c_str());
}

nixl_status_t W7900HipIpcBackend::registerMem(const nixlBlobDesc& mem,
                                               const nixl_mem_t& nixl_mem,
                                               nixlBackendMD*& out) {
    if (nixl_mem != VRAM_SEG) return NIXL_ERR_NOT_SUPPORTED;
    auto metadata = std::make_unique<LocalMetadata>();
    void* allocation_base = nullptr;
    size_t allocation_size = 0;
    hipError_t status = hipSetDevice(device_);
    if (status != hipSuccess) return hipStatus(status, "hipSetDevice registerMem");
    status = hipMemGetAddressRange(
        &allocation_base, &allocation_size, reinterpret_cast<void*>(mem.addr));
    if (status != hipSuccess) return hipStatus(status, "hipMemGetAddressRange");

    metadata->wire.magic = kWireMagic;
    metadata->wire.version = kWireVersion;
    metadata->wire.exporter_device = device_;
    metadata->wire.allocation_base = reinterpret_cast<uintptr_t>(allocation_base);
    metadata->wire.allocation_size = allocation_size;
    status = hipIpcGetMemHandle(&metadata->wire.handle, allocation_base);
    if (status != hipSuccess) return hipStatus(status, "hipIpcGetMemHandle");
    out = metadata.release();
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::deregisterMem(nixlBackendMD* meta) {
    delete static_cast<LocalMetadata*>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::getPublicData(const nixlBackendMD* meta,
                                                 std::string& out) const {
    const auto* local = static_cast<const LocalMetadata*>(meta);
    out.assign(reinterpret_cast<const char*>(&local->wire), sizeof(local->wire));
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::loadRemoteMD(const nixlBlobDesc& input,
                                                const nixl_mem_t& nixl_mem,
                                                const std::string&,
                                                nixlBackendMD*& output) {
    if (nixl_mem != VRAM_SEG || input.metaInfo.size() != sizeof(WireMetadata))
        return NIXL_ERR_INVALID_PARAM;
    WireMetadata wire{};
    std::memcpy(&wire, input.metaInfo.data(), sizeof(wire));
    if (wire.magic != kWireMagic || wire.version != kWireVersion)
        return NIXL_ERR_MISMATCH;

    const std::string key = mappingKey(wire);
    std::shared_ptr<RemoteMapping> mapping;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = remote_mappings_.find(key);
        if (found != remote_mappings_.end()) mapping = found->second.lock();
        if (!mapping) {
            auto created = std::make_shared<RemoteMapping>();
            created->device = device_;
            created->remote_base = wire.allocation_base;
            created->allocation_size = wire.allocation_size;
            void* mapped = nullptr;
            hipError_t status = hipSetDevice(device_);
            if (status != hipSuccess)
                return hipStatus(status, "hipSetDevice loadRemoteMD");
            status = hipIpcOpenMemHandle(
                &mapped, wire.handle, hipIpcMemLazyEnablePeerAccess);
            if (status != hipSuccess) return hipStatus(status, "hipIpcOpenMemHandle");
            created->mapped_base = reinterpret_cast<uintptr_t>(mapped);
            mapping = std::move(created);
            remote_mappings_[key] = mapping;
        }
    }
    output = new RemoteMetadata(std::move(mapping));
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::unloadMD(nixlBackendMD* input) {
    delete static_cast<RemoteMetadata*>(input);
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::getConnInfo(std::string& out) const {
    out = socket_path_;
    return out.empty() ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::loadRemoteConnInfo(
    const std::string& remote_agent, const std::string& remote_conn_info) {
    sockaddr_un address{};
    if (remote_conn_info.empty() || remote_conn_info.size() >= sizeof(address.sun_path))
        return NIXL_ERR_INVALID_PARAM;
    std::lock_guard<std::mutex> lock(mutex_);
    remote_sockets_[remote_agent] = remote_conn_info;
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::connect(const std::string& remote_agent) {
    std::lock_guard<std::mutex> lock(mutex_);
    return remote_sockets_.contains(remote_agent) ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t W7900HipIpcBackend::disconnect(const std::string& remote_agent) {
    std::lock_guard<std::mutex> lock(mutex_);
    remote_sockets_.erase(remote_agent);
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::prepXfer(const nixl_xfer_op_t& operation,
                                            const nixl_meta_dlist_t& local,
                                            const nixl_meta_dlist_t& remote,
                                            const std::string& remote_agent,
                                            nixlBackendReqH*& handle,
                                            const nixl_opt_b_args_t* opt_args) const {
    if (operation != NIXL_READ && operation != NIXL_WRITE)
        return NIXL_ERR_NOT_SUPPORTED;
    if (local.descCount() == 0 || local.descCount() != remote.descCount())
        return NIXL_ERR_INVALID_PARAM;

    auto request = std::make_unique<TransferHandle>();
    request->operation = operation;
    request->device = device_;
    request->remote_agent = remote_agent;
    if (opt_args != nullptr && opt_args->hasNotif)
        request->notification = opt_args->notifMsg;
    request->copies.reserve(local.descCount());
    for (size_t i = 0; i < local.descCount(); ++i) {
        if (local[i].len != remote[i].len || remote[i].metadataP == nullptr)
            return NIXL_ERR_INVALID_PARAM;
        const auto* metadata = static_cast<const RemoteMetadata*>(remote[i].metadataP);
        const auto& mapping = metadata->mapping;
        if (remote[i].addr < mapping->remote_base ||
            remote[i].len > mapping->allocation_size ||
            remote[i].addr - mapping->remote_base >
                mapping->allocation_size - remote[i].len)
            return NIXL_ERR_INVALID_PARAM;
        const uintptr_t mapped_remote =
            mapping->mapped_base + remote[i].addr - mapping->remote_base;
        if (operation == NIXL_READ)
            request->copies.push_back({local[i].addr, mapped_remote, local[i].len});
        else
            request->copies.push_back({mapped_remote, local[i].addr, local[i].len});
    }
    request->copies = coalesce(std::move(request->copies));
    handle = request.release();
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::postXfer(const nixl_xfer_op_t& operation,
                                            const nixl_meta_dlist_t&,
                                            const nixl_meta_dlist_t&,
                                            const std::string& remote_agent,
                                            nixlBackendReqH*& handle,
                                            const nixl_opt_b_args_t* opt_args) const {
    auto* request = static_cast<TransferHandle*>(handle);
    if (request == nullptr || operation != request->operation ||
        remote_agent != request->remote_agent)
        return NIXL_ERR_INVALID_PARAM;

    std::lock_guard<std::mutex> lock(request->mutex);
    nixl_status_t request_status = prepareRequestForPost(*request);
    if (request_status != NIXL_SUCCESS) return request_status;

    request->notification.clear();
    if (opt_args != nullptr && opt_args->hasNotif)
        request->notification = opt_args->notifMsg;

    hipError_t status = hipSetDevice(request->device);
    if (status == hipSuccess) status = hipEventRecord(request->start, request->stream);
    for (const auto& copy : request->copies) {
        if (status != hipSuccess) break;
        status = hipMemcpyAsync(reinterpret_cast<void*>(copy.destination),
                                reinterpret_cast<const void*>(copy.source), copy.bytes,
                                hipMemcpyDeviceToDevice, request->stream);
    }
    if (status == hipSuccess) status = hipEventRecord(request->end, request->stream);
    if (status != hipSuccess) {
        request->state = TransferState::Failed;
        request->failure_status = NIXL_ERR_BACKEND;
        request->hip_failure = status;
        destroyRequestResources(*request, true);
        return hipStatus(status, "HIP IPC transfer submission");
    }
    request->state = TransferState::InProgress;
    ++request->post_count;
    return NIXL_IN_PROG;
}

nixl_status_t W7900HipIpcBackend::checkXfer(nixlBackendReqH* handle) const {
    auto* request = static_cast<TransferHandle*>(handle);
    if (request == nullptr) return NIXL_ERR_INVALID_PARAM;

    std::lock_guard<std::mutex> lock(request->mutex);
    if (request->state == TransferState::Prepared) return NIXL_ERR_INVALID_PARAM;
    if (request->state == TransferState::Failed) return request->failure_status;
    if (request->state == TransferState::Complete) return NIXL_SUCCESS;

    if (request->state == TransferState::InProgress) {
        hipError_t status = hipSetDevice(request->device);
        if (status == hipSuccess) status = hipEventQuery(request->end);
        if (status == hipErrorNotReady) return NIXL_IN_PROG;
        if (status != hipSuccess) {
            request->state = TransferState::Failed;
            request->failure_status = NIXL_ERR_BACKEND;
            request->hip_failure = status;
            return hipStatus(status, "HIP IPC transfer completion");
        }
        request->state = TransferState::DataComplete;
    }

    if (!request->notification.empty() && !request->notification_sent) {
        const nixl_status_t notif_status =
            sendNotif(request->remote_agent, request->notification);
        if (notif_status == NIXL_IN_PROG) return NIXL_IN_PROG;
        if (notif_status != NIXL_SUCCESS) {
            request->state = TransferState::Failed;
            request->failure_status = notif_status;
            return notif_status;
        }
        request->notification_sent = true;
    }
    request->state = TransferState::Complete;
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::releaseReqH(nixlBackendReqH* handle) const {
    auto* request = static_cast<TransferHandle*>(handle);
    if (request == nullptr) return NIXL_SUCCESS;
    std::unique_lock<std::mutex> lock(request->mutex);
    destroyRequestResources(*request, true);
    lock.unlock();
    delete request;
    return NIXL_SUCCESS;
}

nixl_status_t W7900HipIpcBackend::sendNotif(const std::string& remote_agent,
                                             const std::string& msg) const {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = remote_sockets_.find(remote_agent);
        if (found == remote_sockets_.end()) return NIXL_ERR_NOT_FOUND;
        path = found->second;
    }
    if (path.empty() ||
        path.size() >= sizeof(static_cast<sockaddr_un*>(nullptr)->sun_path))
        return NIXL_ERR_INVALID_PARAM;
    if (localAgent.size() > std::numeric_limits<uint32_t>::max() ||
        msg.size() > std::numeric_limits<uint32_t>::max())
        return NIXL_ERR_INVALID_PARAM;
    constexpr size_t header_size = 2 * sizeof(uint32_t);
    if (localAgent.size() > kMaxNotificationDatagram - header_size ||
        msg.size() > kMaxNotificationDatagram - header_size - localAgent.size())
        return NIXL_ERR_INVALID_PARAM;

    const uint32_t sender_size = static_cast<uint32_t>(localAgent.size());
    const uint32_t message_size = static_cast<uint32_t>(msg.size());
    std::string payload(sizeof(sender_size) + sizeof(message_size) + sender_size + message_size,
                        '\0');
    char* cursor = payload.data();
    std::memcpy(cursor, &sender_size, sizeof(sender_size));
    cursor += sizeof(sender_size);
    std::memcpy(cursor, &message_size, sizeof(message_size));
    cursor += sizeof(message_size);
    std::memcpy(cursor, localAgent.data(), sender_size);
    cursor += sender_size;
    std::memcpy(cursor, msg.data(), message_size);

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    while (true) {
        const ssize_t sent = sendto(notif_fd_, payload.data(), payload.size(), 0,
                                    reinterpret_cast<sockaddr*>(&address),
                                    sizeof(address));
        if (sent == static_cast<ssize_t>(payload.size())) return NIXL_SUCCESS;
        if (sent >= 0) return NIXL_ERR_BACKEND;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS)
            return NIXL_IN_PROG;
        if (errno == ENOENT || errno == ECONNREFUSED)
            return NIXL_ERR_REMOTE_DISCONNECT;
        NIXL_ERROR << "HIP IPC notification send failed: " << std::strerror(errno);
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t W7900HipIpcBackend::genNotif(const std::string& remote_agent,
                                            const std::string& msg) const {
    return sendNotif(remote_agent, msg);
}

nixl_status_t W7900HipIpcBackend::getNotifs(notif_list_t& notif_list) {
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;
    std::array<char, kMaxNotificationDatagram> buffer{};
    while (true) {
        iovec vector{buffer.data(), buffer.size()};
        msghdr message{};
        message.msg_iov = &vector;
        message.msg_iovlen = 1;
        const ssize_t bytes = recvmsg(notif_fd_, &message, MSG_TRUNC);
        if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        if (bytes < 0 && errno == EINTR) continue;
        if (bytes < 0) {
            NIXL_ERROR << "HIP IPC notification receive failed: "
                       << std::strerror(errno);
            return NIXL_ERR_BACKEND;
        }
        if ((message.msg_flags & MSG_TRUNC) != 0 ||
            static_cast<size_t>(bytes) > buffer.size())
            return NIXL_ERR_MISMATCH;
        if (bytes < static_cast<ssize_t>(2 * sizeof(uint32_t)))
            return NIXL_ERR_MISMATCH;
        uint32_t sender_size = 0;
        uint32_t message_size = 0;
        std::memcpy(&sender_size, buffer.data(), sizeof(sender_size));
        std::memcpy(&message_size, buffer.data() + sizeof(sender_size), sizeof(message_size));
        constexpr size_t header_size = 2 * sizeof(uint32_t);
        if (sender_size > buffer.size() - header_size ||
            message_size > buffer.size() - header_size - sender_size)
            return NIXL_ERR_MISMATCH;
        const size_t expected = header_size + sender_size + message_size;
        if (expected != static_cast<size_t>(bytes)) return NIXL_ERR_MISMATCH;
        const char* cursor = buffer.data() + 2 * sizeof(uint32_t);
        notif_list.emplace_back(std::string(cursor, sender_size),
                                std::string(cursor + sender_size, message_size));
    }
    return NIXL_SUCCESS;
}
