// Minimal ROCr IPC allocation-lifetime reproducer. No UCX, NIXL, or HIP API.

#include <hsa.h>
#include <hsa_ext_amd.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x48534149;  // "HSAI"

struct WireHandle {
    uint32_t magic;
    uint32_t version;
    uint64_t length;
    uint32_t exporter_gpu;
    uint32_t mode;
    hsa_amd_ipc_memory_t ipc;
};

struct PoolSearch {
    hsa_amd_memory_pool_t pool{};
    bool found = false;
};

std::vector<hsa_agent_t> g_gpu_agents;

const char* status_text(hsa_status_t status)
{
    const char* text = nullptr;
    hsa_status_string(status, &text);
    return text == nullptr ? "unknown" : text;
}

void log_status(const char* role, const char* operation, hsa_status_t status)
{
    std::printf("EVENT role=%s operation=%s status=%d status_text=%s\n", role,
                operation, static_cast<int>(status), status_text(status));
    std::fflush(stdout);
}

bool check_status(const char* role, const char* operation, hsa_status_t status)
{
    log_status(role, operation, status);
    return status == HSA_STATUS_SUCCESS;
}

hsa_status_t agent_callback(hsa_agent_t agent, void*)
{
    hsa_device_type_t type;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) ==
            HSA_STATUS_SUCCESS &&
        type == HSA_DEVICE_TYPE_GPU) {
        g_gpu_agents.push_back(agent);
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t pool_callback(hsa_amd_memory_pool_t pool, void* data)
{
    auto* search = static_cast<PoolSearch*>(data);
    hsa_amd_segment_t segment;
    bool alloc_allowed = false;

    if (hsa_amd_memory_pool_get_info(
                pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) !=
            HSA_STATUS_SUCCESS ||
        segment != HSA_AMD_SEGMENT_GLOBAL ||
        hsa_amd_memory_pool_get_info(
                pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                &alloc_allowed) != HSA_STATUS_SUCCESS ||
        !alloc_allowed) {
        return HSA_STATUS_SUCCESS;
    }

    if (!search->found) {
        search->pool  = pool;
        search->found = true;
    }
    return HSA_STATUS_SUCCESS;
}

bool initialize_hsa(const char* role)
{
    hsa_status_t status = hsa_init();
    if (!check_status(role, "hsa_init", status)) {
        return false;
    }

    g_gpu_agents.clear();
    status = hsa_iterate_agents(agent_callback, nullptr);
    if (!check_status(role, "hsa_iterate_agents", status)) {
        return false;
    }

    std::printf("EVENT role=%s operation=gpu_count value=%zu\n", role,
                g_gpu_agents.size());
    std::fflush(stdout);
    return !g_gpu_agents.empty();
}

bool find_pool(const char* role, unsigned gpu_index,
               hsa_amd_memory_pool_t* pool)
{
    if (gpu_index >= g_gpu_agents.size()) {
        std::fprintf(stderr, "gpu index %u is out of range\n", gpu_index);
        return false;
    }

    PoolSearch search;
    hsa_status_t status = hsa_amd_agent_iterate_memory_pools(
            g_gpu_agents[gpu_index], pool_callback, &search);
    if (!check_status(role, "hsa_amd_agent_iterate_memory_pools", status) ||
        !search.found) {
        return false;
    }

    *pool = search.pool;
    return true;
}

bool write_full(int fd, const void* data, size_t length)
{
    const auto* cursor = static_cast<const unsigned char*>(data);
    while (length > 0) {
        ssize_t written = write(fd, cursor, length);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return false;
        }
        cursor += written;
        length -= static_cast<size_t>(written);
    }
    return true;
}

bool read_full(int fd, void* data, size_t length)
{
    auto* cursor = static_cast<unsigned char*>(data);
    while (length > 0) {
        ssize_t received = read(fd, cursor, length);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            return false;
        }
        cursor += received;
        length -= static_cast<size_t>(received);
    }
    return true;
}

int listen_unix(const std::string& path)
{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.size() >= sizeof(address.sun_path)) {
        close(fd);
        return -1;
    }
    std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
    unlink(path.c_str());
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        listen(fd, 1) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int connect_unix(const std::string& path)
{
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.size() >= sizeof(address.sun_path)) {
        return -1;
    }
    std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);

    for (unsigned attempt = 0; attempt < 200; ++attempt) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd >= 0 &&
            connect(fd, reinterpret_cast<sockaddr*>(&address),
                    sizeof(address)) == 0) {
            return fd;
        }
        if (fd >= 0) {
            close(fd);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return -1;
}

void log_pointer_info(const char* phase, const void* pointer,
                      hsa_amd_pointer_info_t* info)
{
    std::memset(info, 0, sizeof(*info));
    info->size = sizeof(*info);
    hsa_status_t status = hsa_amd_pointer_info(pointer, info, nullptr, nullptr,
                                               nullptr);
    std::printf("EVENT role=importer operation=pointer_info phase=%s status=%d "
                "type=%d base=%p bytes=%zu owner=%" PRIu64 "\n",
                phase, static_cast<int>(status), static_cast<int>(info->type),
                info->agentBaseAddress, info->sizeInBytes,
                info->agentOwner.handle);
    std::fflush(stdout);
}

int run_exporter(const std::string& socket_path, const std::string& mode,
                 unsigned gpu_index, size_t length)
{
    constexpr const char* role = "exporter";
    int listen_fd              = listen_unix(socket_path);
    if (listen_fd < 0 || !initialize_hsa(role)) {
        std::perror("exporter setup");
        return 1;
    }

    hsa_amd_memory_pool_t pool;
    if (!find_pool(role, gpu_index, &pool)) {
        return 1;
    }

    void* allocation    = nullptr;
    hsa_status_t status = hsa_amd_memory_pool_allocate(pool, length, 0,
                                                       &allocation);
    if (!check_status(role, "hsa_amd_memory_pool_allocate", status)) {
        return 1;
    }
    status = hsa_amd_agents_allow_access(1, &g_gpu_agents[gpu_index], nullptr,
                                         allocation);
    if (!check_status(role, "hsa_amd_agents_allow_access", status)) {
        return 1;
    }

    WireHandle wire{};
    wire.magic        = kMagic;
    wire.version      = 1;
    wire.length       = length;
    wire.exporter_gpu = gpu_index;
    wire.mode         = mode == "pre_attach_free" ? 2 :
                        (mode == "stale" ? 1 : 0);
    status = hsa_amd_ipc_memory_create(allocation, length, &wire.ipc);
    if (!check_status(role, "hsa_amd_ipc_memory_create", status)) {
        return 1;
    }

    int peer_fd = accept(listen_fd, nullptr, nullptr);
    if (peer_fd < 0 || !write_full(peer_fd, &wire, sizeof(wire))) {
        return 1;
    }

    char command;
    char importer_state = 0;
    if (!read_full(peer_fd, &importer_state, 1)) {
        return 1;
    }
    if (mode == "pre_attach_free" && importer_state == 'R') {
        status = hsa_amd_memory_pool_free(allocation);
        log_status(role, "hsa_amd_memory_pool_free_before_attach", status);
        allocation = nullptr;
        command    = 'F';
    } else if (mode == "stale" && importer_state == 'A') {
        std::printf("EVENT role=exporter operation=importer_attached value=1\n");
        status = hsa_amd_memory_pool_free(allocation);
        log_status(role, "hsa_amd_memory_pool_free_after_attach", status);
        allocation = nullptr;
        command    = 'F';
    } else if (mode == "valid" && importer_state == 'A') {
        std::printf("EVENT role=exporter operation=importer_attached value=1\n");
        command = 'G';
    } else {
        std::fprintf(stderr, "unexpected importer state %d for mode %s\n",
                     static_cast<int>(importer_state), mode.c_str());
        return 1;
    }
    std::fflush(stdout);
    if (!write_full(peer_fd, &command, 1)) {
        return 1;
    }

    char done = 0;
    bool got_result = read_full(peer_fd, &done, 1);
    std::printf("EVENT role=exporter operation=importer_result value=%d\n",
                got_result ? static_cast<int>(done) : -1);

    if (allocation != nullptr) {
        status = hsa_amd_memory_pool_free(allocation);
        log_status(role, "hsa_amd_memory_pool_free_after_copy", status);
    }
    close(peer_fd);
    close(listen_fd);
    unlink(socket_path.c_str());
    hsa_shut_down();
    return 0;
}

int run_importer(const std::string& socket_path, unsigned gpu_index,
                 unsigned wait_ms)
{
    constexpr const char* role = "importer";
    if (!initialize_hsa(role)) {
        return 1;
    }

    int peer_fd = connect_unix(socket_path);
    if (peer_fd < 0) {
        std::perror("connect");
        return 1;
    }

    WireHandle wire{};
    if (!read_full(peer_fd, &wire, sizeof(wire)) || wire.magic != kMagic ||
        wire.version != 1) {
        return 1;
    }

    hsa_amd_memory_pool_t destination_pool;
    if (!find_pool(role, gpu_index, &destination_pool)) {
        return 1;
    }

    char command = 0;
    if (wire.mode == 2) {
        const char ready = 'R';
        if (!write_full(peer_fd, &ready, 1) ||
            !read_full(peer_fd, &command, 1)) {
            return 1;
        }
    }

    void* mapped = nullptr;
    hsa_status_t status = hsa_amd_ipc_memory_attach(
            &wire.ipc, wire.length, 1, &g_gpu_agents[gpu_index], &mapped);
    check_status(role, "hsa_amd_ipc_memory_attach", status);
    if (status != HSA_STATUS_SUCCESS) {
        const char attach_error = 'E';
        write_full(peer_fd, &attach_error, 1);
        close(peer_fd);
        hsa_shut_down();
        return 3;
    }

    hsa_amd_pointer_info_t before_info{};
    log_pointer_info("after_attach", mapped, &before_info);
    hsa_agent_t source_agent = before_info.agentOwner;

    void* destination = nullptr;
    status = hsa_amd_memory_pool_allocate(destination_pool, wire.length, 0,
                                          &destination);
    if (!check_status(role, "hsa_amd_memory_pool_allocate_dst", status)) {
        return 1;
    }
    status = hsa_amd_agents_allow_access(1, &g_gpu_agents[gpu_index], nullptr,
                                         destination);
    if (!check_status(role, "hsa_amd_agents_allow_access_dst", status)) {
        return 1;
    }

    hsa_signal_t completion{};
    status = hsa_signal_create(1, 0, nullptr, &completion);
    if (!check_status(role, "hsa_signal_create", status)) {
        return 1;
    }

    if (wire.mode != 2) {
        const char attached = 'A';
        if (!write_full(peer_fd, &attached, 1) ||
            !read_full(peer_fd, &command, 1)) {
            return 1;
        }
    }

    hsa_amd_pointer_info_t current_info{};
    const char* pointer_phase = wire.mode == 2 ? "attach_after_exporter_free" :
                                (command == 'F' ? "after_exporter_free" :
                                 "before_copy");
    log_pointer_info(pointer_phase, mapped, &current_info);

    status = hsa_amd_memory_async_copy(
            destination, g_gpu_agents[gpu_index], mapped, source_agent,
            wire.length, 0, nullptr, completion);
    log_status(role, "hsa_amd_memory_async_copy", status);

    hsa_signal_value_t signal_value = hsa_signal_load_scacquire(completion);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(wait_ms);
    while (status == HSA_STATUS_SUCCESS && signal_value > 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        signal_value = hsa_signal_load_scacquire(completion);
    }
    std::printf("EVENT role=importer operation=completion_signal value=%" PRId64
                " wait_ms=%u terminal=%d\n",
                static_cast<int64_t>(signal_value), wait_ms,
                signal_value <= 0 ? 1 : 0);
    std::fflush(stdout);

    char result = signal_value == 0 ? '0' : (signal_value < 0 ? 'N' : 'P');
    write_full(peer_fd, &result, 1);

    if (signal_value > 0) {
        // There is no cancellation acknowledgement. Exiting avoids releasing
        // memory or the signal while an SDMA operation may still be in flight.
        close(peer_fd);
        std::_Exit(5);
    }

    hsa_signal_destroy(completion);
    hsa_amd_memory_pool_free(destination);
    hsa_amd_ipc_memory_detach(mapped);
    close(peer_fd);
    hsa_shut_down();
    return signal_value == 0 ? 0 : 4;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  %s exporter SOCKET valid|stale|pre_attach_free GPU "
                     "SIZE_BYTES\n"
                     "  %s importer SOCKET GPU WAIT_MS\n",
                     argv[0], argv[0]);
        return 2;
    }

    std::string role        = argv[1];
    std::string socket_path = argv[2];
    if (role == "exporter" && argc == 6) {
        return run_exporter(socket_path, argv[3],
                            static_cast<unsigned>(std::stoul(argv[4])),
                            static_cast<size_t>(std::stoull(argv[5])));
    }
    if (role == "importer" && argc == 5) {
        return run_importer(socket_path,
                            static_cast<unsigned>(std::stoul(argv[3])),
                            static_cast<unsigned>(std::stoul(argv[4])));
    }

    std::fprintf(stderr, "invalid arguments\n");
    return 2;
}
