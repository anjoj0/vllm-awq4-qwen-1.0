#include <hip/hip_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct Options {
    int src_device = 0;
    int dst_device = 1;
    size_t bytes = 64ULL * 1024 * 1024;
    int warmup = 10;
    int iterations = 100;
    bool put = false;
};

struct WireResult {
    double elapsed_ms = 0.0;
    double cpu_elapsed_ms = 0.0;
    int valid = 0;
    int src_can_access_dst = 0;
    int dst_can_access_src = 0;
    int hip_error = 0;
    int observed_head = -1;
    int observed_tail = -1;
};

[[noreturn]] void fail(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(2);
}

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        fail(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void write_full(int fd, const void* buffer, size_t size) {
    const auto* cursor = static_cast<const uint8_t*>(buffer);
    while (size > 0) {
        const ssize_t written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail(std::string("write: ") + std::strerror(errno));
        }
        cursor += written;
        size -= static_cast<size_t>(written);
    }
}

void read_full(int fd, void* buffer, size_t size) {
    auto* cursor = static_cast<uint8_t*>(buffer);
    while (size > 0) {
        const ssize_t received = read(fd, cursor, size);
        if (received == 0) {
            fail("unexpected end of IPC control socket");
        }
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail(std::string("read: ") + std::strerror(errno));
        }
        cursor += received;
        size -= static_cast<size_t>(received);
    }
}

void enable_peer_if_available(int local_device, int peer_device) {
    int can_access = 0;
    check_hip(hipDeviceCanAccessPeer(&can_access, local_device, peer_device),
              "hipDeviceCanAccessPeer");
    if (!can_access) {
        return;
    }
    check_hip(hipSetDevice(local_device), "hipSetDevice");
    const hipError_t status = hipDeviceEnablePeerAccess(peer_device, 0);
    if ((status != hipSuccess) && (status != hipErrorPeerAccessAlreadyEnabled)) {
        check_hip(status, "hipDeviceEnablePeerAccess");
    }
}

bool validate_pattern(void* device_pointer, size_t bytes, uint8_t pattern,
                      int* observed_head, int* observed_tail) {
    constexpr size_t kProbeBytes = 4096;
    const size_t probe_size = std::min(bytes, kProbeBytes);
    auto* host = static_cast<uint8_t*>(std::malloc(probe_size * 2));
    if (host == nullptr) {
        fail("host validation allocation failed");
    }

    check_hip(hipMemcpy(host, device_pointer, probe_size, hipMemcpyDeviceToHost),
              "hipMemcpy validation head");
    if (bytes > probe_size) {
        check_hip(hipMemcpy(host + probe_size,
                           static_cast<uint8_t*>(device_pointer) + bytes - probe_size,
                           probe_size, hipMemcpyDeviceToHost),
                  "hipMemcpy validation tail");
    } else {
        std::memcpy(host + probe_size, host, probe_size);
    }

    *observed_head = host[0];
    *observed_tail = host[(probe_size * 2) - 1];
    bool valid = true;
    for (size_t i = 0; i < probe_size * 2; ++i) {
        valid = valid && (host[i] == pattern);
    }
    std::free(host);
    return valid;
}

struct CopyTiming {
    double event_ms;
    double cpu_ms;
};

CopyTiming run_copy(void* destination, int dst_device, const void* source,
                    int src_device, size_t bytes, int warmup, int iterations,
                    int initiating_device) {
    check_hip(hipSetDevice(initiating_device), "hipSetDevice initiator");
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
              "hipStreamCreateWithFlags");
    check_hip(hipEventCreate(&start), "hipEventCreate start");
    check_hip(hipEventCreate(&stop), "hipEventCreate stop");

    for (int i = 0; i < warmup; ++i) {
        check_hip(hipMemcpyPeerAsync(destination, dst_device, source, src_device,
                                    bytes, stream),
                  "hipMemcpyPeerAsync warmup");
    }
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize warmup");

    const auto cpu_start = std::chrono::steady_clock::now();
    check_hip(hipEventRecord(start, stream), "hipEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        check_hip(hipMemcpyPeerAsync(destination, dst_device, source, src_device,
                                    bytes, stream),
                  "hipMemcpyPeerAsync measured");
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord stop");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize stop");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize measured");
    const auto cpu_stop = std::chrono::steady_clock::now();

    float elapsed_ms = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop),
              "hipEventElapsedTime");
    check_hip(hipEventDestroy(stop), "hipEventDestroy stop");
    check_hip(hipEventDestroy(start), "hipEventDestroy start");
    check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
    const double cpu_ms = std::chrono::duration<double, std::milli>(
                                  cpu_stop - cpu_start)
                                  .count();
    return {static_cast<double>(elapsed_ms), cpu_ms};
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (++i >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[i];
        };
        if (arg == "--src") {
            options.src_device = std::stoi(require_value("--src"));
        } else if (arg == "--dst") {
            options.dst_device = std::stoi(require_value("--dst"));
        } else if (arg == "--bytes") {
            options.bytes = std::stoull(require_value("--bytes"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value("--warmup"));
        } else if (arg == "--iterations") {
            options.iterations = std::stoi(require_value("--iterations"));
        } else if (arg == "--direction") {
            const std::string direction = require_value("--direction");
            if ((direction != "get") && (direction != "put")) {
                fail("--direction must be get or put");
            }
            options.put = direction == "put";
        } else {
            fail("unknown argument: " + arg);
        }
    }
    if ((options.src_device == options.dst_device) || (options.bytes == 0) ||
        (options.warmup < 0) || (options.iterations <= 0)) {
        fail("invalid device, size, warmup, or iteration argument");
    }
    return options;
}

void child_process(int fd, const Options& options) {
    constexpr uint8_t kPattern = 0xa5;
    check_hip(hipSetDevice(options.dst_device), "child hipSetDevice");

    int dst_can_access_src = 0;
    int src_can_access_dst = 0;
    check_hip(hipDeviceCanAccessPeer(&dst_can_access_src, options.dst_device,
                                     options.src_device),
              "child hipDeviceCanAccessPeer dst->src");
    check_hip(hipDeviceCanAccessPeer(&src_can_access_dst, options.src_device,
                                     options.dst_device),
              "child hipDeviceCanAccessPeer src->dst");
    enable_peer_if_available(options.dst_device, options.src_device);

    void* local_destination = nullptr;
    check_hip(hipMalloc(&local_destination, options.bytes), "child hipMalloc");
    check_hip(hipMemset(local_destination, 0, options.bytes), "child hipMemset");
    check_hip(hipDeviceSynchronize(), "child sync destination initialization");

    hipIpcMemHandle_t source_handle{};
    read_full(fd, &source_handle, sizeof(source_handle));

    hipIpcMemHandle_t destination_handle{};
    check_hip(hipIpcGetMemHandle(&destination_handle, local_destination),
              "child hipIpcGetMemHandle destination");
    write_full(fd, &destination_handle, sizeof(destination_handle));

    WireResult result{};
    result.src_can_access_dst = src_can_access_dst;
    result.dst_can_access_src = dst_can_access_src;

    if (!options.put) {
        void* remote_source = nullptr;
        check_hip(hipIpcOpenMemHandle(&remote_source, source_handle,
                                      hipIpcMemLazyEnablePeerAccess),
                  "child hipIpcOpenMemHandle source");
        const CopyTiming timing = run_copy(
                local_destination, options.dst_device, remote_source,
                options.src_device, options.bytes, options.warmup,
                options.iterations, options.dst_device);
        result.elapsed_ms = timing.event_ms;
        result.cpu_elapsed_ms = timing.cpu_ms;
        result.valid = validate_pattern(local_destination, options.bytes, kPattern,
                                        &result.observed_head,
                                        &result.observed_tail);
        check_hip(hipIpcCloseMemHandle(remote_source),
                  "child hipIpcCloseMemHandle source");
        write_full(fd, &result, sizeof(result));
    } else {
        read_full(fd, &result, sizeof(result));
        check_hip(hipDeviceSynchronize(), "child hipDeviceSynchronize put");
        result.valid = validate_pattern(local_destination, options.bytes, kPattern,
                                        &result.observed_head,
                                        &result.observed_tail);
        result.src_can_access_dst = src_can_access_dst;
        result.dst_can_access_src = dst_can_access_src;
        write_full(fd, &result, sizeof(result));
    }

    check_hip(hipFree(local_destination), "child hipFree");
    close(fd);
    _exit(result.valid ? 0 : 3);
}

void parent_process(int fd, pid_t child, const Options& options) {
    constexpr uint8_t kPattern = 0xa5;
    check_hip(hipSetDevice(options.src_device), "parent hipSetDevice");
    enable_peer_if_available(options.src_device, options.dst_device);

    void* local_source = nullptr;
    check_hip(hipMalloc(&local_source, options.bytes), "parent hipMalloc");
    check_hip(hipMemset(local_source, kPattern, options.bytes), "parent hipMemset");
    check_hip(hipDeviceSynchronize(), "parent sync source initialization");

    hipIpcMemHandle_t source_handle{};
    check_hip(hipIpcGetMemHandle(&source_handle, local_source),
              "parent hipIpcGetMemHandle source");
    write_full(fd, &source_handle, sizeof(source_handle));

    hipIpcMemHandle_t destination_handle{};
    read_full(fd, &destination_handle, sizeof(destination_handle));

    WireResult result{};
    if (options.put) {
        void* remote_destination = nullptr;
        check_hip(hipIpcOpenMemHandle(&remote_destination, destination_handle,
                                      hipIpcMemLazyEnablePeerAccess),
                  "parent hipIpcOpenMemHandle destination");
        const CopyTiming timing = run_copy(
                remote_destination, options.dst_device, local_source,
                options.src_device, options.bytes, options.warmup,
                options.iterations, options.src_device);
        result.elapsed_ms = timing.event_ms;
        result.cpu_elapsed_ms = timing.cpu_ms;
        check_hip(hipIpcCloseMemHandle(remote_destination),
                  "parent hipIpcCloseMemHandle destination");
        write_full(fd, &result, sizeof(result));
    }

    read_full(fd, &result, sizeof(result));
    int child_status = 0;
    waitpid(child, &child_status, 0);

    const double seconds = result.cpu_elapsed_ms / 1000.0;
    const double total_gb = static_cast<double>(options.bytes) *
                            static_cast<double>(options.iterations) / 1.0e9;
    const double gbps = seconds > 0.0 ? total_gb / seconds : 0.0;
    std::cout << "{\"direction\":\"" << (options.put ? "put" : "get")
              << "\",\"src\":" << options.src_device
              << ",\"dst\":" << options.dst_device
              << ",\"bytes\":" << options.bytes
              << ",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations
              << ",\"elapsed_ms\":" << result.elapsed_ms
              << ",\"cpu_elapsed_ms\":" << result.cpu_elapsed_ms
              << ",\"bandwidth_GBps\":" << gbps
              << ",\"src_can_access_dst\":" << result.src_can_access_dst
              << ",\"dst_can_access_src\":" << result.dst_can_access_src
              << ",\"valid\":" << (result.valid ? "true" : "false")
              << ",\"observed_head\":" << result.observed_head
              << ",\"observed_tail\":" << result.observed_tail
              << ",\"child_exit\":"
              << (WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1)
              << "}" << std::endl;

    check_hip(hipFree(local_source), "parent hipFree");
    close(fd);
    if (!result.valid || !WIFEXITED(child_status) ||
        (WEXITSTATUS(child_status) != 0)) {
        std::exit(3);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    int sockets[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sockets) != 0) {
        fail(std::string("socketpair: ") + std::strerror(errno));
    }

    const pid_t child = fork();
    if (child < 0) {
        fail(std::string("fork: ") + std::strerror(errno));
    }
    if (child == 0) {
        close(sockets[0]);
        child_process(sockets[1], options);
    }

    close(sockets[1]);
    parent_process(sockets[0], child, options);
    return 0;
}
