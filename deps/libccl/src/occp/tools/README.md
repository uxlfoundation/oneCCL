# OCCP Tools

This directory contains standalone tools for testing and using the OCCP (oneCCL Communication Protocol) library.

## Building

To build the OCCP tools:

```bash
cd src/occp/tools
make clean
make release    # Optimized release build
```

## OCCP Test Tool

The main tool is `occp` which can run both server and client components for testing.

### Usage

#### Start a server:
```bash
./build/debug/bin/occp --server <address> <port>
```

Example:
```bash
./build/debug/bin/occp --server 0.0.0.0 5050
```

#### Start clients:
```bash
./build/debug/bin/occp --client <start_rank> <count> <comm_size> <srv_addr> <port>
```

Example:
```bash
hostname -I
# Start 4 clients (ranks 0-3) for a 4-rank communicator
./build/debug/bin/occp --client 0 4 4 10.168.4.56 5050
```

### Environment Variables

The OCCP tools respect the following environment variables:

- `CCL_OCCP_LOG_LEVEL`: Controls logging level (OFF, CRITICAL, ERROR, WARN, INFO, DEBUG, TRACE)
- `CCL_OCCP_SERVER_IO_THREADS`: Number of I/O threads for server (default: 4)
- `CCL_OCCP_SERVER_OP_TIMEOUT`: Operation timeout for server in seconds (default: 120)
- `CCL_OCCP_SERVER_RANKS_PER_THREAD`: Ranks per thread for server (default: 8)
- `CCL_OCCP_CLIENT_IO_THREADS`: Number of I/O threads for client (default: 2)
- `CCL_OCCP_CLIENT_OP_TIMEOUT`: Operation timeout for client in seconds (default: 120)
