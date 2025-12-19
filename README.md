## A basic agent

It uses Ollama API along with Qwen3 7b model.
Currently, it has the following features:
- Interactive chat via command line
- Tool calling capabilities
  - File operations (read, list, edit files)
  - Get current time
  - Run arbitrary shell commands via tool (run_shell) with timeout and output size limits

## Usage

```bash
$ go run main.go
Chat with Ollama
You: what is the time now?
Tool:  time_now with args map[]
Ollama: The current time is 2025-08-31T12:44:54+02:00.
You: can you show me the first 10 lines of main.go file?
Tool:  read_file with args map[path:main.go]
Tool:  read_file with args map[path:main.go]
Ollama: Here are the first 10 lines of `main.go`:

package main

import (
        "bufio"
        "bytes"
        "context"
        "encoding/json"
        "errors"
        "fmt"
        "io"
        "net/http"
        "os"
        "path/filepath"
)
```

### run_shell tool

The agent exposes a tool named `run_shell` that allows executing arbitrary shell commands.

- Parameters:
  - `command` (string, required): The shell command to execute.
  - `timeout_sec` (integer, optional, default 30): Max time to allow the command to run.
- Behavior:
  - Executes via `sh -c` so you can use shell features like pipes and redirection.
  - Captures combined stdout and stderr, limited to 1MB; output beyond that is truncated.
  - Returns an exit code and the combined output.

Example interaction:

```text
You: list files in the repository
Tool:  run_shell with args map[command:ls -1]
Ollama: Here are the files in the repository: ...
```

### fetch_url tool

Fetch the content of a webpage via HTTP GET.

- Parameters:
  - `url` (string, required): The HTTP/HTTPS URL to fetch.
  - `timeout_sec` (integer, optional, default 20): Per-request timeout in seconds.
- Behavior:
  - Supports only `http` and `https` schemes; rejects others.
  - Sends a simple `User-Agent: KutAgent/1.0` and `Accept: */*`.
  - Response is returned as a string prefixed with status code and content type, e.g., `status=200 content_type="text/html; charset=UTF-8"` followed by a newline and the body.
  - The body is limited to 1MB; larger responses are truncated, and a notice is appended.

Example tool return format:

```text
status=200 content_type="text/html; charset=UTF-8"
<!doctype html>...
```