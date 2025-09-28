package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	htmllib "html"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unicode"

	htmlnode "golang.org/x/net/html"
)

// Helper functions for HTML content handling
func isHTMLContentType(ct string) bool {
	ct = strings.ToLower(ct)
	if ct == "" {
		return false
	}
	if strings.HasPrefix(ct, "text/html") {
		return true
	}
	return strings.Contains(ct, "html")
}

func normalizeWS(s string) string {
	var b bytes.Buffer
	prevSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
			continue
		}
		b.WriteRune(r)
		prevSpace = false
	}
	res := strings.TrimSpace(b.String())
	return res
}

func stripTagsQuick(s string) string {
	var out strings.Builder
	inTag := false
	for _, r := range s {
		switch r {
		case '<':
			inTag = true
		case '>':
			inTag = false
		default:
			if !inTag {
				out.WriteRune(r)
			}
		}
	}
	return out.String()
}

func htmlToText(data []byte) string {
	n, err := htmlnode.Parse(bytes.NewReader(data))
	if err != nil {
		// Fallback: naive stripping
		return normalizeWS(htmllib.UnescapeString(stripTagsQuick(string(data))))
	}
	var sb strings.Builder
	var walk func(*htmlnode.Node)
	walk = func(nd *htmlnode.Node) {
		if nd == nil {
			return
		}
		if nd.Type == htmlnode.ElementNode {
			// Skip script/style/noscript content
			if nd.Data == "script" || nd.Data == "style" || nd.Data == "noscript" {
				return
			}
		}
		if nd.Type == htmlnode.TextNode {
			sb.WriteString(nd.Data)
			sb.WriteRune(' ')
		}
		for c := nd.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return normalizeWS(htmllib.UnescapeString(sb.String()))
}

type OllamaClient struct {
}

type chatMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	Name       string `json:"name,omitempty"`
}

func newClient() *OllamaClient {
	return &OllamaClient{}
}

func main() {
	client := newClient()
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}
	agent := NewAgent(client, getUserMessage)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Println(err)
	}
}

type Agent struct {
	client         *OllamaClient
	getUserMessage func() (string, bool)
}

func NewAgent(client *OllamaClient, getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

func (agent *Agent) Run(ctx context.Context) error {
	conversations := []chatMessage{}
	fmt.Println("Chat with Ollama")
	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		message, ok := agent.getUserMessage()
		if !ok {
			break
		}
		conversations = append(conversations, chatMessage{Role: "user", Content: message})

		reply, err := agent.runInference(ctx, conversations)
		if err != nil {
			return err
		}

		// add to conversations
		conversations = append(conversations, reply)

		fmt.Printf("\u001b[93mOllama\u001b[0m: %s\n", reply.Content)
	}
	return nil
}

func (agent *Agent) runInference(ctx context.Context, conversations []chatMessage) (chatMessage, error) {
	// Join conversation history into a single prompt for simplicity.
	if len(conversations) == 0 {
		return chatMessage{}, errors.New("no conversation provided")
	}

	model := os.Getenv("OLLAMA_MODEL")
	if model == "" {
		model = "qwen3-16k"
	}
	return sendToOllamaWithModel(ctx, model, conversations)
}

// sendToOllamaWithModel calls Ollama /api/chat with messages, enabling tool calling.
func sendToOllamaWithModel(ctx context.Context, model string, text []chatMessage) (chatMessage, error) {
	if len(text) == 0 {
		return chatMessage{}, errors.New("text must not be empty")
	}

	endpoint := os.Getenv("OLLAMA_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:11434/api/chat"
	}
	// Provide a minimal tool registry: time_now and read_file
	type functionDef struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	}

	// Define tool request/response types following Ollama tool-calling schema
	type toolDef struct {
		Type     string      `json:"type"`
		Function functionDef `json:"function"`
	}

	type toolCall struct {
		ID       string `json:"id,omitempty"`
		Type     string `json:"type,omitempty"`
		Function struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments,omitempty"`
		} `json:"function"`
	}

	type assistantMessage struct {
		Role      string     `json:"role"`
		Content   string     `json:"content,omitempty"`
		ToolCalls []toolCall `json:"tool_calls,omitempty"`
	}

	type chatRequest struct {
		Model    string        `json:"model"`
		Messages []chatMessage `json:"messages"`
		Tools    []toolDef     `json:"tools,omitempty"`
		Stream   bool          `json:"stream"`
		Options  any           `json:"options,omitempty"`
	}

	type chatResponse struct {
		Model      string           `json:"model"`
		CreatedAt  string           `json:"created_at"`
		Message    assistantMessage `json:"message"`
		Done       bool             `json:"done"`
		DoneReason string           `json:"done_reason"`
	}

	tools := []toolDef{
		{
			Type: "function",
			Function: functionDef{
				Name:        "time_now",
				Description: "Return the current local time in RFC3339 format",
				Parameters: map[string]any{
					"type":                 "object",
					"properties":           map[string]any{},
					"additionalProperties": false,
				},
			},
		},
		{
			Type: "function",
			Function: functionDef{
				Name:        "read_file",
				Description: "Read a text file from the current project directory and return its contents. Input: { path: string }",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string"},
					},
					"required":             []string{"path"},
					"additionalProperties": false,
				},
			},
		},
		{
			Type: "function",
			Function: functionDef{
				Name:        "list_files",
				Description: "List all files under the given directory path recursively, returning full paths. Input: { path: string }",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string"},
					},
					"required":             []string{"path"},
					"additionalProperties": false,
				},
			},
		},
		{
			Type: "function",
			Function: functionDef{
				Name:        "edit_file",
				Description: "Create or overwrite a text file at the given path with provided content. Input: { path: string, content: string }",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":    map[string]any{"type": "string"},
						"content": map[string]any{"type": "string"},
					},
					"required":             []string{"path", "content"},
					"additionalProperties": false,
				},
			},
		},
		{
			Type: "function",
			Function: functionDef{
				Name:        "run_shell",
				Description: "Run an arbitrary shell command and return its output, stderr, and exit code. Input: { command: string, timeout_sec?: integer }",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command":     map[string]any{"type": "string"},
						"timeout_sec": map[string]any{"type": "integer"},
					},
					"required":             []string{"command"},
					"additionalProperties": false,
				},
			},
		},
		{
			Type: "function",
			Function: functionDef{
				Name:        "fetch_url",
				Description: "Fetch the content of a webpage via HTTP GET. Input: { url: string, timeout_sec?: integer }",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"url":         map[string]any{"type": "string"},
						"timeout_sec": map[string]any{"type": "integer"},
					},
					"required":             []string{"url"},
					"additionalProperties": false,
				},
			},
		},
	}

	execTool := func(name string, args map[string]any) (string, error) {
		fmt.Printf("\u001B[91mTool\u001B[0m:  %s with args %v\n", name, args)
		switch name {
		case "time_now":
			return time.Now().Format(time.RFC3339), nil
		case "read_file":
			p, _ := args["path"].(string)
			if p == "" {
				return "", fmt.Errorf("missing required argument: path")
			}
			// Sanitize and scope to project root
			root, err := os.Getwd()
			if err != nil {
				return "", fmt.Errorf("getwd: %w", err)
			}
			clean := filepath.Clean(p)
			joined := filepath.Join(root, clean)
			// Ensure the resolved path stays within root
			rootWithSep := root + string(os.PathSeparator)
			if !(joined == root || strings.HasPrefix(joined, rootWithSep)) {
				return "", fmt.Errorf("access outside project root is not allowed")
			}
			fi, err := os.Stat(joined)
			if err != nil {
				return "", fmt.Errorf("stat file: %w", err)
			}
			if fi.IsDir() {
				return "", fmt.Errorf("path is a directory, not a file")
			}
			const maxSize = 1 << 20 // 1MB
			if fi.Size() > maxSize {
				return "", fmt.Errorf("file too large: %d bytes (limit %d)", fi.Size(), maxSize)
			}
			b, err := os.ReadFile(joined)
			if err != nil {
				return "", fmt.Errorf("read file: %w", err)
			}
			return string(b), nil
		case "list_files":
			p, _ := args["path"].(string)
			if p == "" {
				return "", fmt.Errorf("missing required argument: path")
			}
			root, err := os.Getwd()
			if err != nil {
				return "", fmt.Errorf("getwd: %w", err)
			}
			clean := filepath.Clean(p)
			joined := filepath.Join(root, clean)
			rootWithSep := root + string(os.PathSeparator)
			if !(joined == root || strings.HasPrefix(joined, rootWithSep)) {
				return "", fmt.Errorf("access outside project root is not allowed")
			}
			info, err := os.Stat(joined)
			if err != nil {
				return "", fmt.Errorf("stat path: %w", err)
			}
			if !info.IsDir() {
				return "", fmt.Errorf("path is not a directory")
			}
			// Walk directory tree and collect files
			paths := make([]string, 0, 64)
			const maxEntries = 5000
			const maxOutputBytes = 1 << 20 // 1MB
			var totalBytes int
			err = filepath.WalkDir(joined, func(path string, d os.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if d.IsDir() {
					return nil
				}
				// Ensure still under root (defense in depth)
				if !(path == root || strings.HasPrefix(path, rootWithSep)) {
					return nil
				}
				paths = append(paths, path)
				if len(paths) >= maxEntries {
					return filepath.SkipDir
				}
				return nil
			})
			if err != nil {
				return "", fmt.Errorf("walk dir: %w", err)
			}
			// Build output string with size guard
			var b strings.Builder
			for i, fp := range paths {
				if i > 0 {
					b.WriteString("\n")
					totalBytes++
				}
				b.WriteString(fp)
				totalBytes += len(fp)
				if totalBytes > maxOutputBytes {
					b.WriteString("\n... truncated due to output size limit ...")
					break
				}
			}
			return b.String(), nil
		case "run_shell":
			cmdStr, _ := args["command"].(string)
			if cmdStr == "" {
				return "", fmt.Errorf("missing required argument: command")
			}
			// parse optional timeout_sec
			timeoutSec := 30
			if v, ok := args["timeout_sec"]; ok {
				switch t := v.(type) {
				case float64:
					if t > 0 {
						timeoutSec = int(t)
					}
				case int:
					if t > 0 {
						timeoutSec = t
					}
				}
			}
			// run the command via shell
			cctx := ctx
			var cancelCmd context.CancelFunc
			if timeoutSec > 0 {
				cctx, cancelCmd = context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
				defer cancelCmd()
			}
			cmd := exec.CommandContext(cctx, "sh", "-c", cmdStr)
			var outBuf bytes.Buffer
			cmd.Stdout = &outBuf
			cmd.Stderr = &outBuf
			err := cmd.Run()
			exitCode := 0
			if err != nil {
				var ee *exec.ExitError
				if errors.As(err, &ee) {
					exitCode = ee.ExitCode()
				} else {
					exitCode = -1
				}
			}
			output := outBuf.String()
			const maxCmdOutput = 1 << 20 // 1MB
			if len(output) > maxCmdOutput {
				output = output[:maxCmdOutput] + "\n... truncated due to output size limit ..."
			}
			return fmt.Sprintf("exit_code=%d\n%s", exitCode, output), nil
		case "fetch_url":
			urlStr, _ := args["url"].(string)
			if urlStr == "" {
				return "", fmt.Errorf("missing required argument: url")
			}
			// validate URL
			u, err := url.Parse(urlStr)
			if err != nil || u.Scheme == "" || u.Host == "" {
				return "", fmt.Errorf("invalid url")
			}
			if u.Scheme != "http" && u.Scheme != "https" {
				return "", fmt.Errorf("unsupported url scheme: %s", u.Scheme)
			}
			// parse optional timeout
			fetchTimeout := 20
			if v, ok := args["timeout_sec"]; ok {
				switch t := v.(type) {
				case float64:
					if t > 0 {
						fetchTimeout = int(t)
					}
				case int:
					if t > 0 {
						fetchTimeout = t
					}
				}
			}
			cctx := ctx
			var cancel context.CancelFunc
			if fetchTimeout > 0 {
				cctx, cancel = context.WithTimeout(ctx, time.Duration(fetchTimeout)*time.Second)
				defer cancel()
			}
			req, err := http.NewRequestWithContext(cctx, http.MethodGet, urlStr, nil)
			if err != nil {
				return "", fmt.Errorf("create request: %w", err)
			}
			req.Header.Set("Accept", "*/*")
			req.Header.Set("User-Agent", "KutAgent/1.0 (+https://example.com)")
			client := &http.Client{Timeout: 0}
			resp, err := client.Do(req)
			if err != nil {
				return "", fmt.Errorf("request failed: %w", err)
			}
			defer resp.Body.Close()
			const maxBytes = 1 << 20 // 1MB
			lr := io.LimitReader(resp.Body, maxBytes+1)
			data, err := io.ReadAll(lr)
			if err != nil {
				return "", fmt.Errorf("read body: %w", err)
			}
			truncated := len(data) > maxBytes
			if truncated {
				data = data[:maxBytes]
			}
			ct := resp.Header.Get("Content-Type")
			prefix := fmt.Sprintf("status=%d content_type=\"%s\"\n", resp.StatusCode, ct)
			var body string
			if isHTMLContentType(ct) {
				body = htmlToText(data)
			} else {
				body = string(data)
			}
			if truncated {
				body += "\n... truncated due to 32KB limit ..."
			}
			fmt.Println(body)
			return prefix + body, nil
		default:
			return "", fmt.Errorf("unknown tool: %s", name)
		}
	}

	// Create context with timeout to avoid hanging requests
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 60*time.Second)
		defer cancel()
	}

	messages := append([]chatMessage(nil), text...)
	maxSteps := 5
	for step := 0; step < maxSteps; step++ {
		reqBody := chatRequest{
			Model:    model,
			Stream:   false,
			Messages: messages,
			Tools:    tools,
		}
		payload, err := json.Marshal(reqBody)
		if err != nil {
			return chatMessage{}, fmt.Errorf("marshal request: %w", err)
		}

		httpClient := &http.Client{Timeout: 0} // rely on context timeout
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
		if err != nil {
			return chatMessage{}, fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := httpClient.Do(req)
		if err != nil {
			return chatMessage{}, fmt.Errorf("request ollama: %w", err)
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return chatMessage{}, fmt.Errorf("read response: %w", err)
		}
		if resp.StatusCode != http.StatusOK {
			return chatMessage{}, fmt.Errorf("ollama error: status %d, body: %s", resp.StatusCode, string(body))
		}

		var chatResp chatResponse
		if err := json.Unmarshal(body, &chatResp); err != nil {
			return chatMessage{}, fmt.Errorf("decode response: %w; body: %s", err, string(body))
		}

		// If assistant returned tool calls, execute them and continue the loop
		if len(chatResp.Message.ToolCalls) > 0 {
			// Append assistant tool-calling message to history
			messages = append(messages, chatMessage{Role: chatResp.Message.Role, Content: chatResp.Message.Content})
			for _, tc := range chatResp.Message.ToolCalls {
				// Use arguments provided by the model (may be nil)
				args := tc.Function.Arguments
				if args == nil {
					args = map[string]any{}
				}
				result, err := execTool(tc.Function.Name, args)
				if err != nil {
					result = fmt.Sprintf("tool error: %v", err)
				}
				messages = append(messages, chatMessage{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
					Name:       tc.Function.Name,
				})
			}
			continue
		}

		// Final assistant message
		if chatResp.Message.Content != "" {
			return chatMessage{Role: chatResp.Message.Role, Content: chatResp.Message.Content}, nil
		}

		// If done but no content, return generic
		if chatResp.Done {
			return chatMessage{Role: "assistant", Content: ""}, nil
		}
	}

	return chatMessage{}, fmt.Errorf("max tool-calling steps exceeded")
}
