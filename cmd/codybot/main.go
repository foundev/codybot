package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	defaultBaseURL = "http://localhost:11434/v1"
	defaultModel   = "qwen3-coder"
)

type appState int

const (
	stateSetup appState = iota
	stateChat
)

type config struct {
	BaseURL   string
	Model     string
	APIKey    string
	AgentPath string
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature float64   `json:"temperature,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
}

type Tool struct {
	Type     string              `json:"type"`
	Function *FunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Parameters  *FunctionParameters `json:"parameters"`
}

type FunctionParameters struct {
	Type       string                      `json:"type"`
	Properties map[string]FunctionProperty `json:"properties"`
	Required   []string                    `json:"required"`
}

type FunctionProperty struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

type streamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

type streamMsg struct {
	token string
	done  bool
	err   error
}

type model struct {
	state appState

	cfg          config
	system       message
	history      []message
	agentContent string

	viewport   viewport.Model
	input      textarea.Model
	spinner    spinner.Model
	transcript string

	streaming            bool
	streamCh             chan streamMsg
	currentResponse      *strings.Builder
	currentResponseMutex *sync.Mutex
	lastErr              error

	width  int
	height int
}

func main() {
	cfg := parseConfig()

	agentExists := fileExists(cfg.AgentPath)
	agentContent := ""
	if agentExists {
		data, err := os.ReadFile(cfg.AgentPath)
		if err == nil {
			agentContent = string(data)
		}
	}

	initialState := stateChat
	if !agentExists {
		initialState = stateSetup
	}

	program := tea.NewProgram(
		newModel(cfg, agentContent, initialState),
		tea.WithAltScreen(),
	)
	if _, err := program.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "codybot error: %v\n", err)
		os.Exit(1)
	}
}

func parseConfig() config {
	cfg := config{}
	flag.StringVar(&cfg.BaseURL, "base-url", envOrDefault("OPENAI_BASE_URL", defaultBaseURL), "Base URL for an OpenAI-compatible API")
	flag.StringVar(&cfg.Model, "model", envOrDefault("CODYBOT_MODEL", defaultModel), "Model name")
	flag.StringVar(&cfg.APIKey, "api-key", envOrDefault("OPENAI_API_KEY", ""), "API key for the endpoint")
	flag.StringVar(&cfg.AgentPath, "agents", envOrDefault("CODYBOT_AGENTS", "agents.md"), "Path to agents.md")
	flag.Parse()
	return cfg
}

func envOrDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func newModel(cfg config, agentContent string, state appState) model {
	ta := textarea.New()
	ta.Placeholder = "Describe what you want to build..."
	ta.Prompt = "> "
	ta.CharLimit = 8000
	ta.SetHeight(3)
	ta.Focus()

	spin := spinner.New()
	spin.Spinner = spinner.Dot
	spin.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("69"))
	var mutex sync.Mutex
	m := model{
		state:                state,
		cfg:                  cfg,
		agentContent:         agentContent,
		input:                ta,
		viewport:             viewport.New(0, 0),
		spinner:              spin,
		currentResponse:      &strings.Builder{},
		currentResponseMutex: &mutex,
	}
	m.system = message{
		Role:    "system",
		Content: buildSystemPrompt(agentContent),
	}
	m.history = []message{m.system}
	return m
}

func buildSystemPrompt(agentContent string) string {
	base := "You are Codybot, a CLI coding agent. Be concise and practical. Ask clarifying questions only when required."
	if strings.TrimSpace(agentContent) == "" {
		return base
	}
	return fmt.Sprintf("%s\n\nProject instructions (agents.md):\n%s", base, agentContent)
}

func (m model) Init() tea.Cmd {
	if m.state == stateSetup {
		return nil
	}
	return tea.Batch(m.spinner.Tick, textarea.Blink)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if m.state == stateSetup {
			return m.updateSetup(msg)
		}
		handled, cmd := m.updateChatKeys(msg)
		if handled {
			return m, cmd
		}
	case tea.WindowSizeMsg:
		m = m.applySize(msg.Width, msg.Height)
		return m, nil
	case streamMsg:
		return m.handleStreamMsg(msg)
	case spinner.TickMsg:
		if m.streaming {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			return m, cmd
		}
		return m, nil
	}

	if m.state == stateChat {
		var cmds []tea.Cmd
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		cmds = append(cmds, cmd)
		m.viewport, cmd = m.viewport.Update(msg)
		cmds = append(cmds, cmd)
		return m, tea.Batch(cmds...)
	}

	return m, nil
}

func (m model) updateSetup(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "y", "Y":
		if err := writeAgentsTemplate(m.cfg.AgentPath); err != nil {
			m.lastErr = err
		} else if data, err := os.ReadFile(m.cfg.AgentPath); err == nil {
			m.agentContent = string(data)
			m.system = message{Role: "system", Content: buildSystemPrompt(m.agentContent)}
			m.history = []message{m.system}
		}
		m.state = stateChat
		return m, tea.Batch(m.spinner.Tick, textarea.Blink)
	case "n", "N":
		m.state = stateChat
		return m, tea.Batch(m.spinner.Tick, textarea.Blink)
	case "ctrl+c", "esc":
		return m, tea.Quit
	}
	return m, nil
}

func (m *model) updateChatKeys(msg tea.KeyMsg) (bool, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "esc":
		return true, tea.Quit
	case "ctrl+l":
		m.transcript = ""
		m.currentResponseMutex.Lock()
		m.currentResponse.Reset()
		m.currentResponseMutex.Unlock()
		m.history = []message{m.system}
		m.viewport.SetContent(m.transcript)
		return true, nil
	case "enter":
		if m.streaming {
			return true, nil
		}
		text := strings.TrimSpace(m.input.Value())
		if text == "" {
			return true, nil
		}
		m.input.Reset()
		m.appendTranscript(fmt.Sprintf("You: %s\n\nAssistant: ", text))
		m.history = append(m.history, message{Role: "user", Content: text})
		m.streaming = true
		m.lastErr = nil
		m.currentResponseMutex.Lock()
		m.currentResponse.Reset()
		m.currentResponseMutex.Unlock()
		m.streamCh = make(chan streamMsg)
		go streamCompletion(context.Background(), m.cfg, m.history, m.streamCh)
		return true, tea.Batch(waitStream(m.streamCh), m.spinner.Tick)
	}
	return false, nil
}

func (m model) handleStreamMsg(msg streamMsg) (tea.Model, tea.Cmd) {
	if msg.err != nil {
		m.streaming = false
		m.lastErr = msg.err
		m.appendTranscript(fmt.Sprintf("\n\n[error] %s\n\n", msg.err.Error()))
		return m, nil
	}

	if msg.done {
		m.streaming = false
		m.appendTranscript("\n\n")
		m.currentResponseMutex.Lock()
		if response := m.currentResponse.String(); strings.TrimSpace(response) != "" {
			m.history = append(m.history, message{Role: "assistant", Content: response})
		}
		m.currentResponseMutex.Unlock()
		return m, nil
	}

	if msg.token != "" {
		m.appendTranscript(msg.token)
		m.currentResponseMutex.Lock()
		m.currentResponse.WriteString(msg.token)
		m.currentResponseMutex.Unlock()
	}

	if m.streaming {
		return m, waitStream(m.streamCh)
	}
	return m, nil
}

func (m *model) appendTranscript(text string) {
	m.transcript += text
	m.viewport.SetContent(m.transcript)
	m.viewport.GotoBottom()
}

func (m model) View() string {
	if m.state == stateSetup {
		return m.viewSetup()
	}
	return m.viewChat()
}

func (m model) viewSetup() string {
	title := headerStyle.Render("codybot setup")
	body := "No agents.md found. Create one now? (y/n)"
	if m.lastErr != nil {
		body = fmt.Sprintf("%s\n\nLast error: %s", body, m.lastErr.Error())
	}
	hint := "You can edit it later to steer the agent."
	return lipgloss.JoinVertical(lipgloss.Left, title, body, hint)
}

func (m model) viewChat() string {
	border := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(0, 1)

	header := headerStyle.Render("codybot")
	subtitle := subtleStyle.Render(fmt.Sprintf("%s @ %s", m.cfg.Model, m.cfg.BaseURL))
	headerLine := lipgloss.JoinHorizontal(lipgloss.Left, header, " ", subtitle)

	status := m.statusLine()
	outputBox := border.Width(m.width).Render(m.viewport.View())
	inputBox := border.Width(m.width).Render(m.input.View())

	return lipgloss.JoinVertical(lipgloss.Left, headerLine, status, outputBox, inputBox)
}

func (m model) statusLine() string {
	status := "Ready"
	if m.streaming {
		status = fmt.Sprintf("%s Streaming from %s", m.spinner.View(), m.cfg.Model)
	}
	if m.lastErr != nil {
		status = fmt.Sprintf("Error: %s", m.lastErr.Error())
	}
	help := "Enter to send • Ctrl+L to clear • Esc to quit"
	return lipgloss.JoinHorizontal(lipgloss.Left, subtleStyle.Render(status), "  ", subtleStyle.Render(help))
}

func (m model) applySize(width, height int) model {
	m.width = width
	m.height = height
	contentWidth := max(20, width-4)
	m.input.SetWidth(contentWidth)
	m.input.SetHeight(3)

	headerHeight := 1
	statusHeight := 1
	inputHeight := m.input.Height() + 2
	available := height - headerHeight - statusHeight - inputHeight - 2
	available = max(available, 5)
	m.viewport = viewport.New(contentWidth, available)
	m.viewport.SetContent(m.transcript)
	m.viewport.GotoBottom()
	return m
}

func waitStream(ch <-chan streamMsg) tea.Cmd {
	return func() tea.Msg {
		return <-ch
	}
}

func streamCompletion(ctx context.Context, cfg config, history []message, ch chan<- streamMsg) {
	url := strings.TrimRight(cfg.BaseURL, "/") + "/chat/completions"
	payload := chatCompletionRequest{
		Model:       cfg.Model,
		Messages:    history,
		Stream:      true,
		Temperature: 0.2,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		ch <- streamMsg{err: err}
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		ch <- streamMsg{err: err}
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(cfg.APIKey) != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		ch <- streamMsg{err: err}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		ch <- streamMsg{err: fmt.Errorf("API error: %s - %s", resp.Status, strings.TrimSpace(string(body)))}
		return
	}

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if errorsIsEOF(err) {
				ch <- streamMsg{done: true}
				return
			}
			ch <- streamMsg{err: err}
			return
		}

		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			ch <- streamMsg{done: true}
			return
		}

		var payload streamResponse
		if err := json.Unmarshal([]byte(data), &payload); err != nil {
			continue
		}

		for _, choice := range payload.Choices {
			if choice.Delta.Content != "" {
				ch <- streamMsg{token: choice.Delta.Content}
			}
			if choice.FinishReason != "" {
				ch <- streamMsg{done: true}
				return
			}
		}
	}
}

func writeAgentsTemplate(path string) error {
	dir := filepath.Dir(path)
	if dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	content := defaultAgentsTemplate()
	return os.WriteFile(path, []byte(content), 0o644)
}

func defaultAgentsTemplate() string {
	return `# agents.md

	## Mission
	You are Codybot, a CLI coding agent. Keep responses concise and practical.

	## Project context
	- Describe the product and stack here.
	- Note any constraints or policies.

	## Workflow
	- Prefer small, safe changes.
	- Call out risks and unknowns.
	- Summarize steps taken after each change.
	`
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func errorsIsEOF(err error) bool {
	return err == io.EOF || strings.Contains(err.Error(), "closed network connection")
}

var (
	headerStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("212"))
	subtleStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
)
