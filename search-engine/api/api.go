package api

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/whysooharsh/text-search-engine/index"
)

const (
	defaultK = 10
	maxK     = 50
)

type Server struct {
	idx *index.Index
	mux *http.ServeMux
}

type searchResult struct {
	ID      uint32  `json:"id"`
	Title   string  `json:"title"`
	URL     string  `json:"url"`
	Score   float64 `json:"score"`
	Snippet string  `json:"snippet"`
	Text    string  `json:"text"`
}

type searchResponse struct {
	Query   string         `json:"query"`
	Mode    string         `json:"mode"`
	Total   int            `json:"total"`
	TookMS  float64        `json:"took_ms"`
	Results []searchResult `json:"results"`
}

func New(idx *index.Index) *Server {
	s := &Server{idx: idx, mux: http.NewServeMux()}
	s.mux.HandleFunc("GET /search", s.handleSearch)
	s.mux.HandleFunc("GET /doc/{id}", s.handleDoc)
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("GET /stats", s.handleStats)
	return s
}

func (s *Server) Handler() http.Handler { return s.mux }

func (s *Server) ListenAndServe(ctx context.Context, addr string) error {
	srv := &http.Server{Addr: addr, Handler: s.mux, ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return err
	}
	if err := <-errc; !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	query := q.Get("q")
	if strings.TrimSpace(query) == "" {
		writeErr(w, http.StatusBadRequest, "missing q param")
		return
	}

	k := defaultK
	if v := q.Get("k"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			writeErr(w, http.StatusBadRequest, "k must be a positive integer")
			return
		}
		k = min(n, maxK)
	}

	mode := q.Get("mode")
	switch mode {
	case "":
		mode = index.ModeBM25
	case index.ModeBM25, index.ModeAND:
	default:
		writeErr(w, http.StatusBadRequest, "mode must be bm25 or and")
		return
	}

	start := time.Now()
	hits, total := s.idx.Search(query, k, mode)
	elapsed := time.Since(start)

	terms := index.Analyze(query)
	results := make([]searchResult, len(hits))
	for i, h := range hits {
		results[i] = searchResult{
			ID:      h.Doc.ID,
			Title:   h.Doc.Title,
			URL:     h.Doc.URL,
			Score:   h.Score,
			Snippet: index.Snippet(h.Doc.Text, terms),
			Text:    h.Doc.Text,
		}
	}

	log.Printf("SEARCH | query: %-20q | mode: %s | results: %4d | time: %s", query, mode, total, elapsed)
	writeJSON(w, http.StatusOK, searchResponse{
		Query:   query,
		Mode:    mode,
		Total:   total,
		TookMS:  float64(elapsed.Microseconds()) / 1000,
		Results: results,
	})
}

func (s *Server) handleDoc(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseUint(r.PathValue("id"), 10, 32)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid id")
		return
	}
	doc, ok := s.idx.GetDoc(uint32(id))
	if !ok {
		writeErr(w, http.StatusNotFound, "document not found")
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"docs":  s.idx.N(),
		"terms": s.idx.Terms(),
		"avgdl": s.idx.AvgDL(),
	})
}
