package api

import (
	"encoding/json"

	"net/http/httptest"
	"testing"

	"github.com/whysooharsh/text-search-engine/index"
)

func testServer() *Server {
	idx := index.New()
	idx.Add(index.Document{ID: 1, Title: "Go", URL: "http://go", Text: "Go is a compiled language"})
	idx.Add(index.Document{ID: 2, Title: "Python", Text: "Python is an interpreted language"})
	idx.Finalize()
	return New(idx)
}

func get(s *Server, url string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("GET", url, nil))
	return rec
}

func TestSearchEndpoint(t *testing.T) {
	s := testServer()
	rec := get(s, "/search?q=language")
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var resp searchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Mode != "bm25" || resp.Total != 2 || len(resp.Results) != 2 || resp.Results[0].Snippet == "" || resp.Results[0].Text == "" {
		t.Errorf("unexpected response: %+v", resp)
	}
}

func TestSearchStatusCodes(t *testing.T) {
	s := testServer()
	for url, want := range map[string]int{
		"/search":                 400,
		"/search?q=":              400,
		"/search?q=%20%20":        400,
		"/search?q=go&k=0":        400,
		"/search?q=go&k=abc":      400,
		"/search?q=go&k=-1":       400,
		"/search?q=go&k=9999":     200,
		"/search?q=go&mode=fuzzy": 400,
		"/search?q=go&mode=and":   200,
		"/search?q=zzzunknown":    200,
		"/doc/1":                  200,
		"/doc/99":                 404,
		"/doc/abc":                400,
		"/health":                 200,
		"/stats":                  200,
	} {
		if rec := get(s, url); rec.Code != want {
			t.Errorf("%s = %d, want %d", url, rec.Code, want)
		}
	}
}

func TestKCap(t *testing.T) {
	idx := index.New()
	for i := uint32(1); i <= 60; i++ {
		idx.Add(index.Document{ID: i, Title: "t", Text: "common"})
	}
	idx.Finalize()
	var resp searchResponse
	json.Unmarshal(get(New(idx), "/search?q=common&k=1000").Body.Bytes(), &resp)
	if len(resp.Results) != maxK || resp.Total != 60 {
		t.Errorf("got %d results total %d, want %d / 60", len(resp.Results), resp.Total, maxK)
	}
}

func TestUnknownQueryEmptyResultsArray(t *testing.T) {
	rec := get(testServer(), "/search?q=zzzunknown")
	var raw map[string]json.RawMessage
	json.Unmarshal(rec.Body.Bytes(), &raw)
	if string(raw["results"]) != "[]" || rec.Header().Get("Content-Type") != "application/json" {
		t.Errorf("results = %s", raw["results"])
	}

}
