package index

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestAnalyze(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"the quick brown fox", []string{"quick", "brown", "fox"}},
		{"Running runs runner", []string{"run", "run", "runner"}},
		{"", nil},
	}

	for _, tc := range tests {
		got := Analyze(tc.input)
		if len(got) != len(tc.want) {
			t.Errorf("Analyze(%q) = %v, want %v", tc.input, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("Analyze(%q)[%d] = %q, want %q", tc.input, i, got[i], tc.want[i])
			}
		}
	}
}

func TestAnalyzeOffsets(t *testing.T) {
	text := "Café Running, the naïve fox"
	toks := AnalyzeOffsets(text)
	terms := Analyze(text)
	if len(toks) != len(terms) {
		t.Fatalf("got %d tokens, want %d", len(toks), len(terms))
	}
	for i, tok := range toks {
		if tok.Term != terms[i] {
			t.Errorf("token %d term %q != %q", i, tok.Term, terms[i])
		}
	}
	if got := text[toks[1].Start:toks[1].End]; got != "Running" {
		t.Errorf("offset slice = %q, want Running", got)
	}
}

func TestSnippetUnicode(t *testing.T) {
	text := strings.Repeat("日本語のテキスト ", 100) + "needle " + strings.Repeat("héllo wörld ", 100)
	s := Snippet(text, Analyze("needle"))
	if !utf8.ValidString(s) {
		t.Fatal("snippet is not valid UTF-8")
	}
	if !strings.Contains(s, "needle") {
		t.Error("snippet misses the match")
	}
	n := utf8.RuneCountInString(strings.Trim(s, "…"))
	if n < 295 || n > 305 {
		t.Errorf("snippet has %d runes, want ~300", n)
	}
	if short := Snippet("tiny needle", Analyze("needle")); short != "tiny needle" {
		t.Errorf("short text snippet = %q", short)
	}
	if nomatch := Snippet(text, Analyze("zzz")); !strings.HasPrefix(nomatch, "日本語") {
		t.Errorf("no-match snippet should start at text start, got %q", nomatch[:20])
	}
}

func TestPostingsTFAndStats(t *testing.T) {
	idx := New()
	idx.Add(Document{ID: 7, Title: "fox", Text: "fox fox dog"})
	idx.Add(Document{ID: 3, Title: "dog", Text: "cat"})
	idx.Finalize()

	p := idx.Postings("fox")
	if len(p) != 1 || p[0] != (Posting{7, 3}) {
		t.Errorf("fox postings = %v", p)
	}
	p = idx.Postings("dog")
	if len(p) != 2 || p[0] != (Posting{3, 1}) || p[1] != (Posting{7, 1}) {
		t.Errorf("dog postings = %v", p)
	}
	if d, _ := idx.GetDoc(7); d.Len != 4 {
		t.Errorf("doc 7 len = %d, want 4", d.Len)
	}
	if idx.N() != 2 || idx.AvgDL() != 3 {
		t.Errorf("N=%d avgdl=%v, want 2, 3", idx.N(), idx.AvgDL())
	}
}

func TestSearch(t *testing.T) {
	idx := New()
	idx.Add(Document{ID: 1, Title: "Go programming", Text: "Go is a statically typed compiled language"})
	idx.Add(Document{ID: 2, Title: "Python basics", Text: "Python is a dynamically typed interpreted language"})
	idx.Add(Document{ID: 3, Title: "Go web servers", Text: "Go is great for building fast web servers"})
	idx.Finalize()

	tests := []struct {
		query   string
		wantIDs []uint32
	}{
		{"go language", []uint32{1}},
		{"web servers", []uint32{3}},
		{"python", []uint32{2}},
		{"java", nil},
		{"", nil},
	}

	for _, tc := range tests {
		results, _ := idx.Search(tc.query, 0, ModeAND)
		if len(results) != len(tc.wantIDs) {
			t.Errorf("Search(%q) returned %d results, want %d", tc.query, len(results), len(tc.wantIDs))
			continue
		}
		for i, hit := range results {
			if doc := hit.Doc; doc.ID != tc.wantIDs[i] {
				t.Errorf("Search(%q)[%d].ID = %d, want %d", tc.query, i, doc.ID, tc.wantIDs[i])
			}
		}
	}
}
