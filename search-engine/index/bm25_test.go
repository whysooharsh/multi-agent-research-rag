package index

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"
)

func bm25Corpus() *Index {
	idx := New()
	idx.Add(Document{ID: 1, Text: "apple banana apple"})
	idx.Add(Document{ID: 2, Text: "banana cherry"})
	idx.Add(Document{ID: 3, Text: "cherry cherry cherry date"})
	idx.Finalize()
	return idx
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestBM25HandComputed(t *testing.T) {
	idx := bm25Corpus()
	l16 := math.Log(1.6)

	hits, total := idx.Search("apple", 10, ModeBM25)
	if total != 1 || len(hits) != 1 || hits[0].Doc.ID != 1 {
		t.Fatalf("apple: hits=%v total=%d", hits, total)
	}
	if want := math.Log(8.0/3) * 4.4 / 3.2; !near(hits[0].Score, want) {
		t.Errorf("apple score = %v, want %v", hits[0].Score, want)
	}

	want := map[uint32]float64{
		1: l16 * (1 * 2.2 / (1 + 1.2)),
		2: l16 * (2.2/(1+0.9) + 2.2/(1+0.9)),
		3: l16 * (3 * 2.2 / (3 + 1.5)),
	}
	hits, total = idx.Search("banana cherry", 10, ModeBM25)
	if total != 3 {
		t.Fatalf("total = %d, want 3", total)
	}
	for _, h := range hits {
		if !near(h.Score, want[h.Doc.ID]) {
			t.Errorf("doc %d score = %v, want %v", h.Doc.ID, h.Score, want[h.Doc.ID])
		}
	}
	if got := []uint32{hits[0].Doc.ID, hits[1].Doc.ID, hits[2].Doc.ID}; got[0] != 2 || got[1] != 3 || got[2] != 1 {
		t.Errorf("order = %v, want [2 3 1]", got)
	}
}

func TestANDModeRegression(t *testing.T) {
	idx := bm25Corpus()
	hits, total := idx.Search("banana cherry", 10, ModeAND)
	if total != 1 || len(hits) != 1 || hits[0].Doc.ID != 2 {
		t.Fatalf("and: hits=%v total=%d, want only doc 2", hits, total)
	}
	if want := math.Log(1.6) * 2 * 2.2 / 1.9; !near(hits[0].Score, want) {
		t.Errorf("and score = %v, want %v", hits[0].Score, want)
	}
	if hits, total = idx.Search("apple cherry", 10, ModeAND); total != 0 || len(hits) != 0 {
		t.Errorf("disjoint AND should be empty, got %v", hits)
	}
	if hits, _ = idx.Search("apple zzzunknown", 10, ModeAND); len(hits) != 0 {
		t.Errorf("AND with unknown term should be empty, got %v", hits)
	}
	if _, total = idx.Search("apple zzzunknown", 10, ModeBM25); total != 1 {
		t.Errorf("OR total = %d, want 1", total)
	}
}

func TestEmptyAndUnknownQueries(t *testing.T) {
	idx := bm25Corpus()
	for _, q := range []string{"", "   ", "the of and", "zzzunknown", "!!!"} {
		for _, mode := range []string{ModeBM25, ModeAND} {
			if hits, total := idx.Search(q, 10, mode); len(hits) != 0 || total != 0 {
				t.Errorf("Search(%q,%s) = %v,%d, want empty", q, mode, hits, total)
			}
		}
	}
	if hits, total := New().Search("apple", 10, ModeBM25); len(hits) != 0 || total != 0 {
		t.Error("empty index should return nothing")
	}
}

func TestKBounds(t *testing.T) {
	idx := bm25Corpus()
	if hits, total := idx.Search("banana cherry", 1, ModeBM25); len(hits) != 1 || total != 3 || hits[0].Doc.ID != 2 {
		t.Errorf("k=1: hits=%v total=%d", hits, total)
	}
	if hits, total := idx.Search("banana cherry", 100, ModeBM25); len(hits) != 3 || total != 3 {
		t.Errorf("k=100: got %d hits total %d", len(hits), total)
	}
	if hits, _ := idx.Search("banana cherry", 0, ModeBM25); len(hits) != 3 {
		t.Errorf("k=0 (all): got %d hits", len(hits))
	}
}

func TestTopKMatchesFullSort(t *testing.T) {
	idx, _ := generate(500, rand.New(rand.NewSource(1)))
	all, total := idx.Search("w1 w2 w3", 0, ModeBM25)
	if !sort.SliceIsSorted(all, func(i, j int) bool { return all[i].Score > all[j].Score }) {
		t.Fatal("full result not sorted by score")
	}
	for _, k := range []int{1, 5, 17} {
		top, tot := idx.Search("w1 w2 w3", k, ModeBM25)
		if tot != total || len(top) != k {
			t.Fatalf("k=%d: len=%d total=%d", k, len(top), tot)
		}
		for i := range top {
			if top[i].Doc.ID != all[i].Doc.ID {
				t.Errorf("k=%d rank %d: doc %d, want %d", k, i, top[i].Doc.ID, all[i].Doc.ID)
			}
		}
	}
}

func generate(n int, rng *rand.Rand) (*Index, *rand.Zipf) {
	zipf := rand.NewZipf(rng, 1.1, 10, 19999)
	idx := New()
	buf := make([]byte, 0, 1024)
	for i := 1; i <= n; i++ {
		buf = buf[:0]
		for j := 0; j < 80+rng.Intn(40); j++ {
			buf = fmt.Appendf(buf, "w%d ", zipf.Uint64())
		}
		idx.Add(Document{ID: uint32(i), Title: fmt.Sprintf("doc %d", i), Text: string(buf)})
	}
	idx.Finalize()
	return idx, zipf
}

func BenchmarkSearch10k(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	idx, zipf := generate(10000, rng)

	queries := make([]string, 1000)
	for i := range queries {
		queries[i] = fmt.Sprintf("w%d w%d w%d", zipf.Uint64(), zipf.Uint64(), zipf.Uint64())
	}

	for _, mode := range []string{ModeBM25, ModeAND} {
		b.Run(mode, func(b *testing.B) {
			lat := make([]time.Duration, 0, b.N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				t0 := time.Now()
				idx.Search(q, 10, mode)
				lat = append(lat, time.Since(t0))
			}
			b.StopTimer()
			sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
			p50, p95 := lat[len(lat)*50/100], lat[len(lat)*95/100]
			b.ReportMetric(float64(p50.Nanoseconds())/1000, "p50-µs")
			b.ReportMetric(float64(p95.Nanoseconds())/1000, "p95-µs")
			fmt.Printf("10k docs, mode=%s, n=%d: p50=%s p95=%s\n", mode, len(lat), p50, p95)
		})
	}
}
