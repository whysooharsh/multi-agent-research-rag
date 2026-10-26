package index

import (
	"container/heap"
	"math"
	"sort"
)

const (
	ModeBM25 = "bm25"
	ModeAND  = "and"

	k1 = 1.2
	b  = 0.75
)

type Hit struct {
	Doc   Document
	Score float64
}

func intersect(a, b []Posting) []Posting {
	var out []Posting
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i].DocID == b[j].DocID:
			out = append(out, a[i])
			i++
			j++

		case a[i].DocID < b[j].DocID:
			i++

		default:
			j++
		}
	}
	return out
}

func sortByLen(lists [][]Posting) {
	for i := 1; i < len(lists); i++ {
		for j := i; j > 0 && len(lists[j]) < len(lists[j-1]); j-- {
			lists[j], lists[j-1] = lists[j-1], lists[j]
		}
	}
}

func uniqueTerms(q string) []string {
	terms := Analyze(q)
	seen := make(map[string]struct{}, len(terms))
	out := terms[:0]
	for _, t := range terms {
		if _, dup := seen[t]; !dup {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}
	return out
}

func (idx *Index) andCandidates(terms []string) map[uint32]struct{} {
	lists := make([][]Posting, 0, len(terms))
	for _, term := range terms {
		p := idx.Postings(term)
		if len(p) == 0 {
			return nil
		}
		lists = append(lists, p)
	}
	sortByLen(lists)

	result := lists[0]
	for _, list := range lists[1:] {
		result = intersect(result, list)
		if len(result) == 0 {
			return nil
		}
	}
	set := make(map[uint32]struct{}, len(result))
	for _, p := range result {
		set[p.DocID] = struct{}{}
	}
	return set
}

func idf(n, df int) float64 {
	return math.Log(1 + (float64(n)-float64(df)+0.5)/(float64(df)+0.5))
}

func (idx *Index) Search(query string, k int, mode string) ([]Hit, int) {
	terms := uniqueTerms(query)
	if len(terms) == 0 {
		return nil, 0
	}

	var allowed map[uint32]struct{}
	if mode == ModeAND {
		if allowed = idx.andCandidates(terms); allowed == nil {
			return nil, 0
		}
	}

	n, avgdl := idx.N(), idx.AvgDL()
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	scores := make(map[uint32]float64)
	for _, term := range terms {
		plist := idx.postings[term]
		w := idf(n, len(plist))
		for _, p := range plist {
			if allowed != nil {
				if _, ok := allowed[p.DocID]; !ok {
					continue
				}
			}
			tf := float64(p.TF)
			dl := float64(idx.docs[p.DocID].Len)
			scores[p.DocID] += w * tf * (k1 + 1) / (tf + k1*(1-b+b*dl/avgdl))
		}
	}

	total := len(scores)
	if k <= 0 || k > total {
		k = total
	}

	h := &minHeap{}
	for id, s := range scores {
		e := entry{id, s}
		if h.Len() < k {
			heap.Push(h, e)
		} else if k > 0 && worse((*h)[0], e) {
			(*h)[0] = e
			heap.Fix(h, 0)
		}
	}
	sort.Slice(*h, func(i, j int) bool { return worse((*h)[j], (*h)[i]) })

	hits := make([]Hit, len(*h))
	for i, e := range *h {
		hits[i] = Hit{Doc: idx.docs[e.id], Score: e.score}
	}
	return hits, total
}

type entry struct {
	id    uint32
	score float64
}

func worse(a, b entry) bool {
	if a.score != b.score {
		return a.score < b.score
	}
	return a.id > b.id
}

type minHeap []entry

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return worse(h[i], h[j]) }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(entry)) }
func (h *minHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}
