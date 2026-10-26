package index

import (
	"sort"
	"sync"
)

type Document struct {
	ID    uint32 `json:"id"`
	Title string `json:"title"`
	URL   string `json:"url"`
	Text  string `json:"text"`
	Len   int    `json:"-"`
}

type Posting struct {
	DocID uint32
	TF    uint32
}

type Index struct {
	mu       sync.RWMutex
	postings map[string][]Posting
	docs     map[uint32]Document
	totalLen int
}

func New() *Index {
	return &Index{
		postings: make(map[string][]Posting),
		docs:     make(map[uint32]Document),
	}
}

// Add indexes doc. Not safe for concurrent use with readers; load first, then serve.
func (idx *Index) Add(doc Document) {
	terms := Analyze(doc.Title + " " + doc.Text)
	doc.Len = len(terms)

	if old, ok := idx.docs[doc.ID]; ok {
		idx.totalLen -= old.Len
	}
	idx.docs[doc.ID] = doc
	idx.totalLen += doc.Len

	tf := make(map[string]uint32, len(terms))
	for _, term := range terms {
		tf[term]++
	}
	for term, n := range tf {
		idx.postings[term] = append(idx.postings[term], Posting{DocID: doc.ID, TF: n})
	}
}

func (idx *Index) Finalize() {
	for _, p := range idx.postings {
		sort.Slice(p, func(i, j int) bool { return p[i].DocID < p[j].DocID })
	}
}

func (idx *Index) Postings(term string) []Posting {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.postings[term]
}

func (idx *Index) GetDoc(id uint32) (Document, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	doc, ok := idx.docs[id]
	return doc, ok
}

func (idx *Index) N() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.docs)
}

func (idx *Index) AvgDL() float64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if len(idx.docs) == 0 {
		return 0
	}
	return float64(idx.totalLen) / float64(len(idx.docs))
}

func (idx *Index) Terms() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.postings)
}
