# Text Search Engine

A lightweight full-text search engine in Go with BM25 ranking, built to serve as the retriever in a RAG pipeline. Indexes JSONL, `.txt` directories or Wikipedia XML dumps and serves ranked results with snippets over HTTP.

## How it works

1. Loads documents (JSONL, a directory of `.txt` files, or a Wikipedia XML dump)
2. Tokenizes, removes stop words, and stems each document (title + text)
3. Builds an inverted index: term -> `[]Posting{DocID, TF}`, plus a doc table (id, title, url, text, token length), `N` and `avgdl`
4. Ranks with BM25, term-at-a-time score accumulation, top-k via a min-heap
5. `mode=and` first narrows candidates with two-pointer posting-list intersection

## Project structure

```
├── main.go           entry point, CLI flags, graceful shutdown
├── loader/
│   └── loader.go     JSONL, .txt directory and Wikipedia XML loaders
├── index/
│   ├── analyze.go    tokenize, stopword filter, stem; AnalyzeOffsets adds byte offsets
│   ├── index.go      inverted index, doc table, N / avgdl
│   ├── search.go     BM25 ranking, AND intersection, top-k heap
│   └── snippet.go    ~300 character snippet around the first match
├── api/
│   └── api.go        HTTP endpoints
└── docs/             sample .txt documents
```

## Getting started

**Prerequisites:** Go 1.24+

```bash
go run . -dump ./docs            # directory of .txt files
go run . -dump ./corpus.jsonl    # JSONL
go run . -dump ./wiki.xml        # Wikipedia dump
```

```bash
curl "http://localhost:8000/search?q=machine%20learning&k=5"
```

### JSONL format

One object per line. `id` is required and is taken from the file as-is (never generated); `.txt` files have no id, so they are numbered from 1 in filename order.

```json
{
	"id": 42,
	"title": "Go",
	"url": "https://example.com/go",
	"text": "Go is a compiled language..."
}
```

## API

### `GET /search?q=<query>&k=<n>&mode=<bm25|and>`

| Parameter | Default  | Description                                                                                                                |
| --------- | -------- | -------------------------------------------------------------------------------------------------------------------------- |
| `q`       | required | query text; empty or blank returns 400                                                                                     |
| `k`       | `10`     | number of results; values above 50 are capped at 50; non-integers and values < 1 return 400                                |
| `mode`    | `bm25`   | `bm25`: ranked OR, any term matches. `and`: only docs containing every term, still ranked by BM25. Other values return 400 |

**Response**

```json
{
	"query": "go web",
	"mode": "bm25",
	"total": 3,
	"took_ms": 0.021,
	"results": [
		{
			"id": 1,
			"title": "doc1",
			"url": "",
			"score": 1.4877,
			"snippet": "Go is a statically typed ... building web servers ...",
			"text": "<full document text>"
		}
	]
}
```

`total` is the number of matching documents before `k` is applied. An unknown query returns 200 with `total: 0` and `results: []`.

### Other endpoints

| Endpoint        | Description                                                                            |
| --------------- | -------------------------------------------------------------------------------------- |
| `GET /doc/{id}` | the stored document `{id,title,url,text}`; 404 if absent, 400 if the id isn't a uint32 |
| `GET /health`   | `{"status":"ok"}`                                                                      |
| `GET /stats`    | `{"docs":N,"terms":vocabulary size,"avgdl":mean tokens per doc}`                       |

The server shuts down gracefully on SIGINT/SIGTERM, draining in-flight requests for up to 10 s.

## Ranking: BM25

For each query term `t` (duplicates in the query are ignored) and document `d`:

```
score(d) = Σ_t  idf(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )

idf(t)   = ln( 1 + (N − n_t + 0.5) / (n_t + 0.5) )
k1 = 1.2,  b = 0.75
```

`N` is the number of documents, `n_t` the number containing `t`, `|d|` the document's token count after analysis (title + text), `avgdl` the mean of `|d|`. Ties are broken by lower doc id. `snippet` is about 300 characters centred on the first query-term match (falling back to the start of the text), cut on rune boundaries, with `…` marking truncation.

## CLI flags

| Flag    | Default      | Description                                                     |
| ------- | ------------ | --------------------------------------------------------------- |
| `-dump` | `./wiki.xml` | `.jsonl` file, directory of `.txt` files, or Wikipedia XML dump |
| `-addr` | `:8000`      | HTTP address to listen on                                       |
| `-max`  | `50000`      | max documents to index (0 = all)                                |

## Tests and benchmark

```bash
go test ./...
go test ./index -run xxx -bench Search10k -benchtime 5000x
```

The benchmark indexes 10,000 generated documents (80-120 tokens each, Zipf-distributed vocabulary of 20,000 words) and runs 3-term queries, top-10, single goroutine.

Measured on an AMD Ryzen 5 7520U (8 threads), Go 1.26, 5000 queries per mode:

| mode   | p50     | p95     |
| ------ | ------- | ------- |
| `bm25` | 490 µs  | 1.60 ms |
| `and`  | 15.7 µs | 291 µs  |

OR queries are slower because they score every posting of every query term. The Zipf head terms have posting lists covering a large share of the corpus, which drives the p95. These are synthetic-corpus numbers; real text will differ.

## Built with

- [snowball](https://github.com/kljensen/snowball) - Go implementation of the Snowball stemmer
