package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/whysooharsh/text-search-engine/api"
	"github.com/whysooharsh/text-search-engine/index"
	"github.com/whysooharsh/text-search-engine/loader"
)

func printMemStats(label string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("  %-28s | RAM: %d MB | Sys: %d MB | GC: %d\n",
		label,
		m.Alloc/1024/1024,
		m.Sys/1024/1024,
		m.NumGC,
	)
}

func main() {
	dumpPath := flag.String("dump", "./wiki.xml", "path to a .jsonl file, a directory of .txt files, or a Wikipedia XML dump")
	addr := flag.String("addr", ":8000", "HTTP listen address")
	maxDocs := flag.Int("max", 50000, "max documents to index (0 = all)")
	flag.Parse()

	fmt.Println("==========================================")
	fmt.Println("      Text Search Engine — Starting")
	fmt.Println("==========================================")
	fmt.Printf("  CPUs       : %d\n", runtime.NumCPU())
	fmt.Printf("  Source     : %s\n", *dumpPath)
	fmt.Printf("  Max docs   : %d\n", *maxDocs)
	fmt.Printf("  Address    : %s\n", *addr)
	fmt.Println("------------------------------------------")

	printMemStats("startup")

	idx := index.New()
	count := 0
	start := time.Now()

	fmt.Println("------------------------------------------")
	fmt.Println("  Loading + indexing...")
	fmt.Println("------------------------------------------")

	err := loader.Load(*dumpPath, func(doc index.Document) {
		idx.Add(doc)
		count++
		if count%10000 == 0 {
			elapsed := time.Since(start)
			printMemStats(fmt.Sprintf("%d docs", count))
			fmt.Printf("  %d docs | %s elapsed | %.0f docs/sec\n",
				count, elapsed, float64(count)/elapsed.Seconds())
		}
	}, *maxDocs)

	if err != nil {
		log.Fatalf("load error: %v", err)
	}

	fmt.Println("------------------------------------------")
	fmt.Println("  Finalizing index...")
	finalizeStart := time.Now()
	idx.Finalize()
	fmt.Printf("  Finalized in %s\n", time.Since(finalizeStart))
	fmt.Println("------------------------------------------")

	totalTime := time.Since(start)
	printMemStats("ready")

	fmt.Println("==========================================")
	fmt.Printf("  Documents indexed : %d\n", count)
	fmt.Printf("  Total time        : %s\n", totalTime)
	fmt.Printf("  Avg throughput    : %.0f docs/sec\n", float64(count)/totalTime.Seconds())
	fmt.Println("==========================================")
	fmt.Printf("  Server ready on %s\n", *addr)
	fmt.Println("==========================================")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := api.New(idx).ListenAndServe(ctx, *addr); err != nil {
		log.Fatalf("server error: %v", err)
	}
	log.Println("shut down cleanly")
}
