package loader

import (
	"bufio"
	"bytes"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/whysooharsh/text-search-engine/index"
)

type xmlPage struct {
	Title string `xml:"title"`
	ID    uint32 `xml:"id"`
	Text  string `xml:"revision>text"`
}

func Load(path string, fn func(index.Document), maxDocs int) error {
	st, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("cant open %q: %w", path, err)
	}
	if st.IsDir() {
		return loadTxtDir(path, fn, maxDocs)
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("cant open file %q: %w", path, err)
	}
	defer f.Close()

	switch strings.ToLower(filepath.Ext(path)) {
	case ".jsonl", ".ndjson":
		return loadJSONL(f, fn, maxDocs)
	}
	return loadXML(f, fn, maxDocs)
}

type jsonlDoc struct {
	ID    *uint32 `json:"id"`
	Title string  `json:"title"`
	URL   string  `json:"url"`
	Text  string  `json:"text"`
}

func loadJSONL(r io.Reader, fn func(index.Document), maxDocs int) error {
	br := bufio.NewReader(r)
	count := 0
	for line := 1; ; line++ {
		b, err := br.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return fmt.Errorf("line %d: %w", line, err)
		}
		if b = bytes.TrimSpace(b); len(b) > 0 {
			var d jsonlDoc
			if err := json.Unmarshal(b, &d); err != nil {
				return fmt.Errorf("line %d: %w", line, err)
			}
			if d.ID == nil {
				return fmt.Errorf("line %d: missing id", line)
			}
			fn(index.Document{ID: *d.ID, Title: d.Title, URL: d.URL, Text: d.Text})
			if count++; maxDocs > 0 && count >= maxDocs {
				return nil
			}
		}
		if err == io.EOF {
			return nil
		}
	}
}

func loadTxtDir(dir string, fn func(index.Document), maxDocs int) error {
	paths, err := filepath.Glob(filepath.Join(dir, "*.txt"))
	if err != nil {
		return err
	}
	sort.Strings(paths)
	for i, p := range paths {
		if maxDocs > 0 && i >= maxDocs {
			break
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		fn(index.Document{
			ID:    uint32(i + 1),
			Title: strings.TrimSuffix(filepath.Base(p), filepath.Ext(p)),
			Text:  string(data),
		})
	}
	return nil
}

func loadXML(f io.Reader, fn func(index.Document), maxDocs int) error {
	dec := xml.NewDecoder(f)
	count := 0

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("xml parse error: %w", err)
		}

		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "page" {
			continue
		}

		var page xmlPage
		if err := dec.DecodeElement(&page, &se); err != nil {
			continue
		}

		if page.Text == "" || page.Title == "" {
			continue
		}

		fn(index.Document{
			ID:    page.ID,
			Title: page.Title,
			Text:  page.Text,
		})

		count++
		if count%10000 == 0 {
			log.Printf("processed %d documents...", count)
		}
		if maxDocs > 0 && count >= maxDocs {
			log.Printf("reached limit of %d documents, stopping", maxDocs)
			break
		}
	}

	return nil
}
