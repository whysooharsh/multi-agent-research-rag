package loader

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/whysooharsh/text-search-engine/index"
)

func loadAll(t *testing.T, path string, max int) ([]index.Document, error) {
	t.Helper()
	var docs []index.Document
	err := Load(path, func(d index.Document) { docs = append(docs, d) }, max)
	return docs, err
}

func TestLoadJSONL(t *testing.T) {
	p := filepath.Join(t.TempDir(), "d.jsonl")
	data := `{"id":42,"title":"A","url":"http://a","text":"héllo"}` + "\n\n" +
		`{"id":7,"title":"B","url":"http://b","text":"x"}`
	os.WriteFile(p, []byte(data), 0o644)

	docs, err := loadAll(t, p, 0)
	if err != nil || len(docs) != 2 {
		t.Fatalf("docs=%v err=%v", docs, err)
	}
	if docs[0].ID != 42 || docs[0].URL != "http://a" || docs[0].Text != "héllo" || docs[1].ID != 7 {
		t.Errorf("unexpected docs: %+v", docs)
	}
	if docs, _ = loadAll(t, p, 1); len(docs) != 1 {
		t.Errorf("max=1 gave %d docs", len(docs))
	}
}

func TestLoadJSONLErrors(t *testing.T) {
	for name, data := range map[string]string{
		"missing id": `{"title":"A"}`,
		"bad json":   `{"id":1`,
	} {
		p := filepath.Join(t.TempDir(), "d.jsonl")
		os.WriteFile(p, []byte(data), 0o644)
		if _, err := loadAll(t, p, 0); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestLoadTxtDir(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("bee"), 0o644)
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("ay"), 0o644)
	os.WriteFile(filepath.Join(dir, "skip.md"), []byte("no"), 0o644)

	docs, err := loadAll(t, dir, 0)
	if err != nil || len(docs) != 2 {
		t.Fatalf("docs=%v err=%v", docs, err)
	}
	if docs[0].Title != "a" || docs[0].ID != 1 || docs[1].Title != "b" || docs[1].Text != "bee" {
		t.Errorf("unexpected docs: %+v", docs)
	}
}
