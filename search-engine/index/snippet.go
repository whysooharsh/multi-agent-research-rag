package index

import "unicode/utf8"

const snippetRunes = 300

func Snippet(text string, queryTerms []string) string {
	want := make(map[string]struct{}, len(queryTerms))
	for _, t := range queryTerms {
		want[t] = struct{}{}
	}

	ms, me := 0, 0
	for _, tok := range AnalyzeOffsets(text) {
		if _, ok := want[tok.Term]; ok {
			ms, me = tok.Start, tok.End
			break
		}
	}

	room := snippetRunes - utf8.RuneCountInString(text[ms:me])
	if room < 0 {
		room = 0
	}
	start := ms
	for n := room / 2; n > 0 && start > 0; n-- {
		_, size := utf8.DecodeLastRuneInString(text[:start])
		start -= size
	}
	left := utf8.RuneCountInString(text[start:ms])
	end := me
	for n := room - left; n > 0 && end < len(text); n-- {
		_, size := utf8.DecodeRuneInString(text[end:])
		end += size
	}
	for n := room - left - utf8.RuneCountInString(text[me:end]); n > 0 && start > 0; n-- {
		_, size := utf8.DecodeLastRuneInString(text[:start])
		start -= size
	}

	s := text[start:end]
	if start > 0 {
		s = "…" + s
	}
	if end < len(text) {
		s += "…"
	}
	return s
}
