package index

import (
	"strings"
	"unicode"

	"github.com/kljensen/snowball/english"
)

var stopWords = map[string]struct{}{
	"a": {}, "about": {}, "above": {}, "after": {}, "again": {}, "against": {},
	"all": {}, "am": {}, "an": {}, "and": {}, "any": {}, "are": {}, "as": {},
	"at": {}, "be": {}, "because": {}, "been": {}, "before": {}, "being": {},
	"below": {}, "between": {}, "both": {}, "but": {}, "by": {},
	"for": {}, "from": {}, "further": {}, "had": {}, "has": {}, "have": {},
	"he": {}, "her": {}, "here": {}, "hers": {}, "him": {}, "his": {},
	"i": {}, "if": {}, "in": {}, "into": {}, "is": {}, "it": {}, "its": {},
	"no": {}, "not": {}, "of": {}, "on": {}, "or": {}, "other": {}, "our": {}, "out": {},
	"that": {}, "the": {}, "their": {}, "them": {}, "there": {}, "they": {},
	"this": {}, "those": {}, "through": {}, "to": {}, "too": {},
	"was": {}, "we": {}, "were": {}, "what": {}, "when": {}, "where": {},
	"which": {}, "who": {}, "will": {}, "with": {},
	"you": {}, "your": {}, "yours": {},
}

func normalize(tok string) (string, bool) {
	tok = strings.ToLower(tok)
	if _, stop := stopWords[tok]; stop {
		return "", false
	}
	tok = english.Stem(tok, false)
	return tok, tok != ""
}

func Analyze(text string) []string {
	tokens := tokenize(text)
	var terms []string
	for _, tok := range tokens {
		if term, ok := normalize(tok); ok {
			terms = append(terms, term)
		}
	}
	return terms
}

func isSep(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }

func tokenize(text string) []string {
	return strings.FieldsFunc(text, isSep)
}

type Token struct {
	Term       string
	Start, End int
}

func AnalyzeOffsets(text string) []Token {
	var out []Token
	start := -1
	emit := func(end int) {
		if start < 0 {
			return
		}
		if term, ok := normalize(text[start:end]); ok {
			out = append(out, Token{term, start, end})
		}
		start = -1
	}
	for i, r := range text {
		if isSep(r) {
			emit(i)
		} else if start < 0 {
			start = i
		}
	}
	emit(len(text))
	return out
}
