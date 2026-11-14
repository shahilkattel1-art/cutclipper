import type { CaptionSegment, TranscriptWord } from '../types/api'

const DEVANAGARI_RE = /[ऀ-ॿ]/
const SENTENCE_END_RE = /[।.!?…]$/
const MAX_GROUP_SECONDS = 8

function captionLanguage(text: string): CaptionSegment['language'] {
  const devanagari = DEVANAGARI_RE.test(text)
  const latin = /[A-Za-z]/.test(text)
  if (devanagari && latin) return 'mixed'
  if (devanagari) return 'ne'
  if (latin) return 'en'
  return null
}

function wordsForSegment(segment: CaptionSegment): TranscriptWord[] {
  const tokens = segment.text.split(/\s+/).filter(Boolean)
  if (!tokens.length) return []
  if (segment.words?.length === tokens.length) {
    return segment.words.map((word, index) => ({ ...word, text: tokens[index] ?? word.text }))
  }
  // Edited text without one-to-one word timings: spread the original span
  // across the visible tokens so density changes keep the line in place.
  const span = Math.max(0.01, segment.end - segment.start)
  return tokens.map((text, index) => ({
    start: segment.start + (span * index) / tokens.length,
    end: segment.start + (span * (index + 1)) / tokens.length,
    text,
  }))
}

/**
 * Rebuild caption lines for a words-per-caption setting.
 *
 * The backend groups words at render time; the editor normally keeps the
 * transcript's original line breaks, so changing density looked inert there.
 * This mirrors the renderer's grouping rules on the client so the editable
 * rows and the burned captions agree.
 */
export function regroupCaptionSegments(
  segments: CaptionSegment[],
  maxWords: number,
): CaptionSegment[] {
  const limit = Math.max(2, Math.floor(maxWords))
  const words = segments.flatMap(wordsForSegment)
  if (!words.length) return segments

  const groups: TranscriptWord[][] = []
  let current: TranscriptWord[] = []
  let groupStart = words[0]?.start ?? 0
  let previous: TranscriptWord | undefined

  for (const word of words) {
    if (previous) {
      const gap = word.start - previous.end
      const tooLong = word.end - groupStart > MAX_GROUP_SECONDS
      if (gap > 0.72 || SENTENCE_END_RE.test(previous.text) || tooLong || current.length >= limit) {
        if (current.length) groups.push(current)
        current = []
        groupStart = word.start
      }
    }
    if (!current.length) groupStart = word.start
    current.push(word)
    previous = word
  }
  if (current.length) groups.push(current)

  return groups.map((group) => {
    const text = group
      .map((word) => word.text)
      .join(' ')
      .trim()
    return {
      start: group[0]?.start ?? 0,
      end: group[group.length - 1]?.end ?? 0,
      text,
      words: group,
      language: captionLanguage(text),
    }
  })
}
