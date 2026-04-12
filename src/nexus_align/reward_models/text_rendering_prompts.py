"""Prompt templates for three-stage text rendering evaluation.

Each model has three dimension-specific prompts (word / glyph / text).
Prompts contain a {ground_truth} placeholder that must be filled at runtime.
"""

SCORE_PER_INDICATOR = 12.5
TOTAL_INDICATORS = 8
DIMENSIONS = ("word", "glyph", "text")
TEXT_RENDERING_MAX_NEW_TOKENS = 2048

TEXT_RENDERING_PROMPTS = {
    # -------------------------------------------------------------------------
    # GLM-4.6V-Flash
    # -------------------------------------------------------------------------
    "GLM-4.6V-Flash": {
        "word": {
            "prompt": (
                'Look at this image. The text should read: "{ground_truth}"\n'
                "\n"
                "For each question, answer 1 (yes) or 0 (no). Ignore case, punctuation, and single-letter errors.\n"
                "\n"
                "- replace_word: Any word so badly corrupted (2+ wrong characters) it's a different word?\n"
                "- add_word: Any extra words not in the expected text?\n"
                "- drop_word: Any words missing from the expected text?\n"
                "\n"
                'Answer with ONLY: {"replace_word": <0 or 1>, "add_word": <0 or 1>, "drop_word": <0 or 1>}'
            ),
            "keys": ["replace_word", "add_word", "drop_word"],
        },
        "glyph": {
            "prompt": (
                "Task: Multi-dimensional classification of text rendering quality at the glyph (character) level.\n"
                "\n"
                'Ground truth text: "{ground_truth}"\n'
                "\n"
                "Examine the image and classify EACH dimension independently:\n"
                "\n"
                "1. **replace_glyph** — Does any word contain exactly one character that has been substituted with an incorrect character, while the rest of the word remains correct? → 0 (no) or 1 (yes)\n"
                "2. **add_glyph** — Does any word contain extra inserted character(s) while the original character sequence is preserved in order? (The word appears longer than expected but the original letters are all there in the right order.) → 0 (no) or 1 (yes)\n"
                "3. **drop_glyph** — Is any word missing one or more characters, with the remaining characters preserving their original relative order? (The word appears shorter than expected.) → 0 (no) or 1 (yes)\n"
                "\n"
                "Ignore capitalization and punctuation variations. Whole-word errors (entirely missing/added/replaced words) are out of scope.\n"
                "\n"
                'Output: {"replace_glyph": <0 or 1>, "add_glyph": <0 or 1>, "drop_glyph": <0 or 1>}'
            ),
            "keys": ["replace_glyph", "add_glyph", "drop_glyph"],
        },
        "text": {
            "prompt": (
                "As a print defect analyst, examine this image for text rendering shape defects.\n"
                "\n"
                'The image should display: "{ground_truth}"\n'
                "\n"
                "A shape defect (misshape) is when a character appears to be the correct letter but is visually malformed. Think of it as a printing defect: the ink spread unevenly, the font rasterizer introduced distortion, or the rendering pipeline warped the glyph. The character is still recognizable but its visual form deviates noticeably from the expected canonical shape—broken curves, uneven stroke widths, asymmetric features, warped baselines, or jagged edges that shouldn't be there.\n"
                "\n"
                "This is DIFFERENT from:\n"
                "- Wrong character (that's a substitution error)\n"
                "- Missing character (that's a deletion error)\n"
                "- Extra character (that's an insertion error)\n"
                "- Two characters stuck together (that's a merge error)\n"
                "\n"
                "Defect report — evaluate each independently:\n"
                "\n"
                "1. **misshape**: Any word in the image has character shape defects. (1 = yes, 0 = no)\n"
                "2. **no_text**: The image contains zero readable text. (1 = yes, 0 = no)\n"
                "\n"
                'Report: {"misshape": <0 or 1>, "no_text": <0 or 1>}'
            ),
            "keys": ["misshape", "no_text"],
        },
    },
    # -------------------------------------------------------------------------
    # Qwen3.5-27B
    # -------------------------------------------------------------------------
    "Qwen3.5-27B": {
        "word": {
            "prompt": (
                "Role: Senior OCR Verification Specialist.\n"
                "\n"
                "Your task is to read the text rendered in the provided image and compare it word-by-word against the reference string.\n"
                "\n"
                'Reference string: "{ground_truth}"\n'
                "\n"
                "Perform a strict word-level alignment and evaluate each dimension independently:\n"
                "\n"
                "1. **replace_word**: Check if any word has 2+ character-level mutations (substitutions, transpositions, or scrambling) making it unrecognizable as the original word. Report 1 if found, 0 otherwise.\n"
                "2. **add_word**: Check if any non-reference word appears in the rendered text that has no counterpart in the reference. Report 1 if found, 0 otherwise.\n"
                "3. **drop_word**: Check if any reference word is completely absent from the rendered text. Report 1 if found, 0 otherwise.\n"
                "\n"
                "Case and punctuation are irrelevant—only alphabetic/numeric word content matters.\n"
                "\n"
                'Output ONLY: {"replace_word": <0 or 1>, "add_word": <0 or 1>, "drop_word": <0 or 1>}'
            ),
            "keys": ["replace_word", "add_word", "drop_word"],
        },
        "glyph": {
            "prompt": (
                "You are a precise glyph-level text error classifier. Your task has two phases.\n"
                "\n"
                "**PHASE 1 — Read the image carefully.**\n"
                "List every word you can see in the image, exactly as rendered (preserve spelling). Write them in order.\n"
                "\n"
                "**PHASE 2 — Compare against ground truth.**\n"
                'Ground truth: "{ground_truth}"\n'
                "\n"
                "For each rendered word, find its matching ground-truth word (by position). Align characters one-by-one and classify using this taxonomy:\n"
                "\n"
                '• **replace_glyph** — The rendered word has the SAME length as the expected word, but EXACTLY ONE character position differs. Example: "hallo" vs "hello" (position 2: \'a\'→\'e\'). If 2+ characters differ, it is a word-level error — do NOT flag replace_glyph.\n'
                '• **add_glyph** — The rendered word is LONGER than expected. The expected word\'s characters appear as an in-order subsequence of the rendered word. Example: "appple" vs "apple" — \'a\',\'p\',\'p\',\'l\',\'e\' all appear in order with an extra \'p\'. If the order is disrupted, do NOT flag.\n'
                '• **drop_glyph** — The rendered word is SHORTER than expected. The rendered word\'s characters appear as an in-order subsequence of the expected word. Example: "aple" vs "apple" — \'a\',\'p\',\'l\',\'e\' appear in the same order. If the order is disrupted, do NOT flag.\n'
                "\n"
                "Exclusions: Ignore case differences, punctuation, and whole-word errors (entire words missing, added, or unrecognizable). When in doubt, report 0.\n"
                "\n"
                'Output ONLY: {"replace_glyph": <0 or 1>, "add_glyph": <0 or 1>, "drop_glyph": <0 or 1>}'
            ),
            "keys": ["replace_glyph", "add_glyph", "drop_glyph"],
        },
        "text": {
            "prompt": (
                "Perform a structured visual shape analysis of each word rendered in this image compared to the ground truth.\n"
                "\n"
                'Ground truth: "{ground_truth}"\n'
                "\n"
                "Analysis method:\n"
                "1. Verify text presence: Is there any text in the image? If not → no_text = 1.\n"
                "2. For each rendered word, examine the VISUAL QUALITY of its characters:\n"
                "   - Does each character have clean, well-defined strokes?\n"
                "   - Are the curves smooth and proportional?\n"
                "   - Are the character outlines consistent and free of distortion?\n"
                "   - Does the character geometry match what you'd expect for that letter?\n"
                "3. If a character has the correct identity but poor visual form (warped, broken, distorted), that word has a misshape.\n"
                "4. If any word is affected → misshape = 1. If no words are affected → misshape = 0.\n"
                "\n"
                "Not misshape: wrong letter, missing letter, extra letter, merged letters, font style differences, case changes, punctuation.\n"
                "\n"
                'Report ONLY: {"misshape": <0 or 1>, "no_text": <0 or 1>}'
            ),
            "keys": ["misshape", "no_text"],
        },
    },
    # -------------------------------------------------------------------------
    # Qwen3-VL-32B
    # -------------------------------------------------------------------------
    "Qwen3-VL-32B": {
        "word": {
            "prompt": (
                "Perform a text integrity audit on this image.\n"
                "\n"
                "The image is expected to render the following text:\n"
                '"{ground_truth}"\n'
                "\n"
                "Audit each word-level dimension INDEPENDENTLY:\n"
                "\n"
                "1. **drop_word** — COMPLETENESS: Are all words from the ground truth present in the rendered text? If any word is entirely absent, report 1; otherwise 0.\n"
                "2. **add_word** — EXTRANEOUS CONTENT: Does the rendered text contain any words not found in the ground truth? If so, report 1; otherwise 0.\n"
                "3. **replace_word** — WORD CORRUPTION: Is any word altered with 2 or more incorrect, rearranged, or garbled characters, effectively making it a different word? If so, report 1; otherwise 0.\n"
                "\n"
                "Single-character errors, case differences, and punctuation differences should be ignored.\n"
                "\n"
                'Final answer: {"replace_word": <0 or 1>, "add_word": <0 or 1>, "drop_word": <0 or 1>}'
            ),
            "keys": ["replace_word", "add_word", "drop_word"],
        },
        "glyph": {
            "prompt": (
                "Accessibility & Readability Review — Character Level\n"
                "\n"
                "You are reviewing an image for character-level text accuracy. The intended text content is:\n"
                '"{ground_truth}"\n'
                "\n"
                "Evaluate whether individual characters within words are rendered correctly across three dimensions independently:\n"
                "\n"
                "1. **drop_glyph**: Are any characters missing from words, causing them to be shorter than expected? The remaining characters must maintain their original order. (1 = yes, 0 = no)\n"
                "2. **add_glyph**: Do any words contain extra inserted characters, making them longer than expected? The original character sequence must be preserved in order. (1 = yes, 0 = no)\n"
                "3. **replace_glyph**: In any word, is exactly one character wrong (substituted), while the word length and remaining characters are correct? (1 = yes, 0 = no)\n"
                "\n"
                "Single-character issues ARE what we're looking for here. Case and punctuation differences are acceptable. Whole-word errors are out of scope.\n"
                "\n"
                'Accessibility verdict: {"replace_glyph": <0 or 1>, "add_glyph": <0 or 1>, "drop_glyph": <0 or 1>}'
            ),
            "keys": ["replace_glyph", "add_glyph", "drop_glyph"],
        },
        "text": {
            "prompt": (
                "You are inspecting a digital display for text shape quality.\n"
                "\n"
                'The display should show: "{ground_truth}"\n'
                "\n"
                "Inspection focus: CHARACTER SHAPE, not character identity. Look for characters that are the right letter but are drawn badly—warped, distorted, broken, or deformed in their visual appearance.\n"
                "\n"
                "Examples of misshape:\n"
                "- A letter 'O' that is oval when it should be round, with irregular curves\n"
                "- A letter 'T' with a crooked crossbar or uneven stem\n"
                "- Characters with broken strokes, jagged edges, or unnatural thickness variations\n"
                '- Letters that look "melted", "stretched", or "compressed" compared to their canonical form\n'
                "\n"
                "Examples that are NOT misshape:\n"
                "- A different letter altogether (that's a substitution)\n"
                "- A missing letter (that's a deletion)\n"
                "- Two letters overlapping (that's a merge)\n"
                "- Slightly different font style (acceptable variation)\n"
                "\n"
                "Inspection checklist — evaluate each independently:\n"
                "\n"
                "1. **misshape**: Is any word affected by shape defects? (0 = no word affected, 1 = one or more words affected)\n"
                "2. **no_text**: Is the display completely blank with no text at all? (0 = text present, 1 = no text)\n"
                "\n"
                'Inspection result: {"misshape": <0 or 1>, "no_text": <0 or 1>}'
            ),
            "keys": ["misshape", "no_text"],
        },
    },
}
