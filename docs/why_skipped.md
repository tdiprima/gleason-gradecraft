### Why labels 4 and 5 are skipped in prostate classification

**They're not skipping Gleason grades 4 and 5 — they're skipping label indices 4 and 5.**

Here's what's happening:

The original dataset has 6 classes (labels 0-5), but they're consolidating it into a 4-way classification:

| Label Index | Original Meaning | Used in Training? |
|-------------|------------------|-------------------|
| 0 | Benign | ✓ → "benign" |
| 1 | Gleason 3 | ✓ → "grade3" |
| 2 | Gleason 4 | ✓ → "grade4" |
| 3 | Gleason 5-Single Cells | ✓ → "grade5" |
| 4 | Gleason 5-Secretions | ✗ skipped |
| 5 | Gleason 5 | ✗ skipped |

**The reason:** Labels 3, 4, and 5 are all subtypes of Gleason grade 5. Rather than using all three Gleason 5 variants, they're only using label 3 ("Gleason 5-Single Cells") to represent grade 5 in their 4-way classifier.

This could be because:

1. They wanted a cleaner 4-class problem (benign + the three main Gleason grades)
2. The other Gleason 5 subtypes may have had fewer samples or were less distinct
3. They're treating "Single Cells" as the canonical Gleason 5 representation

<br>
