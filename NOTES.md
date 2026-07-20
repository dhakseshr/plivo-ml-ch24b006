# NOTES

The model uses 50 causal features extracted strictly before each pause: F0 slope and final/mean ratio (falling pitch signals statement completion), energy slope and decay into the pause, spectral centroid trend, 13 MFCC means and 13 MFCC delta means from the last 0.5 s of speech (capturing spectral dynamics), last voiced region duration (syllable lengthening), voiced fraction, ZCR, pause index, pause_start time, and inter-pause interval. A GradientBoostingClassifier (300 trees, depth 4) with isotonic probability calibration is trained jointly on English and Hindi data.

The model still fails on pauses where the speaker uses a flat or slightly rising pitch for declarative statements (common in certain Hindi dialects and in list-enumeration contexts), causing hold pauses to look like EOT. It also struggles on very short turns (< 1 s of speech) where there is insufficient context for reliable F0 estimation.

With one more day I would add: (1) a learned speaker normalization step to handle pitch range variability across speakers; (2) a recurrent model (LSTM or GRU) over the F0 and energy contour sequence to capture longer-range intonation patterns; (3) deliberate negative mining — training on the grader's hardest-to-classify pauses — using the confusion matrix from cross-validation.
