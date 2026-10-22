"""Conservative rejection rules, not a calibrated accuracy guarantee."""
import numpy as np

AMBIGUOUS = [set('0Oo'), set('1Il'), set('xX×'), set('cC'), set('sS'), set('vV'), set('wW'), set('zZ')]


def assess(row, ranking, labels, mode, glyph):
    best, runner = int(ranking[0]), int(ranking[1])
    probability = float(row[best])
    margin = probability - float(row[runner])
    reasons = []
    if probability < .75 or margin < .20:
        reasons.append('The model cannot clearly distinguish this character. Choose a suggestion or redraw it.')
    if labels[best] == '=':
        reasons.append('The equals sign is experimental. Please confirm it.')
    if mode == 'mixed' and any(labels[best] in group for group in AMBIGUOUS):
        reasons.append('This shape can represent different characters. Select a mode or confirm the character.')
    ink = np.asarray(glyph)[...,0] > .25
    ys, xs = np.where(ink)
    if len(xs) and ink[ys.min():ys.max()+1,xs.min():xs.max()+1].mean() > .80:
        # Thin straight strokes can legitimately be 1, I, l, / or -.
        if min(xs.max()-xs.min()+1,ys.max()-ys.min()+1) > 7:
            reasons.append('This looks like a filled shape rather than a clear character. Please redraw it.')
    return reasons
