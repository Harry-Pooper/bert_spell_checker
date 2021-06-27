from correcting import Corrector
from correcting import PossibleValues

print("[INFO] " + 'reading and init')
# RUS
possibleValues = PossibleValues()
corrector = Corrector(possibleValues)
# EN
# possibleValues = PossibleValues(False)
# corrector = Corrector(possibleValues, False)
print("[INFO] " + 'read and init completed')

entryText = "Он прешёл сюда но ушол так же быстро как и пришел"
print("[INFO] " + entryText)
entryText = corrector.normalize(entryText)
print("[INFO] Этап нормализации текста:")
print("[INFO] " + entryText)
entryText = corrector.correctGrammar(entryText)
print("[INFO] " + "Этап исправления грамматических и орфографических ошибок")
print("[INFO] " + entryText)
entryText = corrector.correctPunctuation(entryText)
print("[INFO] " + "Этап исправления пунктуационных ошибок")
print("[INFO] " + entryText)
