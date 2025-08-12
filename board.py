import chess

# Erstelle ein neues Schachbrett
board = chess.Board()

# Mache ein paar Züge
board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")

# Aktuelles FEN abrufen
current_fen = board.fen()
print("Aktuelles FEN:", current_fen)