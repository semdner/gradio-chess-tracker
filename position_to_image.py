import chess
import chess.svg
import cv2
import cairosvg
import numpy as np

board = chess.Board(None)
svg_img = chess.svg.board(board, size=1000,    colors={
        "square light": "#f0d9b5",
        "square dark": "#b58863"
    })

png_data = cairosvg.svg2png(bytestring=svg_img.encode('utf-8'))

image_array = np.frombuffer(png_data, dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

cv2.imwrite("media/empty_board.png", image)