import gradio as gr
import cv2
import numpy as np
import os
import chess
import chess.pgn
import io
from fastapi import FastAPI
import uvicorn
from PIL import Image
from setup_game import calibrate, init_board
from record_game import record, generate_image


def print_mode(image_updated, previous_img, current_img):
    print(image_updated)
    print(previous_img)
    print(current_img)


def process_frame(frame, mode, is_calibrated, board, corners, matrix, squares, image_updated, previous_img, current_img):
    if frame is None:
        return None

    if mode == "calibrate":
        is_calibrated, corners, matrix, squares = calibrate(frame)
        if is_calibrated:
            print("calibrated correctly")
            board = init_board() # initalize board (empty board --> board with starting position)
            current_img = "media/starting_board.png" # set image of position
            image_updated = False
            gr.Success("Calibration Successful!")
        else:
            print("Not calibrated correctly")

        mode = None
    elif mode == "record":
        is_valid_move, board = record(frame, corners, matrix, board, squares)
        if is_valid_move:
            os.system("clear")
            previous_img = current_img
            current_img = generate_image(board)
            image_updated = False
    else:
        pass

    return frame, mode, is_calibrated, board, corners, matrix, squares, image_updated, previous_img, current_img


def update_img(mode, is_calibrated, image_updated, previous_img, current_img):
    if is_calibrated:
        if image_updated is False and previous_img is not current_img:
            return current_img, True
        
        return None, True
    else:
        return "media/empty_board.png", True


def update_fen(txt_fen, board):
    if txt_fen != board.fen():
        print("-----------------------")
        print(board.fen())
        print("-----------------------")

        return board.fen()
    else:
        return txt_fen


def update_pgn(board, pgn):
    new_game = chess.Board(None)

    game = chess.pgn.Game() # overall game
    node = game             # main variatioin of the game

    for move in board.move_stack:
        node = node.add_main_variation(move)

    if pgn != str(game):
        return str(game)
    else:
        return pgn

def update_ui(mode, is_calibrated, image_updated, previous_img, current_img, txt_fen, txt_pgn, board):
    img, image_updated = update_img(mode, is_calibrated, image_updated, previous_img, current_img)
    if img is None:
        img = current_img
    txt_fen = update_fen(txt_fen, board)
    txt_pgn = update_pgn(board, pgn)

    return img, image_updated, txt_fen, txt_pgn, pgn


with gr.Blocks() as demo:
    mode = gr.State(None)
    is_calibrated = gr.State(False)
    is_valid_move = gr.State(False)
    board = gr.State(chess.Board(None))
    corners = gr.State(None)
    matrix = gr.State(None)
    squares = gr.State(None)
    image_updated = gr.State(True)
    previous_img = gr.State(None)
    current_img = gr.State(None)
    pgn = gr.State(None)


    with gr.Sidebar():
        gr.Markdown("## REALTIME CHESS TRACKER")
        gr.Markdown("A neural network assisted system to track your chess games")
        gr.Markdown("### Game Recording")
        btn_calibrate = gr.Button("Calibrate")
        btn_record = gr.Button("Record")
        btn_save = gr.Button("Save")
        btn_reset = gr.Button("Reset", variant="stop")
        # btn_print = gr.Button("Print", variant="primary")

        gr.Markdown("### Game Settings")
        btn_settings = gr.Button("Game Settings")
        txt_fen = gr.Textbox(board.value.fen(), label="FEN", show_copy_button=True, interactive=False)
        txt_pgn = gr.Textbox("", label="PGN", show_copy_button=True, interactive=False)

    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(sources="webcam", streaming=True, mirror_webcam=False)
            processed_output = gr.Image()
        with gr.Column():
            pos_img = gr.Image("media/empty_board.png")

            timer = gr.Timer(value=1.0)
            timer.tick(
                fn=update_ui,
                inputs=[mode, is_calibrated, image_updated, previous_img, current_img, txt_fen, txt_pgn, board],
                outputs=[pos_img, image_updated, txt_fen, txt_pgn, pgn]
            )

    # Frame-Verarbeitung
    webcam_input.stream(
        fn=process_frame,
        inputs=[webcam_input, mode, is_calibrated, board, corners, matrix, squares, image_updated, previous_img, current_img],
        outputs=[processed_output, mode, is_calibrated, board, corners, matrix, squares, image_updated, previous_img, current_img]
    )

    # Update session state on button press
    btn_calibrate.click(lambda: "calibrate", outputs=[mode])
    btn_record.click(lambda: "record", outputs=[mode])
    btn_reset.click(lambda: None, outputs=[mode])
    # btn_print.click(print_mode, inputs=[image_updated, previous_img, current_img])


app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    uvicorn.run(app, host="0.0.0.0", port=7860)