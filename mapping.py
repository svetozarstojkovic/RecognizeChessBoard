from PIL import Image


def connect_number_and_name(number):
    if number == 0:
        return 'Pawn black'
    elif number == 1:
        return 'Pawn white'
    elif number == 2:
        return 'Empty white field'
    elif number == 3:
        return 'Empty black field'
    elif number == 4:
        return 'King white'
    elif number == 5:
        return 'King black'
    elif number == 6:
        return 'Queen white'
    elif number == 7:
        return 'Queen black'
    elif number == 8:
        return 'Rook white'
    elif number == 9:
        return 'Rook black'
    elif number == 10:
        return 'Knight white'
    elif number == 11:
        return 'Knight black'
    elif number == 12:
        return 'Bishop white'
    elif number == 13:
        return 'Bishop black'
    else:
     print 'Neither one'
     return 'Neither one'

def connect_number_and_model(number):
    if number == 0:
        return Image.open('model_figures/pawn_b.png')
    elif number == 1:
        return Image.open('model_figures/pawn_w.png')
    elif number == 2:
        return Image.new('RGBA', (60, 60), (255,255,255,255))
    elif number == 3:
        return Image.new('RGBA', (60, 60), (100,100,100,255))
    elif number == 4:
        return Image.open('model_figures/king_w.png')
    elif number == 5:
        return Image.open('model_figures/king_b.png')
    elif number == 6:
        return Image.open('model_figures/queen_w.png')
    elif number == 7:
        return Image.open('model_figures/queen_b.png')
    elif number == 8:
        return Image.open('model_figures/rook_w.png')
    elif number == 9:
        return Image.open('model_figures/rook_b.png')
    elif number == 10:
        return Image.open('model_figures/knight_w.png')
    elif number == 11:
        return Image.open('model_figures/knight_b.png')
    elif number == 12:
        return Image.open('model_figures/bishop_w.png')
    elif number == 13:
        return Image.open('model_figures/bishop_b.png')
    else:
     print 'Neither one'
     return 'Neither one'

def connect_name_and_number(name):
    if name == 'pawn_b':
        return 0
    elif name == 'pawn_w':
        return 1
    elif name == 'field_w':
        return 2
    elif name == 'field_b':
        return 3
    elif name == 'king_w':
        return 4
    elif name == 'king_b':
        return 5
    elif name == 'queen_w':
        return 6
    elif name == 'queen_b':
        return 7
    elif name == 'rook_w':
        return 8
    elif name == 'rook_b':
        return 9
    elif name == 'knight_w':
        return 10
    elif name == 'knight_b':
        return 11
    elif name == 'bishop_w':
        return 12
    elif name == 'bishop_b':
        return 13
    else:
     print 'neither one'
     return 2

def connect_number_and_code(number):
    if number == 0:
        return 'pawn_b'
    elif number == 1:
        return 'pawn_w'
    elif number == 2:
        return 'field_w'
    elif number == 3:
        return 'field_b'
    elif number == 4:
        return 'king_w'
    elif number == 5:
        return 'king_b'
    elif number == 6:
        return 'queen_w'
    elif number == 7:
        return 'queen_b'
    elif number == 8:
        return 'rook_w'
    elif number == 9:
        return 'rook_b'
    elif number == 10:
        return 'knight_w'
    elif number == 11:
        return 'knight_b'
    elif number == 12:
        return 'bishop_w'
    elif number == 13:
        return 'bishop_b'
    else:
     print 'Neither one'
     return 'Neither one'