//! Évaluation statique : matériel, tables pièce-case interpolées milieu / finale,
//! structure de pions, tours sur colonne ouverte, abri du roi, mobilité.
//!
//! Portage fidèle de `chessio/moteur/evaluation.py` (v0.2) : mêmes tables, mêmes
//! termes, pour pouvoir comparer les deux moteurs position par position.
//! Le score est en centipions, du point de vue du camp au trait.

use shakmaty::attacks::{bishop_attacks, knight_attacks, queen_attacks, rook_attacks};
use shakmaty::{Bitboard, Board, Color, File, Position, Rank, Role, Square};

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0]; // P N B R Q K
const BISHOP_PAIR: i32 = 30;

#[rustfmt::skip]
const PAWN_MG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];
#[rustfmt::skip]
const PAWN_EG: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    15, 15, 15, 15, 15, 15, 15, 15,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
];
#[rustfmt::skip]
const KNIGHT: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];
#[rustfmt::skip]
const BISHOP: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];
#[rustfmt::skip]
const ROOK: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];
#[rustfmt::skip]
const QUEEN: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];
#[rustfmt::skip]
const KING_MG: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];
#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

const MAX_PHASE: i32 = 24;
const PASSED_BONUS: [(i32, i32); 8] = [
    (0, 0),
    (5, 10),
    (10, 15),
    (15, 25),
    (25, 45),
    (40, 75),
    (60, 120),
    (0, 0),
];
const DOUBLED_PAWN: (i32, i32) = (-10, -20);
const ISOLATED_PAWN: (i32, i32) = (-12, -15);
const ROOK_OPEN_FILE: (i32, i32) = (25, 10);
const ROOK_SEMI_OPEN_FILE: (i32, i32) = (12, 6);
const KING_SHIELD_PAWN: i32 = 12;
// Mobilité (milieu, finale, mobilité « normale ») pour N, B, R, Q.
const MOBILITY: [(i32, i32, i32); 4] = [(4, 4, 4), (5, 5, 7), (2, 4, 7), (1, 2, 14)];

fn tables(role: Role) -> (&'static [i32; 64], &'static [i32; 64]) {
    match role {
        Role::Pawn => (&PAWN_MG, &PAWN_EG),
        Role::Knight => (&KNIGHT, &KNIGHT),
        Role::Bishop => (&BISHOP, &BISHOP),
        Role::Rook => (&ROOK, &ROOK),
        Role::Queen => (&QUEEN, &QUEEN),
        Role::King => (&KING_MG, &KING_EG),
    }
}

/// Indice dans une table écrite « 8e rangée en haut », vue du camp `color`.
#[inline]
fn table_index(square: Square, color: Color) -> usize {
    let file = square.file().to_usize();
    let rank = square.rank().to_usize();
    let row = if color.is_white() { 7 - rank } else { rank };
    row * 8 + file
}

fn file_mask(file: usize) -> Bitboard {
    Bitboard::from_file(File::new(file as u32))
}

fn adjacent_files(file: usize) -> Bitboard {
    let mut mask = Bitboard::EMPTY;
    if file > 0 {
        mask |= file_mask(file - 1);
    }
    if file < 7 {
        mask |= file_mask(file + 1);
    }
    mask
}

/// Cases devant un pion (sa colonne et les voisines) où un pion adverse l'arrêterait.
fn passed_mask(square: Square, color: Color) -> Bitboard {
    let file = square.file().to_usize();
    let rank = square.rank().to_usize();
    let files = file_mask(file) | adjacent_files(file);
    let mut mask = Bitboard::EMPTY;
    let ranks: Vec<usize> = if color.is_white() {
        (rank + 1..8).collect()
    } else {
        (0..rank).collect()
    };
    for r in ranks {
        mask |= Bitboard::from_rank(Rank::new(r as u32)) & files;
    }
    mask
}

fn shield_mask(king: Square, color: Color) -> Bitboard {
    let file = king.file().to_usize();
    let rank = king.rank().to_usize() as i32;
    let front = if color.is_white() { rank + 1 } else { rank - 1 };
    if !(0..8).contains(&front) {
        return Bitboard::EMPTY;
    }
    Bitboard::from_rank(Rank::new(front as u32)) & (file_mask(file) | adjacent_files(file))
}

pub struct Tables {
    mg: [[[i32; 64]; 6]; 2], // [couleur][pièce][case] : matériel + table
    eg: [[[i32; 64]; 6]; 2],
    passed: [[Bitboard; 64]; 2],
    shield: [[Bitboard; 64]; 2],
    files: [Bitboard; 8],
    adjacent: [Bitboard; 8],
}

impl Tables {
    pub fn new() -> Tables {
        let mut t = Tables {
            mg: [[[0; 64]; 6]; 2],
            eg: [[[0; 64]; 6]; 2],
            passed: [[Bitboard::EMPTY; 64]; 2],
            shield: [[Bitboard::EMPTY; 64]; 2],
            files: [Bitboard::EMPTY; 8],
            adjacent: [Bitboard::EMPTY; 8],
        };
        for color in Color::ALL {
            let c = color_index(color);
            for (r, role) in Role::ALL.iter().enumerate() {
                let (mg, eg) = tables(*role);
                for square in Square::ALL {
                    let i = table_index(square, color);
                    t.mg[c][r][square.to_usize()] = PIECE_VALUES[r] + mg[i];
                    t.eg[c][r][square.to_usize()] = PIECE_VALUES[r] + eg[i];
                }
            }
            for square in Square::ALL {
                t.passed[c][square.to_usize()] = passed_mask(square, color);
                t.shield[c][square.to_usize()] = shield_mask(square, color);
            }
        }
        for file in 0..8 {
            t.files[file] = file_mask(file);
            t.adjacent[file] = adjacent_files(file);
        }
        t
    }
}

impl Default for Tables {
    fn default() -> Self {
        Tables::new()
    }
}

#[inline]
pub fn color_index(color: Color) -> usize {
    if color.is_white() { 0 } else { 1 }
}

#[inline]
pub fn role_index(role: Role) -> usize {
    match role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

pub fn phase(board: &Board) -> i32 {
    let total = (board.knights() | board.bishops()).count() as i32
        + 2 * board.rooks().count() as i32
        + 4 * board.queens().count() as i32;
    total.min(MAX_PHASE)
}

fn side(board: &Board, t: &Tables, color: Color) -> (i32, i32) {
    let c = color_index(color);
    let own = board.by_color(color);
    let occupied = board.occupied();
    let mut mg = 0;
    let mut eg = 0;
    for (r, role) in Role::ALL.iter().enumerate() {
        for square in board.by_role(*role) & own {
            mg += t.mg[c][r][square.to_usize()];
            eg += t.eg[c][r][square.to_usize()];
        }
    }
    if (board.bishops() & own).count() >= 2 {
        mg += BISHOP_PAIR;
        eg += BISHOP_PAIR;
    }

    // Structure de pions.
    let own_pawns = board.pawns() & own;
    let their_pawns = board.pawns() & board.by_color(color.other());
    for square in own_pawns {
        let file = square.file().to_usize();
        if (their_pawns & t.passed[c][square.to_usize()]).is_empty() {
            let rank = square.rank().to_usize();
            let advanced = if color.is_white() { rank } else { 7 - rank };
            mg += PASSED_BONUS[advanced].0;
            eg += PASSED_BONUS[advanced].1;
        }
        if (own_pawns & t.adjacent[file]).is_empty() {
            mg += ISOLATED_PAWN.0;
            eg += ISOLATED_PAWN.1;
        }
    }
    for file in 0..8 {
        let count = (own_pawns & t.files[file]).count() as i32;
        if count > 1 {
            mg += DOUBLED_PAWN.0 * (count - 1);
            eg += DOUBLED_PAWN.1 * (count - 1);
        }
    }

    // Tours sur colonne ouverte ou semi-ouverte.
    for square in board.rooks() & own {
        let mask = t.files[square.file().to_usize()];
        if (own_pawns & mask).is_empty() {
            let bonus = if (their_pawns & mask).is_empty() {
                ROOK_OPEN_FILE
            } else {
                ROOK_SEMI_OPEN_FILE
            };
            mg += bonus.0;
            eg += bonus.1;
        }
    }

    // Abri du roi (milieu de partie).
    if let Some(king) = board.king_of(color) {
        mg += KING_SHIELD_PAWN * (own_pawns & t.shield[c][king.to_usize()]).count() as i32;
    }

    // Mobilité : cases attaquées non occupées par ses propres pièces.
    let not_own = !own;
    for square in board.knights() & own {
        let (wm, we, typical) = MOBILITY[0];
        let moves = (knight_attacks(square) & not_own).count() as i32 - typical;
        mg += wm * moves;
        eg += we * moves;
    }
    for square in board.bishops() & own {
        let (wm, we, typical) = MOBILITY[1];
        let moves = (bishop_attacks(square, occupied) & not_own).count() as i32 - typical;
        mg += wm * moves;
        eg += we * moves;
    }
    for square in board.rooks() & own {
        let (wm, we, typical) = MOBILITY[2];
        let moves = (rook_attacks(square, occupied) & not_own).count() as i32 - typical;
        mg += wm * moves;
        eg += we * moves;
    }
    for square in board.queens() & own {
        let (wm, we, typical) = MOBILITY[3];
        let moves = (queen_attacks(square, occupied) & not_own).count() as i32 - typical;
        mg += wm * moves;
        eg += we * moves;
    }
    (mg, eg)
}

/// Score statique en centipions du point de vue du camp au trait.
pub fn evaluate<P: Position>(pos: &P, t: &Tables) -> i32 {
    let board = pos.board();
    let (white_mg, white_eg) = side(board, t, Color::White);
    let (black_mg, black_eg) = side(board, t, Color::Black);
    let mg = white_mg - black_mg;
    let eg = white_eg - black_eg;
    let game_phase = phase(board);
    // Division entière de Rust : troncature vers zéro, symétrique entre les camps.
    let score = (mg * game_phase + eg * (MAX_PHASE - game_phase)) / MAX_PHASE;
    if pos.turn().is_white() { score } else { -score }
}
