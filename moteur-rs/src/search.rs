//! Recherche : négamax alpha-bêta, portage de `chessio/moteur/search.py` (v0.2).
//!
//! Approfondissement itératif avec fenêtres d'aspiration, recherche à fenêtre
//! nulle (PVS), table de transposition indexée par hachage de Zobrist, coup nul,
//! réductions des coups tardifs (LMR), élagages statique et de frontière,
//! quiescence sur prises et promotions, ordre des coups (coup de la table,
//! MVV-LVA, coups tueurs, historique), MultiPV par exclusion à la racine.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, EnPassantMode, Move, MoveList, Position, Role};

use crate::eval::{PIECE_VALUES, Tables, evaluate, role_index};

pub const INF: i32 = 1_000_000;
pub const MATE: i32 = 100_000;
pub const MATE_BOUND: i32 = MATE - 1_000;
pub const MAX_PLY: usize = 128;
const ASPIRATION: i32 = 40;
const REVERSE_FUTILITY: i32 = 120;
const FUTILITY_MARGIN: i32 = 150;
const DELTA_MARGIN: i32 = 200;
const ORDER_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 2000];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    mv: Option<Move>,
    score: i32,
    depth: i16,
    bound: Bound,
}

const EMPTY_ENTRY: Entry = Entry {
    key: 0,
    mv: None,
    score: 0,
    depth: -1,
    bound: Bound::Upper,
};

#[derive(Default, Clone)]
pub struct Limits {
    pub depth: Option<u32>,
    pub movetime: Option<Duration>,
    pub nodes: Option<u64>,
    pub wtime: Option<Duration>,
    pub btime: Option<Duration>,
    pub winc: Duration,
    pub binc: Duration,
    pub movestogo: Option<u32>,
    pub infinite: bool,
}

impl Limits {
    /// (temps souple, temps dur) : pas de nouvelle itération après le premier,
    /// interruption au second.
    fn budget(&self, white: bool) -> (Option<Duration>, Option<Duration>) {
        if self.infinite {
            return (None, None);
        }
        if let Some(movetime) = self.movetime {
            return (Some(movetime.mul_f64(0.6)), Some(movetime));
        }
        let remaining = if white { self.wtime } else { self.btime };
        let Some(remaining) = remaining else {
            return (None, None);
        };
        let increment = if white { self.winc } else { self.binc };
        let soft = remaining / self.movestogo.unwrap_or(30).max(1) + increment.mul_f64(0.75);
        let hard = (remaining.mul_f64(0.4).min(soft * 3))
            .saturating_sub(Duration::from_millis(50))
            .max(Duration::from_millis(10));
        (Some(soft.min(hard)), Some(hard))
    }
}

pub struct Line {
    pub score: i32,
    pub moves: Vec<Move>,
}

pub struct Info {
    pub depth: u32,
    pub seldepth: u32,
    pub lines: Vec<Line>,
    pub nodes: u64,
    pub elapsed: Duration,
    pub hashfull: u32,
}

impl Info {
    pub fn best(&self) -> Option<Move> {
        self.lines.first().and_then(|line| line.moves.first().copied())
    }
}

/// Coups (et non demi-coups) avant le mat : positif si l'on mate, négatif sinon.
pub fn mate_in(score: i32) -> Option<i32> {
    if score.abs() < MATE_BOUND {
        return None;
    }
    let moves = (MATE - score.abs() + 1) / 2;
    Some(if score > 0 { moves } else { -moves })
}

#[inline]
pub fn hash(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

pub struct Searcher {
    tables: Tables,
    tt: Vec<Entry>,
    killers: Vec<[Option<Move>; 2]>,
    history: Box<[[[i32; 64]; 64]; 2]>,
    pv: Vec<Vec<Move>>,
    hashes: Vec<u64>, // positions depuis le début de la partie, pour les répétitions
    lmr: Box<[[i32; 64]; 64]>,
    nodes: u64,
    seldepth: u32,
    stop: Arc<AtomicBool>,
    deadline: Option<Instant>,
    node_limit: Option<u64>,
    can_abort: bool,
    aborted: bool,
}

impl Searcher {
    pub fn new(hash_mb: usize, stop: Arc<AtomicBool>) -> Searcher {
        let mut lmr = Box::new([[0; 64]; 64]);
        for (depth, row) in lmr.iter_mut().enumerate().skip(1) {
            for (index, value) in row.iter_mut().enumerate().skip(1) {
                *value = (0.75 + (depth as f64).ln() * (index as f64).ln() / 2.25) as i32;
            }
        }
        let mut searcher = Searcher {
            tables: Tables::new(),
            tt: Vec::new(),
            killers: vec![[None, None]; MAX_PLY + 2],
            history: Box::new([[[0; 64]; 64]; 2]),
            pv: vec![Vec::new(); MAX_PLY + 2],
            hashes: Vec::new(),
            lmr,
            nodes: 0,
            seldepth: 0,
            stop,
            deadline: None,
            node_limit: None,
            can_abort: false,
            aborted: false,
        };
        searcher.resize(hash_mb);
        searcher
    }

    /// Table de transposition d'environ `mb` Mo (nombre d'entrées arrondi à une puissance de 2).
    pub fn resize(&mut self, mb: usize) {
        let entries = (mb.max(1) * 1024 * 1024 / std::mem::size_of::<Entry>()).max(1024);
        let size = 1usize << (usize::BITS - 1 - entries.leading_zeros());
        self.tt = vec![EMPTY_ENTRY; size];
    }

    pub fn clear(&mut self) {
        self.tt.fill(EMPTY_ENTRY);
        *self.history = [[[0; 64]; 64]; 2];
    }

    pub fn evaluate(&self, pos: &Chess) -> i32 {
        evaluate(pos, &self.tables)
    }

    fn hashfull(&self) -> u32 {
        let sample = self.tt.len().min(1000);
        (self.tt[..sample].iter().filter(|e| e.depth >= 0).count() * 1000 / sample) as u32
    }

    // --- approfondissement itératif -----------------------------------------------

    pub fn search(
        &mut self,
        root: &Chess,
        game_hashes: &[u64],
        limits: &Limits,
        multipv: usize,
        report: &mut dyn FnMut(&Info),
    ) -> Info {
        let start = Instant::now();
        self.nodes = 0;
        self.seldepth = 0;
        self.aborted = false;
        let (soft, hard) = limits.budget(root.turn().is_white());
        self.deadline = hard.map(|h| start + h);
        self.node_limit = limits.nodes;
        self.hashes = game_hashes.to_vec();
        if self.hashes.last() != Some(&hash(root)) {
            self.hashes.push(hash(root));
        }
        for killer in &mut self.killers {
            *killer = [None, None];
        }
        for color in self.history.iter_mut() {
            for row in color.iter_mut() {
                for value in row.iter_mut() {
                    *value /= 2;
                }
            }
        }

        let root_moves = root.legal_moves();
        if root_moves.is_empty() {
            return Info {
                depth: 0,
                seldepth: 0,
                lines: Vec::new(),
                nodes: 0,
                elapsed: start.elapsed(),
                hashfull: 0,
            };
        }
        let multipv = multipv.clamp(1, root_moves.len());
        let max_depth = limits.depth.unwrap_or(MAX_PLY as u32 - 2).min(MAX_PLY as u32 - 2);
        let mut result: Option<Info> = None;
        let mut previous: Vec<i32> = Vec::new();
        for depth in 1..=max_depth {
            self.can_abort = depth > 1; // la profondeur 1 va toujours à son terme
            let mut lines: Vec<Line> = Vec::new();
            let mut excluded: Vec<Move> = Vec::new();
            for index in 0..multipv {
                let guess = previous.get(index).copied();
                let (score, pv) = self.root(root, depth as i32, &excluded, guess);
                if self.aborted || pv.is_empty() {
                    break;
                }
                excluded.push(pv[0]);
                lines.push(Line { score, moves: pv });
            }
            if self.aborted || lines.len() < multipv {
                break;
            }
            lines.sort_by(|a, b| b.score.cmp(&a.score));
            previous = lines.iter().map(|line| line.score).collect();
            let info = Info {
                depth,
                seldepth: self.seldepth,
                lines,
                nodes: self.nodes,
                elapsed: start.elapsed(),
                hashfull: self.hashfull(),
            };
            report(&info);
            let best_score = info.lines[0].score;
            result = Some(info);
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            if soft.is_some_and(|soft| start.elapsed() >= soft) {
                break;
            }
            if self.node_limit.is_some_and(|limit| self.nodes >= limit) {
                break;
            }
            if let Some(mate) = mate_in(best_score)
                && !limits.infinite
                && depth as i32 >= 2 * mate.abs() + 1
            {
                break; // mat trouvé et confirmé
            }
        }
        result.expect("la profondeur 1 ne peut pas être interrompue")
    }

    fn check_time(&mut self) {
        if !self.can_abort {
            return;
        }
        if self.stop.load(Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
            || self.node_limit.is_some_and(|limit| self.nodes >= limit)
        {
            self.aborted = true;
        }
    }

    // --- racine -------------------------------------------------------------------

    fn root(
        &mut self,
        pos: &Chess,
        depth: i32,
        excluded: &[Move],
        guess: Option<i32>,
    ) -> (i32, Vec<Move>) {
        let Some(guess) = guess.filter(|g| depth >= 4 && g.abs() < MATE_BOUND) else {
            return self.root_window(pos, depth, excluded, -INF, INF);
        };
        let mut window = ASPIRATION;
        let (mut alpha, mut beta) = (guess - window, guess + window);
        loop {
            let (score, pv) = self.root_window(pos, depth, excluded, alpha, beta);
            if self.aborted {
                return (score, pv);
            }
            if score <= alpha && alpha > -INF {
                window *= 4;
                alpha = (score - window).max(-INF);
            } else if score >= beta && beta < INF {
                window *= 4;
                beta = (score + window).min(INF);
            } else {
                return (score, pv);
            }
        }
    }

    fn root_window(
        &mut self,
        pos: &Chess,
        depth: i32,
        excluded: &[Move],
        mut alpha: i32,
        beta: i32,
    ) -> (i32, Vec<Move>) {
        let original_alpha = alpha;
        let key = *self.hashes.last().expect("position racine");
        let tt_move = self.probe(key).and_then(|e| e.mv);
        let moves = self.ordered(pos, pos.legal_moves(), tt_move, 0);
        let mut best_score = -INF;
        let mut best_move = None;
        let mut best_pv = Vec::new();
        let mut first = true;
        for (_, m) in moves {
            if excluded.contains(&m) {
                continue;
            }
            let mut child = pos.clone();
            child.play_unchecked(m);
            self.hashes.push(hash(&child));
            let score = if first {
                -self.negamax(&child, depth - 1, -beta, -alpha, 1, true)
            } else {
                let mut score = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, 1, true);
                if !self.aborted && alpha < score && score < beta {
                    score = -self.negamax(&child, depth - 1, -beta, -alpha, 1, true);
                }
                score
            };
            self.hashes.pop();
            if self.aborted {
                return (0, Vec::new());
            }
            first = false;
            if score > best_score {
                best_score = score;
                best_move = Some(m);
                best_pv = std::iter::once(m).chain(self.pv[1].iter().copied()).collect();
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        if excluded.is_empty() && original_alpha < best_score && best_score < beta {
            self.store(key, depth, best_score, Bound::Exact, best_move, 0);
        }
        (best_score, best_pv)
    }

    // --- négamax --------------------------------------------------------------------

    fn negamax(
        &mut self,
        pos: &Chess,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        allow_null: bool,
    ) -> i32 {
        self.nodes += 1;
        if self.nodes & 2047 == 0 {
            self.check_time();
        }
        if self.aborted {
            return 0;
        }
        self.pv[ply].clear();
        self.seldepth = self.seldepth.max(ply as u32);

        // Nulles : règle des 50 coups, répétition (une seule suffit dans la recherche).
        if pos.halfmoves() >= 100 || self.is_repetition(pos.halfmoves()) {
            return 0;
        }
        // Élagage par distance au mat.
        alpha = alpha.max(-MATE + ply as i32);
        beta = beta.min(MATE - ply as i32 - 1);
        if alpha >= beta {
            return alpha;
        }

        let in_check = pos.is_check();
        let depth = if in_check { depth + 1 } else { depth }; // extension d'échec
        if depth <= 0 || ply >= MAX_PLY - 1 {
            return self.quiescence(pos, alpha, beta, ply);
        }

        let pv_node = beta - alpha > 1;
        let key = *self.hashes.last().expect("position courante");
        let mut tt_move = None;
        if let Some(entry) = self.probe(key) {
            tt_move = entry.mv;
            if entry.depth as i32 >= depth && !pv_node {
                let score = from_tt(entry.score, ply);
                match entry.bound {
                    Bound::Exact => return score,
                    Bound::Lower if score >= beta => return score,
                    Bound::Upper if score <= alpha => return score,
                    _ => {}
                }
            }
        }

        let mut static_eval = None;
        if !in_check && !pv_node {
            let eval = self.evaluate(pos);
            static_eval = Some(eval);
            // Élagage statique : loin au-dessus de bêta, l'adversaire évitera cette position.
            if depth <= 3 && beta.abs() < MATE_BOUND && eval - REVERSE_FUTILITY * depth >= beta {
                return eval;
            }
            // Coup nul : si passer son tour suffit encore à dépasser bêta, inutile de chercher.
            if allow_null && depth >= 3 && eval >= beta && has_pieces(pos) {
                if let Ok(child) = pos.clone().swap_turn() {
                    let reduction = 3 + depth / 6;
                    self.hashes.push(hash(&child));
                    let score =
                        -self.negamax(&child, depth - 1 - reduction, -beta, -beta + 1, ply + 1, false);
                    self.hashes.pop();
                    if self.aborted {
                        return 0;
                    }
                    if score >= beta {
                        return if score >= MATE_BOUND { beta } else { score };
                    }
                }
            }
        }
        // Élagage de frontière : près des feuilles, un coup tranquille ne rattrapera pas alpha.
        let futile = static_eval.is_some_and(|e| depth <= 2 && e + FUTILITY_MARGIN * depth <= alpha);

        let original_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move = None;
        let mut searched: i32 = 0;
        let killers = self.killers[ply];
        let moves = self.ordered(pos, pos.legal_moves(), tt_move, ply);
        for (_, m) in moves {
            let quiet = !m.is_capture() && !m.is_promotion();
            let mut child = pos.clone();
            child.play_unchecked(m);
            let gives_check = child.is_check();
            if futile && quiet && searched > 0 && !gives_check {
                continue;
            }
            // Réduction des coups tardifs : les derniers coups tranquilles, bien ordonnés,
            // sont rarement les meilleurs ; on les cherche moins loin, quitte à recommencer.
            let mut reduction = 0;
            if depth >= 3
                && searched >= 3
                && quiet
                && !in_check
                && !gives_check
                && !killers.contains(&Some(m))
            {
                reduction = self.lmr[depth.min(63) as usize][searched.min(63) as usize];
                if pv_node {
                    reduction -= 1;
                }
                reduction = reduction.clamp(0, depth - 2);
            }
            self.hashes.push(hash(&child));
            let score = if searched == 0 {
                -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, true)
            } else {
                let mut score =
                    -self.negamax(&child, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, true);
                if !self.aborted && reduction > 0 && score > alpha {
                    score = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, ply + 1, true);
                }
                if !self.aborted && pv_node && alpha < score && score < beta {
                    score = -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, true);
                }
                score
            };
            self.hashes.pop();
            if self.aborted {
                return 0;
            }
            searched += 1;
            if score > best_score {
                best_score = score;
                best_move = Some(m);
                if score > alpha {
                    alpha = score;
                    let mut line = Vec::with_capacity(self.pv[ply + 1].len() + 1);
                    line.push(m);
                    line.extend_from_slice(&self.pv[ply + 1]);
                    self.pv[ply] = line;
                    if alpha >= beta {
                        if quiet {
                            self.remember_quiet(pos, m, depth, ply);
                        }
                        break;
                    }
                }
            }
        }
        if searched == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 }; // mat ou pat
        }
        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store(key, depth, best_score, bound, best_move, ply);
        best_score
    }

    fn quiescence(&mut self, pos: &Chess, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.nodes & 2047 == 0 {
            self.check_time();
        }
        if self.aborted {
            return 0;
        }
        self.pv[ply].clear();
        self.seldepth = self.seldepth.max(ply as u32);
        if ply >= MAX_PLY - 1 {
            return self.evaluate(pos);
        }
        let in_check = pos.is_check();
        let mut best;
        let moves;
        if in_check {
            // En échec, pas de « statu quo » : toutes les parades.
            best = -MATE + ply as i32;
            moves = pos.legal_moves();
        } else {
            let stand_pat = self.evaluate(pos);
            if stand_pat >= beta {
                return stand_pat;
            }
            alpha = alpha.max(stand_pat);
            best = stand_pat;
            let mut noisy = pos.capture_moves();
            for m in pos.promotion_moves() {
                if !m.is_capture() && m.promotion() == Some(Role::Queen) {
                    noisy.push(m);
                }
            }
            moves = noisy;
        }
        for (_, m) in self.ordered(pos, moves, None, ply) {
            if !in_check && m.promotion().is_none() {
                let victim = m.capture().map_or(0, |role| PIECE_VALUES[role_index(role)]);
                if best + victim + DELTA_MARGIN < alpha {
                    continue; // même gagnée, cette prise ne rattraperait pas alpha
                }
            }
            let mut child = pos.clone();
            child.play_unchecked(m);
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    // --- ordre des coups --------------------------------------------------------------

    fn ordered(
        &self,
        pos: &Chess,
        moves: MoveList,
        tt_move: Option<Move>,
        ply: usize,
    ) -> Vec<(i32, Move)> {
        let killers = self.killers[ply.min(MAX_PLY + 1)];
        let color = if pos.turn().is_white() { 0 } else { 1 };
        let mut scored: Vec<(i32, Move)> = moves
            .into_iter()
            .map(|m| {
                let score = if Some(m) == tt_move {
                    30_000_000
                } else if let Some(victim) = m.capture() {
                    20_000_000 + 10 * ORDER_VALUES[role_index(victim)]
                        - ORDER_VALUES[role_index(m.role())]
                } else if let Some(promotion) = m.promotion() {
                    19_000_000 + ORDER_VALUES[role_index(promotion)]
                } else if Some(m) == killers[0] {
                    18_000_000
                } else if Some(m) == killers[1] {
                    17_000_000
                } else {
                    let from = m.from().map_or(0, |s| s.to_usize());
                    self.history[color][from][m.to().to_usize()]
                };
                (score, m)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        scored
    }

    fn remember_quiet(&mut self, pos: &Chess, m: Move, depth: i32, ply: usize) {
        let killers = &mut self.killers[ply];
        if killers[0] != Some(m) {
            killers[1] = killers[0];
            killers[0] = Some(m);
        }
        let color = if pos.turn().is_white() { 0 } else { 1 };
        let from = m.from().map_or(0, |s| s.to_usize());
        let value = &mut self.history[color][from][m.to().to_usize()];
        *value = (*value + depth * depth).min(1_000_000);
    }

    // --- répétitions et table de transposition ---------------------------------------

    fn is_repetition(&self, halfmoves: u32) -> bool {
        let n = self.hashes.len();
        let current = self.hashes[n - 1];
        let limit = (halfmoves as usize).min(n - 1);
        let mut distance = 2;
        while distance <= limit {
            if self.hashes[n - 1 - distance] == current {
                return true;
            }
            distance += 2;
        }
        false
    }

    fn probe(&self, key: u64) -> Option<Entry> {
        let entry = self.tt[key as usize & (self.tt.len() - 1)];
        (entry.key == key && entry.depth >= 0).then_some(entry)
    }

    fn store(&mut self, key: u64, depth: i32, score: i32, bound: Bound, mv: Option<Move>, ply: usize) {
        let index = key as usize & (self.tt.len() - 1);
        let slot = &mut self.tt[index];
        // Remplacement : position différente, ou recherche au moins aussi profonde.
        if slot.key != key || depth as i16 >= slot.depth || bound == Bound::Exact {
            *slot = Entry {
                key,
                mv: mv.or(if slot.key == key { slot.mv } else { None }),
                score: to_tt(score, ply),
                depth: depth as i16,
                bound,
            };
        }
    }
}

fn has_pieces(pos: &Chess) -> bool {
    let board = pos.board();
    let pieces = board.knights() | board.bishops() | board.rooks() | board.queens();
    (pieces & board.by_color(pos.turn())).any()
}

/// Score de mat compté depuis le nœud stocké, et non depuis la racine.
fn to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

fn from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}
