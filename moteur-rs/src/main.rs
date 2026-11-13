//! Chessio Moteur (lot L4) : moteur d'échecs maison en Rust, protocole UCI.
//!
//! Même logique que le moteur Python v0.2 (`chessio/moteur`), avec la génération
//! de coups par bitboards et tables d'attaque « magiques » de shakmaty et le
//! hachage de Zobrist. La recherche tourne dans un fil à part : le moteur répond
//! à tout moment à `isready` et s'arrête sur `stop`. `go perft N` détaille le
//! perft par coup, comme Stockfish.

mod eval;
mod search;

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position, perft};

use search::{Info, Limits, Searcher, hash, mate_in};

const NAME: &str = "Chessio Moteur";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_HASH_MB: usize = 64;

fn uci_score(score: i32) -> String {
    match mate_in(score) {
        Some(mate) => format!("mate {mate}"),
        None => format!("cp {score}"),
    }
}

fn print_info(info: &Info) {
    let ms = info.elapsed.as_millis().max(1);
    let nps = info.nodes as u128 * 1000 / ms;
    for (index, line) in info.lines.iter().enumerate() {
        let pv: Vec<String> = line
            .moves
            .iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect();
        println!(
            "info depth {} seldepth {} multipv {} score {} nodes {} nps {} hashfull {} time {} pv {}",
            info.depth,
            info.seldepth,
            index + 1,
            uci_score(line.score),
            info.nodes,
            nps,
            info.hashfull,
            ms,
            pv.join(" ")
        );
    }
}

fn parse_limits(tokens: &[&str]) -> Limits {
    let mut limits = Limits::default();
    let mut index = 0;
    while index < tokens.len() {
        let value = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok());
        let ms = value.map(Duration::from_millis);
        match tokens[index] {
            "infinite" => limits.infinite = true,
            "depth" => limits.depth = value.map(|v| v as u32),
            "nodes" => limits.nodes = value,
            "movetime" => limits.movetime = ms,
            "wtime" => limits.wtime = ms,
            "btime" => limits.btime = ms,
            "winc" => limits.winc = ms.unwrap_or_default(),
            "binc" => limits.binc = ms.unwrap_or_default(),
            "movestogo" => limits.movestogo = value.map(|v| v as u32),
            _ => {}
        }
        index += 1;
    }
    limits
}

/// « position startpos|fen … [moves …] » : la position et les hachages de la partie.
fn parse_position(tokens: &[&str]) -> Option<(Chess, Vec<u64>)> {
    let moves_at = tokens.iter().position(|t| *t == "moves");
    let (setup, moves) = match moves_at {
        Some(i) => (&tokens[..i], &tokens[i + 1..]),
        None => (tokens, &[][..]),
    };
    let mut pos = match setup.first()? {
        &"startpos" => Chess::default(),
        &"fen" => {
            let fen: Fen = setup[1..].join(" ").parse().ok()?;
            fen.into_position(CastlingMode::Standard).ok()?
        }
        _ => return None,
    };
    let mut hashes = vec![hash(&pos)];
    for text in moves {
        let m = UciMove::from_ascii(text.as_bytes()).ok()?.to_move(&pos).ok()?;
        pos.play_unchecked(m);
        hashes.push(hash(&pos));
    }
    Some((pos, hashes))
}

fn perft_divide(pos: &Chess, depth: u32) {
    let start = Instant::now();
    let mut total = 0;
    for m in pos.legal_moves() {
        let mut child = pos.clone();
        child.play_unchecked(m);
        let count = if depth > 1 { perft(&child, depth - 1) } else { 1 };
        total += count;
        println!("{}: {count}", m.to_uci(CastlingMode::Standard));
    }
    println!();
    println!("Nodes searched: {total}");
    println!("Time: {} ms", start.elapsed().as_millis());
}

struct Engine {
    searcher: Arc<Mutex<Searcher>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    position: Chess,
    hashes: Vec<u64>,
    multipv: usize,
}

impl Engine {
    fn new() -> Engine {
        let stop = Arc::new(AtomicBool::new(false));
        let start = Chess::default();
        Engine {
            searcher: Arc::new(Mutex::new(Searcher::new(DEFAULT_HASH_MB, Arc::clone(&stop)))),
            stop,
            thread: None,
            hashes: vec![hash(&start)],
            position: start,
            multipv: 1,
        }
    }

    fn stop_search(&mut self) {
        if let Some(handle) = self.thread.take() {
            self.stop.store(true, Ordering::Relaxed);
            let _ = handle.join();
        }
        self.stop.store(false, Ordering::Relaxed);
    }

    fn go(&mut self, limits: Limits) {
        let searcher = Arc::clone(&self.searcher);
        let stop = Arc::clone(&self.stop);
        let position = self.position.clone();
        let hashes = self.hashes.clone();
        let multipv = self.multipv;
        self.thread = Some(thread::spawn(move || {
            let mut searcher = searcher.lock().expect("moteur");
            let info = searcher.search(&position, &hashes, &limits, multipv, &mut print_info);
            if limits.infinite {
                // UCI : en analyse infinie, « bestmove » n'est envoyé qu'après « stop ».
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(5));
                }
            }
            match info.best() {
                Some(best) => {
                    let ponder = info.lines[0].moves.get(1);
                    match ponder {
                        Some(p) => println!(
                            "bestmove {} ponder {}",
                            best.to_uci(CastlingMode::Standard),
                            p.to_uci(CastlingMode::Standard)
                        ),
                        None => println!("bestmove {}", best.to_uci(CastlingMode::Standard)),
                    }
                }
                None => println!("bestmove 0000"),
            }
        }));
    }

    /// Traite une commande ; renvoie `false` sur « quit ».
    fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&command, args)) = tokens.split_first() else {
            return true;
        };
        match command {
            "uci" => {
                println!("id name {NAME} {VERSION}");
                println!("id author Chessio");
                println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 4096");
                println!("option name MultiPV type spin default 1 min 1 max 64");
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                self.stop_search();
                self.searcher.lock().expect("moteur").clear();
            }
            "setoption" => {
                self.stop_search();
                let text = args.join(" ");
                if let Some((name, value)) = text
                    .strip_prefix("name ")
                    .and_then(|rest| rest.split_once(" value "))
                {
                    let value = value.trim().parse::<usize>().unwrap_or(1);
                    match name.trim().to_lowercase().as_str() {
                        "hash" => self.searcher.lock().expect("moteur").resize(value),
                        "multipv" => self.multipv = value.clamp(1, 64),
                        _ => {}
                    }
                }
            }
            "position" => {
                self.stop_search();
                if let Some((position, hashes)) = parse_position(args) {
                    self.position = position;
                    self.hashes = hashes;
                }
            }
            "go" => {
                self.stop_search();
                if args.first() == Some(&"perft") {
                    let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(1);
                    perft_divide(&self.position, depth);
                } else {
                    self.go(parse_limits(args));
                }
            }
            "stop" => self.stop_search(),
            "eval" => {
                let score = self.searcher.lock().expect("moteur").evaluate(&self.position);
                println!("eval {score}");
            }
            "quit" => {
                self.stop_search();
                return false;
            }
            _ => {}
        }
        true
    }
}

fn main() {
    let mut engine = Engine::new();
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if !engine.handle(line.trim()) {
            break;
        }
    }
    engine.stop_search();
}
