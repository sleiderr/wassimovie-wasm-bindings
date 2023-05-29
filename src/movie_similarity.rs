use std::cmp::max;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::text_vector::TextVector;

pub struct MovieVector {
    pub(crate) uuid: uuid::Uuid,
    pub(crate) actors: Vec<ActorVector>,
    pub(crate) description: TextVector,
    pub(crate) genres: Vec<Genre>,
    pub(crate) original_lang: Language,
    pub(crate) popularity: usize,
    pub(crate) title: String,
    pub(crate) vote_avg: usize,
    pub(crate) vote_count: usize,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Clone, Copy)]
pub enum Genre {
    Drama,
    Comedy,
    Crime,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Language {
    English,
    French,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd)]
pub struct ActorVector {
    pub(crate) id: Uuid,
    popularity: usize,
}

pub const POPULARITY_WEIGHT: f32 = 3_f32;
pub const VOTE_WEIGHT: f32 = 3_f32;
pub const ACTORS_WEIGHT: f32 = 12_f32;
pub const GENRES_WEIGHT: f32 = 9_f32;
pub const LANG_WEIGHT: f32 = 4_f32;
pub const DESC_WEIGHT: f32 = 13_f32;

pub fn similarity(a: MovieVector, b: MovieVector) -> f32 {
    let pop_score = (1 - (a.popularity - b.popularity) / max(a.popularity, b.popularity)) as f32;
    let actors_score = _actors_similarity(&a.actors, &b.actors);
    let genres_scores = _genres_similarity(&a.genres, &b.genres);
    let lang_score = if a.original_lang == b.original_lang {
        1_f32
    } else {
        0_f32
    };
    let desc_score: f32 = 0 as f32;

    (pop_score * POPULARITY_WEIGHT
        + actors_score * ACTORS_WEIGHT
        + genres_scores * GENRES_WEIGHT
        + lang_score * LANG_WEIGHT
        + desc_score * DESC_WEIGHT)
        / (POPULARITY_WEIGHT + ACTORS_WEIGHT + GENRES_WEIGHT + LANG_WEIGHT + DESC_WEIGHT)
}

fn _actors_similarity(a: &Vec<ActorVector>, b: &Vec<ActorVector>) -> f32 {
    let inter_card = _inter_cardinality(a, b) as f32;
    inter_card / ((a.len() + b.len()) as f32 - inter_card)
}

fn _genres_similarity(a: &Vec<Genre>, b: &Vec<Genre>) -> f32 {
    _inter_cardinality(a, b) as f32 / (a.len() * b.len()) as f32
}

fn _inter_cardinality<T>(a: &[T], b: &[T]) -> usize
where
    T: PartialOrd,
    T: PartialEq,
{
    let mut count = 0;
    let mut b_iter = b.iter();

    if let Some(mut curr_b) = b_iter.next() {
        for curr_a in a {
            while curr_b < curr_a {
                curr_b = match b_iter.next() {
                    Some(item) => item,
                    None => return count,
                };
            }
            if curr_a == curr_b {
                count += 1;
            }
        }
    }
    count
}
