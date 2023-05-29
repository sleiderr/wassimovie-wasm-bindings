use std::{
    cmp::max,
    collections::{hash_map::Entry, HashMap},
};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    movie_similarity::{self, ActorVector, Genre, Language, MovieVector},
    text_vector::TextSpace,
};

type ActorMap = IndexMap<ActorVector, f32>;
type GenreMap = IndexMap<Genre, f32>;
type LanguageMap = HashMap<Language, f32>;

#[derive(Serialize, Deserialize)]
pub struct UserProfile {
    actors_rank: ActorMap,
    genre_rank: GenreMap,
    lang_rank: LanguageMap,
    text_rank: TextSpace,
    _size: usize,
    _counter: usize,
    _score_cache: HashMap<Uuid, f32>,
}

impl UserProfile {
    pub fn new() -> Self {
        Self {
            actors_rank: IndexMap::new(),
            genre_rank: IndexMap::new(),
            lang_rank: HashMap::new(),
            text_rank: TextSpace::new(512),
            _size: 0,
            _counter: 0,
            _score_cache: HashMap::new(),
        }
    }

    pub fn load(raw_json: &str) -> Self {
        serde_json::from_str(raw_json).expect("Failed to load user profile !")
    }

    pub fn rank(&mut self, inputs: &mut [&MovieVector]) {
        let mut score_map = HashMap::new();
        inputs.sort_by(|a, b| {
            let sim_a = match score_map.entry(a.uuid) {
                Entry::Occupied(o) => *(o.into_mut()),
                Entry::Vacant(v) => *(v.insert(self.similarity(a))),
            };
            let sim_b = match score_map.entry(b.uuid) {
                Entry::Occupied(o) => o.into_mut(),
                Entry::Vacant(v) => v.insert(self.similarity(b)),
            };
            sim_a.partial_cmp(sim_b).unwrap()
        });
        self._score_cache.extend(score_map.into_iter());
    }

    #[allow(clippy::map_entry)]
    pub fn similarity(&mut self, a: &MovieVector) -> f32 {
        if self._score_cache.contains_key(&a.uuid) {
            *self._score_cache.get(&a.uuid).unwrap()
        } else {
            let new_val = self._similarity_calc(a);
            self._score_cache.insert(a.uuid, new_val);
            new_val
        }
    }

    fn _similarity_calc(&self, a: &MovieVector) -> f32 {
        let actor_score = self._vector_similarity::<ActorVector>(&a.actors, &self.actors_rank)
            * movie_similarity::ACTORS_WEIGHT;
        let genre_score = self._vector_similarity::<Genre>(&a.genres, &self.genre_rank)
            * movie_similarity::GENRES_WEIGHT;
        let lang_score = self._movie_similarity_by_lang(a) * movie_similarity::LANG_WEIGHT;

        let text_rank = self.text_rank.evaluate(&a.description) * movie_similarity::DESC_WEIGHT;

        (actor_score + genre_score + lang_score + text_rank)
            / (movie_similarity::ACTORS_WEIGHT
                + movie_similarity::GENRES_WEIGHT
                + movie_similarity::LANG_WEIGHT
                + movie_similarity::DESC_WEIGHT)
    }

    fn _vector_similarity<T: PartialEq + Eq + std::hash::Hash + indexmap::Equivalent<T>>(
        &self,
        a: &Vec<T>,
        b: &IndexMap<T, f32>,
    ) -> f32 {
        let mut scalar_x = 0_f32;
        let mov_act_l2_norm_sq = a.len();

        let k = max(mov_act_l2_norm_sq, b.len());
        let mut max_norm_for_k = 0_f32;
        {
            (0..k)
                .map(|idx| {
                    b.get_index(k).unwrap_or_else(|| {
                        panic!(
                            "Failed to retrieve data for index {}. This is not normal behavior",
                            idx
                        )
                    })
                })
                .map(|(_, score)| {
                    max_norm_for_k += score;
                })
                .last();
        }

        a.iter()
            .map(|act| scalar_x += b.get(act).unwrap_or(&0_f32))
            .last();

        scalar_x / max_norm_for_k
    }

    fn _movie_similarity_by_lang(&self, a: &MovieVector) -> f32 {
        *self.lang_rank.get(&a.original_lang).unwrap_or(&0_f32)
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self::new()
    }
}
