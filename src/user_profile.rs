use std::{
    cmp::{max, Ordering},
    collections::{hash_map::Entry, HashMap},
};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use wasm_bindgen::prelude::*;

use crate::{
    movie_similarity::{self, Genre, Language, MovieVector},
    text_vector::TextSpace,
};

type ActorMap = IndexMap<Uuid, f32>;
type GenreMap = IndexMap<Genre, f32>;
type LanguageMap = HashMap<Language, f32>;

macro_rules! _vectorial_insertion {
    ($name:tt, $rank:tt, $v:tt: $t:ty ) => {
        #[allow(clippy::ptr_arg)]
        fn $name(&mut self, $v: $t, inc_weight: f32) {
            $v.iter()
                .map(|elem| match self.$rank.entry(*elem) {
                    indexmap::map::Entry::Occupied(e) => {
                        *(e.into_mut()) += (inc_weight / $v.len() as f32)
                    }
                    indexmap::map::Entry::Vacant(v) => {
                        v.insert(inc_weight / $v.len() as f32);
                    }
                })
                .last();
        }
    };
}

#[derive(Serialize, Deserialize)]
#[wasm_bindgen]
pub struct UserProfile {
    actors_rank: ActorMap,
    genre_rank: GenreMap,
    lang_rank: LanguageMap,
    text_rank: TextSpace,
    _queued_weight: f32,
    _profile_weight: f32,
    _score_cache: HashMap<Uuid, f32>,
}

#[wasm_bindgen]
impl UserProfile {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            actors_rank: IndexMap::new(),
            genre_rank: IndexMap::new(),
            lang_rank: HashMap::new(),
            text_rank: TextSpace::new(512),
            _queued_weight: 0_f32,
            _profile_weight: 0_f32,
            _score_cache: HashMap::new(),
        }
    }

    pub fn load(raw_json: &str) -> Self {
        serde_json::from_str(raw_json).expect("Failed to load user profile !")
    }

    pub async fn insert_movie_interaction(&mut self, a: &MovieVector, interaction_weight: f32) {
        self._vectorial_insertion_actor(
            &a.actors.iter().map(|act| act.id).collect(),
            interaction_weight,
        );
        self._vectorial_insertion_genre(&a.genres, interaction_weight);
        self._vectorial_insertion_lang(&a.original_lang, interaction_weight);

        let invalidated = self._invalidate_cache(interaction_weight).await;

        self.text_rank
            .add_vector(a.description.clone(), invalidated);
    }

    async fn _invalidate_cache(&mut self, incoming_weight: f32) -> bool {
        let mut relative_inc_weight =
            (self._profile_weight - (self._queued_weight + incoming_weight)) / self._profile_weight;
        if self._profile_weight == 0_f32 {
            relative_inc_weight = 1_f32;
        }

        if let Ordering::Greater = relative_inc_weight.abs().partial_cmp(&0.05_f32).unwrap() {
            self._queued_weight = 0_f32;
            self._score_cache = HashMap::default();
            self.text_rank.reload_index();
            let idx_db = crate::indexed_db::load_db()
                .await
                .expect("Failed to load database");
            crate::indexed_db::save_profile(&idx_db, "profile", self)
                .await
                .expect("Failed to save profile");

            return true;
        }
        self._queued_weight += incoming_weight;
        false
    }

    #[allow(clippy::ptr_arg)]
    fn _vectorial_insertion_actor(&mut self, a: &Vec<Uuid>, inc_weight: f32) {
        a.iter()
            .map(|elem| match self.actors_rank.entry(*elem) {
                indexmap::map::Entry::Occupied(e) => *(e.into_mut()) += inc_weight / a.len() as f32,
                indexmap::map::Entry::Vacant(v) => {
                    v.insert(inc_weight / a.len() as f32);
                }
            })
            .last();
    }

    #[allow(clippy::ptr_arg)]
    fn _vectorial_insertion_genre(&mut self, a: &[Genre], inc_weight: f32) {
        a.iter()
            .map(|elem| match self.genre_rank.entry(*elem) {
                indexmap::map::Entry::Occupied(e) => *(e.into_mut()) += inc_weight / a.len() as f32,
                indexmap::map::Entry::Vacant(v) => {
                    v.insert(inc_weight / a.len() as f32);
                }
            })
            .last();
    }

    fn _vectorial_insertion_lang(&mut self, a: &Language, inc_weight: f32) {
        match self.lang_rank.entry(*a) {
            Entry::Occupied(e) => *(e.into_mut()) += inc_weight,
            Entry::Vacant(v) => {
                v.insert(inc_weight);
            }
        };
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
        let actor_score = self._vector_similarity::<Uuid>(
            &a.actors.iter().map(|act| act.id).collect(),
            &self.actors_rank,
        ) * movie_similarity::ACTORS_WEIGHT;
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

impl UserProfile {
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
}

impl UserProfile {
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
}
