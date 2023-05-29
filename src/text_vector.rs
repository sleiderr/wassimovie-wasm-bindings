use hora::core::ann_index::ANNIndex;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct TextVector {
    tv: Vec<f32>,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct TextSpace {
    vecs: Vec<TextVector>,
    index: hora::index::hnsw_idx::HNSWIndex<f32, usize>,
}

#[wasm_bindgen]
impl TextSpace {
    pub fn new(dimension: usize) -> Self {
        Self {
            vecs: vec![],
            index: hora::index::hnsw_idx::HNSWIndex::new(
                dimension,
                &hora::index::hnsw_params::HNSWParams::<f32>::default(),
            ),
        }
    }

    pub fn load(raw_json: &str) -> Self {
        serde_json::from_str(raw_json).expect("Failed to load on-storage space")
    }

    pub fn reload_index(&mut self) {
        self.index
            .build(hora::core::metrics::Metric::CosineSimilarity)
            .expect("Failed to reload index");
    }

    pub fn add_vector(&mut self, vec: TextVector, should_reload: bool) -> usize {
        self.index
            .add(&vec.tv, self.vecs.len())
            .expect("Failed to add vector to index");
        self.vecs.push(vec);
        if should_reload {
            self.reload_index();
        }
        self.vecs.len() - 1
    }

    pub fn evaluate(&self, vec: &TextVector) -> f32 {
        todo!();
    }

    pub fn query(&mut self, vec: TextVector, count: usize) -> Vec<usize> {
        if !self.index.built() {
            self.reload_index();
        }
        self.index.search(&vec.tv, count)
    }
}
