use hora::core::ann_index::ANNIndex;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

const BETA: f32 = 1.2;

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone)]
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
        let vec_count = (self.vecs.len() as f32 * 0.2) as usize + 1;

        let nn: Vec<f32> = self
            .index
            .search_nodes(&vec.tv, vec_count)
            .iter()
            .map(|x| x.1)
            .collect();

        let mut metric_1 = 0_f32;
        let mut i = 1;
        nn.iter()
            .map(|f| {
                i += 1;
                metric_1 += f / (i as f32).log2();
            })
            .last();

        let mut metric_2 = 0_f32;
        nn.iter().map(|f| metric_2 += f).last();
        metric_2 /= vec_count as f32;

        (1_f32 + BETA.powi(2)) * (metric_1 * metric_2) / (BETA.powi(2) * metric_2 + metric_1)
    }

    pub fn query(&mut self, vec: TextVector, count: usize) -> Vec<usize> {
        if !self.index.built() {
            self.reload_index();
        }
        self.index.search(&vec.tv, count)
    }
}
