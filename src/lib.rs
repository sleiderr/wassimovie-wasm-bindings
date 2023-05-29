#![feature(entry_insert)]

pub mod indexed_db;
pub mod movie_similarity;
pub mod text_vector;
pub mod user_profile;

use user_profile::UserProfile;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    web_sys::console::log_1(&JsValue::from_str("Welcome to WassiMovie !"));
}

#[wasm_bindgen]
pub async fn load_profile() -> UserProfile {
    let idx_db = indexed_db::load_db()
        .await
        .expect("Failed to load database");
    indexed_db::get_profile(&idx_db, "profile")
        .await
        .unwrap_or_default()
}
